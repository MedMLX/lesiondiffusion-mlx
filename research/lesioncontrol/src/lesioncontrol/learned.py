"""Portable learned radiomic context adapters for the production MLX generator.

Checkpoints contain only the small FP32 MLP and training-only feature scaling.
Frozen LINet and VQ weights remain in the separately supplied model bundle.
Feature and context ordering are part of the checkpoint's scientific contract.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import mlx.core as mx
import numpy as np
from lesiondiffusion_mlx.medmlx import load_stage

from lesioncontrol.adapter import DifferentiableLINet, RadiomicContextAdapter
from lesioncontrol.backend import require_metal
from lesioncontrol.profile import default_settings

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Mlx


class _Evaluator(Protocol):
    def eval(self, *arrays: object) -> None: ...


_evaluate = cast(_Evaluator, mx).eval
SCHEMA = "lesioncontrol.radiomic_adapter.v1"
CONTEXT_FIELDS = (
    "enhancement status",
    "lesion location",
    "shape",
    "density",
    "density variations",
    "CT value",
    "surface characteristics",
    "relationship with adjacent organs",
    "specific features",
    "cavity",
)


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    fields = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in fields):
        raise ValueError(f"{name} must have string keys")
    return cast(dict[str, object], fields)


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of strings")
    entries = cast(list[object], value)
    if any(not isinstance(entry, str) for entry in entries):
        raise ValueError(f"{name} must be a list of strings")
    return tuple(cast(list[str], entries))


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _numbers(value: object, name: str, count: int) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a numeric list")
    entries = cast(list[object], value)
    if len(entries) != count:
        raise ValueError(f"{name} must contain exactly {count} numbers")
    return tuple(_number(entry, name) for entry in entries)


@dataclass(frozen=True)
class TrainedRadiomicAdapter:
    """Loaded MLP and immutable feature ordering/scales fitted on training patients."""

    checkpoint: Path
    network: RadiomicContextAdapter
    feature_names: tuple[str, ...]
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    metadata: dict[str, object]

    @property
    def target_space(self) -> str:
        return cast(str, self.metadata["target_space"])

    def normalize(self, features: Mapping[str, float]) -> mx.array:
        """Normalize one profile in the declared training space without clipping."""
        require_metal()
        if set(features) != set(self.feature_names):
            raise ValueError("Target features differ from the adapter's ordered feature contract")
        values = [_number(features[name], name) for name in self.feature_names]
        array = mx.array([values], dtype=mx.float32)
        center = mx.array([self.feature_center], dtype=mx.float32)
        scale = mx.array([self.feature_scale], dtype=mx.float32)
        return (array - center) / scale

    def bind(
        self,
        features: Mapping[str, float],
        *,
        target_space: str,
        gain: float = 1.0,
    ) -> BoundRadiomicConditioner:
        """Bind a supplied feature target and explicit nonnegative adapter gain."""
        if target_space != self.target_space:
            raise ValueError(f"This checkpoint requires {self.target_space} feature targets")
        if not math.isfinite(gain) or gain < 0:
            raise ValueError("Adapter gain must be finite and nonnegative")
        normalized = self.normalize(features)
        _evaluate(normalized)
        if not bool(mx.all(mx.isfinite(normalized)).item()):
            raise ValueError("Target features overflow the adapter's float32 normalization")
        return BoundRadiomicConditioner(self, normalized, gain)


@dataclass(frozen=True)
class BoundRadiomicConditioner:
    """One target's context transform, evaluated before frozen LINet K/V caching."""

    adapter: TrainedRadiomicAdapter
    normalized_features: mx.array
    gain: float

    @property
    def conditioning_schema(self) -> str:
        return "training"

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "checkpoint": str(self.adapter.checkpoint),
            "target_space": self.adapter.target_space,
            "conditioning_schema": self.conditioning_schema,
            "feature_count": len(self.adapter.feature_names),
            "gain": self.gain,
            "normalization": "training-only mean/std; constant scales are one; no clipping",
        }

    def __call__(self, context: mx.array, attributes: Mapping[str, object]) -> mx.array:
        if tuple(attributes) != CONTEXT_FIELDS:
            raise ValueError("LINet context fields differ from the adapter's training order")
        if context.shape != (1, len(CONTEXT_FIELDS), 512):
            raise ValueError("Learned conditioning requires one 10-token, 512-dimensional context")
        if self.gain == 0:
            return context
        # Adding the per-token residual before context projection also refreshes
        # every cached cross-attention key/value in prepare_denoiser.
        adapted = self.adapter.network(self.normalized_features, context)
        result = adapted if self.gain == 1 else context + self.gain * (adapted - context)
        _evaluate(result)
        if not bool(mx.all(mx.isfinite(result)).item()):
            raise ValueError("Learned adapter produced non-finite LINet conditioning")
        return result


def load_adapter(checkpoint: Path) -> TrainedRadiomicAdapter:
    """Load the strict CUDA/MLX-portable FP32 radiomic adapter checkpoint."""
    require_metal()
    checkpoint = checkpoint.expanduser().resolve()
    payload: object = json.loads((checkpoint / "metadata.json").read_text())
    metadata = _object(payload, "adapter metadata")
    if metadata.get("schema") != SCHEMA:
        raise ValueError("Unsupported learned adapter schema")
    if metadata.get("target_space") not in ("absolute_hu", "liver_contrast"):
        raise ValueError("Adapter must declare absolute_hu or liver_contrast training features")
    if metadata.get("conditioning_schema") != "training":
        raise ValueError("Adapter requires the pinned training conditioning schema")
    if _strings(metadata.get("context_fields"), "context_fields") != CONTEXT_FIELDS:
        raise ValueError("Adapter context fields differ from the pinned training order")
    for key, expected in (("token_count", 10), ("hidden_dim", 64)):
        if type(metadata.get(key)) is not int or metadata[key] != expected:
            raise ValueError(f"Adapter {key} must be {expected}")
    names = _strings(metadata.get("feature_names"), "feature_names")
    expected_names = {f"original_{name}" for name in default_settings().features}
    if len(names) != 101 or len(set(names)) != 101 or set(names) != expected_names:
        raise ValueError("Adapter must declare all 101 appearance features exactly once")
    center = _numbers(metadata.get("feature_center"), "feature_center", 101)
    scale = _numbers(metadata.get("feature_scale"), "feature_scale", 101)
    if any(value <= 0 for value in scale):
        raise ValueError("Feature scales must be positive; training constants use scale one")
    parameters = cast(dict[str, mx.array], mx.load(str(checkpoint / "adapter.safetensors")))
    shapes = {
        "hidden.weight": (64, 101),
        "hidden.bias": (64,),
        "output.weight": (5120, 64),
        "output.bias": (5120,),
    }
    if set(parameters) != set(shapes):
        raise ValueError("Adapter weight keys differ from the portable MLP architecture")
    for name, shape in shapes.items():
        value = parameters[name]
        if value.shape != shape or value.dtype != mx.float32:
            raise ValueError(f"{name} requires shape {shape} and float32 weights")
        if not bool(mx.all(mx.isfinite(value)).item()):
            raise ValueError(f"Adapter weight {name} contains non-finite values")
    network = RadiomicContextAdapter(token_count=10)
    # Torch and MLX Linear both store (out_features, in_features), so no transpose
    # or conversion of the trained MLP parameters is needed.
    network.hidden.weight = parameters["hidden.weight"]
    network.hidden.bias = parameters["hidden.bias"]
    network.output.weight = parameters["output.weight"]
    network.output.bias = parameters["output.bias"]
    _evaluate(*parameters.values())
    return TrainedRadiomicAdapter(checkpoint, network, names, center, scale, metadata)


def parity_preflight(
    checkpoint: Path, fixture: Path, base_weights: Path, output: Path
) -> dict[str, object]:
    """Compare a real CUDA fixture with the loaded MLP and frozen FP32 MLX LINet.

    The CUDA producer supplies matching real latent/context inputs, normalization,
    adapted context and prediction after a nonzero adapter update. The acceptance
    tolerances reuse the native project's FP32 source-model comparison contract.
    """
    adapter = load_adapter(checkpoint)
    arrays = cast(dict[str, mx.array], mx.load(str(fixture)))
    required = {
        "raw_features",
        "normalized_features",
        "base_context",
        "adapted_context",
        "input",
        "timestep",
        "prediction",
    }
    if not required.issubset(arrays):
        raise ValueError("CUDA parity fixture is missing its adapter/model inputs or outputs")
    raw = np.array(arrays["raw_features"], dtype=np.float32)
    if raw.shape != (1, 101) or not np.isfinite(raw).all():
        raise ValueError("CUDA parity fixture requires a finite (1,101) raw feature vector")
    features = {name: float(raw[0, index]) for index, name in enumerate(adapter.feature_names)}
    bound = adapter.bind(features, target_space=adapter.target_space)
    context = arrays["base_context"]
    transformed = bound(context, dict.fromkeys(CONTEXT_FIELDS, ""))
    frozen = load_stage(base_weights, "linet", cast("Mlx", mx), dtype=mx.float32)
    model = DifferentiableLINet(frozen)
    prediction = model(arrays["input"], arrays["timestep"], transformed)
    _evaluate(bound.normalized_features, transformed, prediction)

    def difference(
        actual: mx.array, expected: mx.array, atol: float, rtol: float
    ) -> dict[str, object]:
        observed = np.array(actual, dtype=np.float32)
        reference = np.array(expected, dtype=np.float32)
        if observed.shape != reference.shape:
            raise ValueError("CUDA and MLX parity output shapes differ")
        delta = observed.astype(np.float64) - reference.astype(np.float64)
        return {
            "passed": bool(np.isfinite(observed).all() and np.isfinite(reference).all())
            and bool(np.allclose(observed, reference, atol=atol, rtol=rtol)),
            "atol": atol,
            "rtol": rtol,
            "max_absolute_error": float(np.max(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
        }

    comparisons = {
        "normalized_features": difference(
            bound.normalized_features, arrays["normalized_features"], 2e-6, 2e-5
        ),
        "adapted_context": difference(transformed, arrays["adapted_context"], 2e-6, 2e-5),
        "noise_prediction": difference(prediction, arrays["prediction"], 3e-5, 3e-4),
    }
    if "base_prediction" in arrays:
        baseline = model(arrays["input"], arrays["timestep"], context)
        _evaluate(baseline)
        comparisons["base_noise_prediction"] = difference(
            baseline, arrays["base_prediction"], 3e-5, 3e-4
        )
    passed = all(row["passed"] is True for row in comparisons.values())
    report: dict[str, object] = {
        "passed": passed,
        "scope": "CUDA to MLX real-input MLP and frozen FP32 LINet parity",
        "checkpoint": str(checkpoint.resolve()),
        "fixture": str(fixture.resolve()),
        "backend": "MLX Metal",
        "comparisons": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not passed:
        raise ValueError(f"CUDA to MLX learned-adapter parity failed; see {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a CUDA-trained adapter on MLX Metal")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = parity_preflight(args.checkpoint, args.fixture, args.weights, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
