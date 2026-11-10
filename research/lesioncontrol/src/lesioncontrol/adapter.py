"""Learned numeric-feature residuals for frozen LINet conditioning.

This module supplies the adapter architecture and a gradient execution preflight.
A finite gradient proves trainability, not HCC fidelity or generalization.
VQ encoding is frozen preprocessing; hard radiomic discretization is not a loss.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import mlx.core as mx
import mlx.nn as nn
from lesiondiffusion_mlx.medmlx import load_stage
from lesiondiffusion_mlx.mlx_model import LINET_CONFIG, LdmUnet

from lesioncontrol.backend import require_metal

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Mlx


class _Evaluator(Protocol):
    def eval(self, *arrays: object) -> None: ...


_evaluate = cast(_Evaluator, mx).eval


def _float(value: mx.array) -> float:
    return float(cast(float, value.item()))


class RadiomicContextAdapter(nn.Module):
    """Map normalized radiomic features to a distinct residual per context token.

    Features have shape (batch, feature_count); context has shape
    (batch, token_count, 512). A zero final projection preserves the exact base
    context at initialization without adding tokens to the attention softmax.
    Feature normalization must be fitted on the permitted training references.
    """

    def __init__(self, token_count: int, feature_count: int = 101, hidden: int = 64):
        super().__init__()
        if min(token_count, feature_count, hidden) < 1:
            raise ValueError("Adapter dimensions must be positive")
        self.token_count = token_count
        self.feature_count = feature_count
        self.hidden = nn.Linear(feature_count, hidden)
        self.output = nn.Linear(hidden, token_count * 512)
        self.output.weight = mx.zeros_like(self.output.weight)
        self.output.bias = mx.zeros_like(self.output.bias)

    def __call__(self, features: mx.array, base_context: mx.array) -> mx.array:
        if features.ndim != 2 or features.shape[-1] != self.feature_count:
            raise ValueError("Features must have shape (batch, feature_count)")
        if base_context.shape != (features.shape[0], self.token_count, 512):
            raise ValueError("Context must have shape (batch, token_count, 512)")
        residual = self.output(nn.silu(self.hidden(features.astype(mx.float32))))
        return base_context + mx.reshape(residual, base_context.shape)


class DifferentiableLINet(LdmUnet):
    """Use stock FP32 MLX operations for gradients through frozen LINet.

    Inference-only custom convolution, norm, attention and copy kernels have no
    registered VJP. The stock convolution/norm/attention path and repeat-based
    upsampling retain the architecture while making input-context gradients
    possible. This training path is distinct from FP16 production inference.
    """

    def __init__(self, weights: Mapping[str, mx.array]):
        super().__init__(
            weights,
            cast("Mlx", mx),
            replace(LINET_CONFIG, skip_context=False),
            source_arithmetic=True,
            fused_attention=False,
            native_kernels=False,
        )

    def resize_nearest(self, x: mx.array, shape: tuple[int, ...]) -> mx.array:
        if shape == tuple(2 * value for value in x.shape[2:]):
            for axis in (2, 3, 4):
                x = mx.repeat(x, 2, axis=axis)
            return x
        return super().resize_nearest(x, shape)


def gradient_preflight(weights: Path, output: Path) -> dict[str, object]:
    """Check real frozen weights, zero initialization and one directional derivative.

    Use a synthetic 8³ latent, one noise-prediction timestep and no patient data.
    Only adapter parameters are differentiated. The output projection starts at
    zero, so the hidden layer initially receives zero gradients by design.
    """
    require_metal()
    mx.random.seed(614)
    frozen = load_stage(weights, "linet", cast("Mlx", mx), dtype=mx.float32)
    model = DifferentiableLINet(frozen)
    _evaluate(*frozen.values())
    adapter = RadiomicContextAdapter(token_count=7)
    features = mx.random.normal((1, 101))
    context = mx.random.normal((1, 7, 512))
    channels = int(frozen["input_blocks.0.0.weight"].shape[1])
    latent = mx.random.normal((1, channels, 8, 8, 8)) * 0.2
    timestep = mx.array([500], dtype=mx.int32)
    baseline = model(latent, timestep, context)
    adapted = model(latent, timestep, adapter(features, context))
    _evaluate(baseline, adapted)
    zero_exact = bool(mx.array_equal(baseline, adapted).item())
    target = mx.random.normal(baseline.shape) * 0.1

    def loss_fn(network: RadiomicContextAdapter) -> mx.array:
        prediction = model(latent, timestep, network(features, context))
        return mx.mean(mx.square(prediction - target))

    loss, grads = nn.value_and_grad(adapter, loss_fn)(adapter)
    _evaluate(loss, grads)
    layer_gradients = cast(dict[str, dict[str, mx.array]], grads)
    gradient_rows = {
        name: {
            "finite": bool(mx.all(mx.isfinite(value)).item()),
            "l2_norm": _float(mx.sqrt(mx.sum(mx.square(value)))),
        }
        for layer, parameters in layer_gradients.items()
        for parameter, value in parameters.items()
        for name in (f"{layer}.{parameter}",)
    }
    output_gradient = layer_gradients["output"]["weight"]
    norm = mx.sqrt(mx.sum(output_gradient * output_gradient))
    if _float(norm) <= 0:
        raise ValueError("Frozen real-weight LINet produced no adapter output gradient")
    direction = output_gradient / norm
    original = adapter.output.weight
    step = 0.01
    adapter.output.weight = original + step * direction
    plus = loss_fn(adapter)
    _evaluate(plus)
    adapter.output.weight = original - step * direction
    minus = loss_fn(adapter)
    _evaluate(minus)
    adapter.output.weight = original
    finite_difference = _float((plus - minus) / (2 * step))
    analytic = _float(mx.sum(output_gradient * direction))
    relative_error = abs(finite_difference - analytic) / max(abs(analytic), 1e-8)
    relative_tolerance = 0.05
    passed = (
        zero_exact
        and math.isfinite(_float(loss))
        and all(row["finite"] for row in gradient_rows.values())
        and math.isfinite(finite_difference)
        and relative_error < relative_tolerance
    )
    report: dict[str, object] = {
        "passed": passed,
        "scope": "Synthetic tiny-latent execution proof; no trained HCC adapter",
        "backend": "MLX Metal",
        "mlx_version": importlib.metadata.version("mlx"),
        "weights": str(weights.resolve()),
        "dtype": "float32",
        "latent_shape": list(latent.shape),
        "context_shape": list(context.shape),
        "zero_init_prediction_exact": zero_exact,
        "loss": _float(loss),
        "adapter_gradients": gradient_rows,
        "finite_difference_step": step,
        "directional_derivative": analytic,
        "finite_difference": finite_difference,
        "relative_error": relative_error,
        "finite_difference_relative_tolerance": relative_tolerance,
        "base_parameters_optimized": False,
        "vq_parameters_optimized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not passed:
        raise ValueError(f"Adapter gradient preflight failed; see {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualify the untrained numeric context adapter")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(gradient_preflight(args.weights, args.output), indent=2))


if __name__ == "__main__":
    main()
