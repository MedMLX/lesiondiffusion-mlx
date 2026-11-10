"""Metal generation with supplied masks and qualified DPM editing controls.

The clean host is the only image passed to LesionDiffusion. Numeric conditioning
is an explicit desired HU value, never an intensity read from a reference tumor.
The process-local hooks reuse the production solver and restore it on every exit.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import mlx.core as mx
import nibabel as nib
import numpy as np
import numpy.typing as npt
from lesiondiffusion_mlx import medmlx
from lesiondiffusion_mlx._dpm import Dpmpp2mSampler

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Mlx
    from lesiondiffusion_mlx.mlx_model import BiomedTextEncoder

    from lesioncontrol.profile import PreparedHost


_GENERATION_LOCK = threading.RLock()


class _RuntimeHooks(Protocol):
    Dpmpp2mSampler: type[Dpmpp2mSampler]


class _Nifti(Protocol):
    shape: tuple[int, ...]
    affine: npt.NDArray[np.float64]
    dataobj: npt.ArrayLike


class _NiftiLoader(Protocol):
    def load(self, filename: str) -> _Nifti: ...


_hooks = cast(_RuntimeHooks, medmlx)
_nifti = cast(_NiftiLoader, nib)
_runtime_namespace = cast(dict[str, object], vars(medmlx))
type _AttributeBuilder = Callable[
    [Mapping[str, object], medmlx.LesionDiffusionOptions], dict[str, object]
]
type _AttributeEncoder = Callable[
    [Mapping[str, object], "BiomedTextEncoder", Path, "Mlx"], mx.array
]


class ContextConditioner(Protocol):
    """Request-local learned context, applied before LINet's projection cache."""

    @property
    def metadata(self) -> Mapping[str, object]: ...

    @property
    def conditioning_schema(self) -> str: ...

    def __call__(self, context: mx.array, attributes: Mapping[str, object]) -> mx.array: ...


@dataclass(frozen=True)
class GenerationControls:
    """Appearance and schedule controls for liver CT supplied-mask inpainting."""

    base_steps: int = 25
    retained_steps: int = 25
    density: str = "isodense lesion"
    density_variation: str = "heterogeneous"
    surface: str = "ill-defined margin"
    specific_features: str = ""
    conditioning_schema: str = "native"
    numeric_offset: float = 0.0

    def __post_init__(self) -> None:
        if not 1 <= self.retained_steps <= self.base_steps <= 999:
            raise ValueError("Require 1 <= retained_steps <= base_steps <= 999")
        if self.conditioning_schema not in {"native", "training"}:
            raise ValueError("conditioning_schema must be native or training")
        if not math.isfinite(self.numeric_offset):
            raise ValueError("numeric_offset must be finite HU")


class GenerationRecord(TypedDict):
    """Public result plus the controls and completed artifact checks."""

    outputs: dict[str, str]
    details: object
    arguments: dict[str, object]
    validation: dict[str, object]


def require_metal() -> None:
    """Select the actual Metal backend or fail before model execution."""
    if not mx.metal.is_available():
        raise RuntimeError("LesionControl generation requires MLX on Metal")
    mx.set_default_device(mx.gpu)


@contextmanager
def generation_context(
    controls: GenerationControls,
    conditioning_mean_hu: float | None = None,
    conditioner: ContextConditioner | None = None,
) -> Generator[None]:
    """Bind fresh-history DPM suffixes and optional pinned training text schema.

    Full-strength native conditioning performs no patch and draws no extra RNG.
    Shorter trajectories initialize alpha*clean_latent + sigma*shared_noise, as
    qualified against Diffusers 0.42.1's inpainting strength contract. The existing
    solver's clean projection outside the supplied mask remains unchanged.
    """
    target = None
    if conditioner is not None and conditioner.conditioning_schema != controls.conditioning_schema:
        raise ValueError("Learned adapter and generation controls use different context schemas")
    if controls.conditioning_schema == "training":
        if conditioning_mean_hu is None or not math.isfinite(conditioning_mean_hu):
            raise ValueError("Training conditioning requires a finite desired mean in HU")
        target = conditioning_mean_hu + controls.numeric_offset
    with _GENERATION_LOCK:
        original_sampler = _hooks.Dpmpp2mSampler
        original_attributes = cast(_AttributeBuilder, _runtime_namespace["_linet_attributes"])
        original_encoder = cast(_AttributeEncoder, _runtime_namespace["encode_attributes"])

        class EditingSampler(Dpmpp2mSampler):
            """Run a production DPM schedule suffix with empty initial history."""

            def __init__(self, alphas_cumprod: mx.array, runtime: Mlx, steps: int):
                super().__init__(alphas_cumprod, runtime, steps)
                begin = len(self.timesteps) - controls.retained_steps
                self.timesteps = self.timesteps[begin:]
                coefficients = self.coefficients[begin:]
                first = coefficients[0]
                self.coefficients = (
                    runtime.concatenate([first[:4], mx.zeros_like(first[4:])]),
                    *coefficients[1:],
                )
                self.start_alpha, self.start_sigma = first[0], first[1]
                runtime.eval(*self.coefficients, self.start_alpha, self.start_sigma)

            def prepare(
                self,
                denoiser: Callable[[mx.array, mx.array], mx.array],
                conditioning_image: mx.array,
                *,
                mask: mx.array | None = None,
            ) -> Callable[[], mx.array]:
                if mask is None:
                    raise ValueError("Editing strength requires supplied-mask LINet inpainting")
                sample = self._bind(denoiser, conditioning_image, mask=mask, compiled=True)
                clean = conditioning_image.astype(mx.float32)

                def draw() -> mx.array:
                    noise = mx.random.normal(conditioning_image.shape)
                    return sample(self.start_alpha * clean + self.start_sigma * noise)

                return draw

        def training_attributes(
            attributes: Mapping[str, object], options: medmlx.LesionDiffusionOptions
        ) -> dict[str, object]:
            assert target is not None
            fields = dict(attributes)
            fields.pop("size", None)
            _, fields["lesion location"] = medmlx.source_organ_label(options.organ)
            fields["CT value"] = f"{float(target)} HU,"
            # Pinned ToTensor retains empty fields as separate conditioning tokens.
            return fields

        def encode_learned(
            attributes: Mapping[str, object],
            model: BiomedTextEncoder,
            tokenizer_path: Path,
            runtime: Mlx,
        ) -> mx.array:
            context = original_encoder(attributes, model, tokenizer_path, runtime)
            assert conditioner is not None
            return conditioner(context, attributes)

        try:
            if controls.retained_steps != controls.base_steps:
                _hooks.Dpmpp2mSampler = EditingSampler
            if target is not None:
                _runtime_namespace["_linet_attributes"] = training_attributes
            if conditioner is not None:
                _runtime_namespace["encode_attributes"] = encode_learned
            yield
        finally:
            _hooks.Dpmpp2mSampler = original_sampler
            _runtime_namespace["_linet_attributes"] = original_attributes
            _runtime_namespace["encode_attributes"] = original_encoder


def generate(
    host: PreparedHost,
    controls: GenerationControls,
    *,
    weights: Path,
    output_dir: Path,
    seed: int,
    conditioning_mean_hu: float | None = None,
    conditioner: ContextConditioner | None = None,
) -> GenerationRecord:
    """Generate CT in a clean host using the supplied placement mask unchanged.

    PreparedHost owns input geometry and clean-host provenance. This boundary
    checks public outputs for finite values, the prepared grid and mask identity.
    Caller-owned GPU locking spans generation and subsequent radiomics extraction.
    """
    require_metal()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    arguments: dict[str, object] = {
        "image_path": str(host.image_path),
        "organ_mask_path": str(host.liver_mask_path),
        "lesion_mask_path": str(host.placement_mask_path),
        "weights_path": str(weights.resolve()),
        "output_dir": str(output_dir),
        "seed": seed,
        "controls": asdict(controls),
        "conditioning_mean_hu": conditioning_mean_hu,
        "target_mean_hu": (
            conditioning_mean_hu + controls.numeric_offset
            if controls.conditioning_schema == "training" and conditioning_mean_hu is not None
            else None
        ),
        "backend": "MLX Metal",
        "learned_adapter": dict(conditioner.metadata) if conditioner is not None else None,
        "host_provenance": {
            "declared_clean": host.declared_clean,
            "known_tumor_mask_path": (
                str(host.known_tumor_mask_path) if host.known_tumor_mask_path is not None else None
            ),
        },
    }
    _write_json(output_dir / "arguments.json", arguments)
    started = time.perf_counter()
    with generation_context(controls, conditioning_mean_hu, conditioner):
        result = medmlx.run(
            image_path=host.image_path,
            organ_mask_path=host.liver_mask_path,
            lesion_mask_path=host.placement_mask_path,
            output_dir=output_dir,
            weights_path=weights,
            organ="liver",
            lesion_type="tumor",
            placement="inside_parenchyma",
            density=controls.density,
            density_variation=controls.density_variation,
            surface=controls.surface,
            specific_features=controls.specific_features,
            enhancement="enhanced CT",
            diffusion_steps=controls.base_steps,
            sampler="dpmpp_2m",
            decoder_mode="reference",
            inference_mode="conditioned",
            output_region="lesion",
            seed=seed,
        )
    _write_json(output_dir / "result.json", result)
    for role, path in result["outputs"].items():
        image = _nifti.load(path)
        data = np.asarray(image.dataobj, dtype=np.float32)
        if image.shape != host.image.shape or not np.allclose(
            image.affine, host.affine, rtol=0, atol=1e-5
        ):
            raise ValueError(f"Generated {role} differs from the prepared host grid")
        if not np.isfinite(data).all():
            raise ValueError(f"Generated {role} contains non-finite values")
        if role == "target_mask" and not np.array_equal(data, host.placement_mask):
            raise ValueError("Generated lesion mask differs from the supplied placement mask")
    actual = result["details"]["stages"]["linet"]["diffusion_steps_actual"]
    if actual != controls.retained_steps:
        raise ValueError("Actual LINet evaluations differ from the requested retained schedule")
    validation: dict[str, object] = {
        "valid": True,
        "finite": True,
        "grid_matches_inputs": True,
        "mask_matches_supplied": True,
        "actual_steps": actual,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "validation.json", validation)
    return {
        "outputs": dict(result["outputs"]),
        "details": result["details"],
        "arguments": arguments,
        "validation": validation,
    }


def _write_json(path: Path, record: object) -> None:
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
