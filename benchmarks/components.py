"""Capture real model inputs for repeatable component benchmarks.

The optional capture wraps existing inference boundaries only for the benchmark
process. It preserves the arrays and RNG calls consumed by the public pipeline.
Exports synchronize arrays and add I/O, so their elapsed time is recorded and
capture runs must not be mixed with uninstrumented throughput comparisons.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from lesiondiffusion_mlx._arrays import BoolArray, HostArray, NumericArray
    from lesiondiffusion_mlx._typing import Array, Mlx


class _ArrayWriter(Protocol):
    def save_safetensors(self, file: str, arrays: dict[str, Array]) -> None:
        """Write already materialized arrays in safetensors format."""
        ...


def save_arrays(path: Path, arrays: dict[str, Array]) -> None:
    """Write arrays through the runtime's safetensors API."""
    import mlx.core as mx

    cast(_ArrayWriter, mx).save_safetensors(str(path), arrays)


@contextmanager
def capture_components(directory: Path) -> Generator[dict[str, float]]:
    """Save stage inputs/latents and every raw LMNet QC attempt when enabled.

    ``lmnet-qc-attempt-N`` contains continuous source-order scores before any
    thresholding, the reference box, and the selected component. This host
    boundary works for both samplers. ``linet-lesion-mask`` records the full
    padded mask actually passed to LINet in source and RAS array order; output
    NIfTIs drop only the trailing even-grid padding. Hooks exist only inside this
    optional benchmark context, leaving ordinary inference unchanged.
    """
    from safetensors.numpy import save_file

    from lesiondiffusion_mlx import medmlx
    from lesiondiffusion_mlx.mlx_model import VqAutoencoder
    from lesiondiffusion_mlx.mlx_sampling import DdimSampler

    directory.mkdir(parents=True, exist_ok=True)
    original_sample = DdimSampler.__call__
    original_prepare = DdimSampler.prepare
    original_encode = VqAutoencoder.encode
    original_decode = VqAutoencoder.decode
    original_select = medmlx.select_lesion_component
    original_crop = medmlx.source_crop
    exports: dict[str, float] = {}
    decode_count = 0
    lmnet_attempt = 0

    def save(name: str, arrays: dict[str, Array]) -> None:
        if name in exports:
            return
        started = time.perf_counter()
        save_arrays(directory / f"{name}.safetensors", arrays)
        exports[name] = time.perf_counter() - started

    def save_host(name: str, arrays: dict[str, HostArray]) -> None:
        if name in exports:
            return
        started = time.perf_counter()
        save_file(arrays, directory / f"{name}.safetensors")
        exports[name] = time.perf_counter() - started

    def select_component(samples: NumericArray, reference: BoolArray) -> BoolArray:
        nonlocal lmnet_attempt
        lmnet_attempt += 1
        # Snapshot before selection, independently of the sampler implementation.
        # The exact source thresholds can then be analyzed offline without a rerun.
        raw = samples.copy()
        retained = original_select(samples, reference)
        save_host(
            f"lmnet-qc-attempt-{lmnet_attempt}",
            {
                "sample_source_zyx": raw,
                "reference_box_source_zyx": reference,
                "retained_component_source_zyx": retained,
            },
        )
        return retained

    def crop_source(mask: HostArray, shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
        crop = original_crop(mask, shape)
        if shape == medmlx.LINET_CROP:
            # LINet's crop is resolved from its full-resolution input mask.
            save_host(
                "linet-lesion-mask",
                {"mask_source_zyx": mask, "mask_ras_xyz": medmlx.source_to_ras(mask)},
            )
        return crop

    def sample(
        self: DdimSampler,
        denoiser: Callable[[Array, Array, Array | None], Array],
        condition: Array | None,
        conditioning_image: Array,
        *,
        initial_noise: Array | None = None,
        step_noise: tuple[Array, ...] | None = None,
        mask: Array | None = None,
    ) -> Array:
        stage = "lmnet" if mask is None else "linet"

        def forward(x: Array, timestep: Array, context: Array | None) -> Array:
            arrays = {"network_input": x, "timestep": timestep}
            if context is not None:
                arrays["context"] = context
            arrays["conditioning_image"] = conditioning_image
            if mask is not None:
                arrays["mask"] = mask
            save(stage, arrays)
            return denoiser(x, timestep, context)

        result = original_sample(
            self,
            forward,
            condition,
            conditioning_image,
            initial_noise=initial_noise,
            step_noise=step_noise,
            mask=mask,
        )
        save(f"{stage}-sample", {"sample": result})
        return result

    def prepare(
        self: DdimSampler,
        denoiser: Callable[[Array, Array], Array],
        conditioning_image: Array,
        *,
        mask: Array | None = None,
    ) -> Callable[[], Array]:
        # Export after the trajectory. Saving from the bound denoiser would run
        # disk I/O during compiled-step tracing and change the execution graph.
        prepared = original_prepare(self, denoiser, conditioning_image, mask=mask)
        stage = "lmnet" if mask is None else "linet"

        def sampled() -> Array:
            result = prepared()
            save(f"{stage}-sample", {"sample": result})
            return result

        return sampled

    def encode(self: VqAutoencoder, x: Array) -> Array:
        latent = original_encode(self, x)
        save("vq-encode", {"input": x, "latent": latent})
        return latent

    def decode(self: VqAutoencoder, z: Array, *, quantize: bool = True) -> Array:
        nonlocal decode_count
        name = "vq-generated" if decode_count == 0 else "vq-reconstruction"
        decode_count += 1
        save(name, {"latent": z})
        return original_decode(self, z, quantize=quantize)

    DdimSampler.__call__ = sample
    DdimSampler.prepare = prepare
    VqAutoencoder.encode = encode
    VqAutoencoder.decode = decode
    medmlx.select_lesion_component = select_component
    medmlx.source_crop = crop_source
    try:
        yield exports
    finally:
        DdimSampler.__call__ = original_sample
        DdimSampler.prepare = original_prepare
        VqAutoencoder.encode = original_encode
        VqAutoencoder.decode = original_decode
        medmlx.select_lesion_component = original_select
        medmlx.source_crop = original_crop


def component_case() -> dict[str, object]:
    """Load one real fixture and model for ``perfkit benchmark call`` setup.

    Set ``LESION_BENCH_WEIGHTS`` to the converted weights directory,
    ``LESION_BENCH_FIXTURE`` to a captured safetensors file, and
    ``LESION_BENCH_COMPONENT`` to ``lmnet``, ``linet``, ``vq-encode``,
    ``vq-decode`` or ``vq-quantize``. ``LESION_BENCH_DTYPE`` optionally selects
    float32, float16 or bfloat16 for precision experiments. Defaults match the
    conditioned pipeline. Model loading and input materialization are setup;
    the first timed call includes graph compilation for the two UNets.
    """
    import os
    from dataclasses import replace

    import mlx.core as mx

    from lesiondiffusion_mlx.medmlx import load_stage
    from lesiondiffusion_mlx.mlx_model import (
        LINET_CONFIG,
        LMNET_CONFIG,
        LdmUnet,
        VqAutoencoder,
    )

    component = os.environ["LESION_BENCH_COMPONENT"]
    if component not in {"lmnet", "linet", "vq-encode", "vq-decode", "vq-quantize"}:
        raise ValueError(f"Unknown benchmark component: {component}")
    weights_dir = Path(os.environ["LESION_BENCH_WEIGHTS"])
    fixture = Path(os.environ["LESION_BENCH_FIXTURE"])
    runtime = cast("Mlx", mx)
    arrays = runtime.load(str(fixture), format="safetensors")
    dtype_name = os.environ.get(
        "LESION_BENCH_DTYPE", "float16" if component in {"linet", "vq-encode"} else "float32"
    )
    dtypes = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
    if dtype_name not in dtypes:
        raise ValueError(f"Unknown benchmark dtype: {dtype_name}")
    dtype = dtypes[dtype_name]
    weights = load_stage(
        weights_dir, component if component in {"lmnet", "linet"} else "vq", runtime, dtype=dtype
    )
    operation: Callable[[Array], Array]
    if component in {"lmnet", "linet"}:
        config = LMNET_CONFIG if component == "lmnet" else replace(LINET_CONFIG, skip_context=False)
        model = LdmUnet(
            weights,
            runtime,
            config,
            source_arithmetic=component == "lmnet",
            fused_attention=True,
            native_kernels=True,
        )
        forward = mx.compile(model.__call__)
        x, timestep, context = arrays["network_input"], arrays["timestep"], arrays["context"]

        def model_forward(x: Array) -> Array:
            return forward(x, timestep, context)

        operation = model_forward

    else:
        vq = VqAutoencoder(weights, runtime, source_arithmetic=False)
        x = arrays["input" if component == "vq-encode" else "latent"].astype(dtype)
        operation = (
            vq.encode
            if component == "vq-encode"
            else vq.decode
            if component == "vq-decode"
            else vq.quantize
        )
    runtime.eval(*weights.values(), *arrays.values(), x)
    return {"name": f"{component}-{dtype_name}-{tuple(x.shape)}", "args": (operation, x)}


def run_component(operation: Callable[[Array], Array], x: Array) -> Array:
    """Return the measured result so perfkit evaluates completed GPU work."""
    return operation(x)
