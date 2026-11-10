"""VQ comparisons using ``perfkit benchmark call`` and captured real latents.

Set LESION_BENCH_WEIGHTS and LESION_BENCH_FIXTURE as for ``components``.
``quantize_case`` compares LESION_BENCH_VARIANT=baseline/fused, with optional
LESION_BENCH_DTYPE=float32/float16. ``decode_pair_case`` compares
LESION_BENCH_VARIANT=sequential/sequential-clear/batched and requires a second captured latent in
LESION_BENCH_SECOND_FIXTURE. Models and inputs are prepared outside timing;
the first call includes kernel compilation. Return values let perfkit evaluate
completed GPU work. Use --kwarg chunk_size=1024 (etc.) for a reference chunk sweep.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from lesiondiffusion_mlx._typing import Array, Mlx
    from lesiondiffusion_mlx.mlx_model import VqAutoencoder


def quantize_case() -> dict[str, object]:
    """Load the same codebook and latent for reference and fused VQ search."""
    from medmlx_core.runtime import import_mlx

    from lesiondiffusion_mlx.mlx_model import VqAutoencoder

    mx = cast("Mlx", import_mlx())
    variant = os.environ.get("LESION_BENCH_VARIANT", "fused")
    dtype_name = os.environ.get("LESION_BENCH_DTYPE", "float32")
    if variant not in {"baseline", "fused"} or dtype_name not in {"float32", "float16"}:
        raise ValueError("Choose baseline/fused and float32/float16 for VQ search")
    dtype = mx.float32 if dtype_name == "float32" else mx.float16
    packed = mx.load(
        str(Path(os.environ["LESION_BENCH_WEIGHTS"]) / "model.safetensors"), format="safetensors"
    )
    codebook = packed["vq.quantize.embedding.weight"].astype(dtype)
    del packed
    arrays = mx.load(os.environ["LESION_BENCH_FIXTURE"], format="safetensors")
    latent = arrays["latent"].astype(dtype)
    model = VqAutoencoder(
        {"quantize.embedding.weight": codebook}, mx, source_arithmetic=variant == "baseline"
    )
    mx.eval(codebook, latent)
    return {"name": f"vq-{variant}-{dtype_name}-{tuple(latent.shape)}", "args": (model, latent)}


def run_quantize(model: VqAutoencoder, latent: Array, *, chunk_size: int = 4096) -> Array:
    """Return quantized values; chunk size controls the reference distance matrices."""
    return model.quantize(latent, chunk_size=chunk_size)


def parity_case() -> dict[str, object]:
    """Export one real latent and codebook to independent parity workers."""
    case = quantize_case()
    model, latent = cast("tuple[VqAutoencoder, Array]", case["args"])
    return {
        "name": case["name"],
        "args": (
            np.array(latent),
            np.array(model.weights["quantize.embedding.weight"]),
        ),
    }


def parity_quantize(
    latent: npt.NDArray[np.floating],
    codebook: npt.NDArray[np.floating],
    *,
    native: bool = True,
) -> npt.NDArray[np.float32]:
    """Compare retained chunked lookup and native lookup with identical arithmetic."""
    from medmlx_core.runtime import import_mlx

    from lesiondiffusion_mlx._typing import ArrayConstructor
    from lesiondiffusion_mlx.mlx_model import VqAutoencoder

    mx = cast("Mlx", import_mlx())
    array = cast(ArrayConstructor, mx.array)
    model = VqAutoencoder(
        {"quantize.embedding.weight": array(codebook)},
        mx,
        source_arithmetic=False,
        native_kernels=native,
    )
    return np.array(model.quantize(array(latent)), dtype=np.float32)


def decode_pair_case() -> dict[str, object]:
    """Prepare two independent real latents for sequential or batch-two FP32 decoding."""
    from medmlx_core.runtime import import_mlx

    from lesiondiffusion_mlx.medmlx import load_stage
    from lesiondiffusion_mlx.mlx_model import VqAutoencoder

    mx = cast("Mlx", import_mlx())
    variant = os.environ.get("LESION_BENCH_VARIANT", "sequential")
    if variant not in {"sequential", "sequential-clear", "batched"}:
        raise ValueError("Choose sequential/sequential-clear/batched for paired decoding")
    weights = load_stage(Path(os.environ["LESION_BENCH_WEIGHTS"]), "vq", mx, dtype=mx.float32)
    model = VqAutoencoder(weights, mx, source_arithmetic=False)
    first = mx.load(os.environ["LESION_BENCH_FIXTURE"], format="safetensors")["latent"]
    second = mx.load(os.environ["LESION_BENCH_SECOND_FIXTURE"], format="safetensors")["latent"]
    if first.shape != second.shape:
        raise ValueError("Paired VQ latents must have the same shape")
    first, second = first.astype(mx.float32), second.astype(mx.float32)
    combined = mx.concatenate([first, second], axis=0)
    mx.eval(first, second, combined)

    def decode() -> tuple[Array, Array]:
        if variant != "batched":
            first_output = model.decode(first)
            if variant == "sequential-clear":
                mx.clear_cache()
            return first_output, model.decode(second)
        output = model.decode(combined)
        return output[: first.shape[0]], output[first.shape[0] :]

    return {"name": f"vq-decode-{variant}-{tuple(first.shape)}", "args": (decode,)}


def run_decode_pair(decode: Callable[[], tuple[Array, Array]]) -> tuple[Array, Array]:
    """Return both decoded volumes so the timer waits for both outputs."""
    return decode()
