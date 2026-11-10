"""Qualify native FP16 attention with ``perfkit benchmark call`` and ``parity run``.

Use ``attention_case`` / ``run_attention`` for isolated attention. Set
LESION_BENCH_VARIANT=native/padded, LESION_BENCH_LENGTH (default 4096),
LESION_BENCH_HEADS (default 16), and optionally LESION_BENCH_MAGNITUDE
(default 1). ``attention_parity_case`` / ``parity_attention`` export identical
seeded inputs to separate workers; select ``native=false`` for the reference.

Use ``denoiser_case`` / ``run_denoiser`` for the real conditioned LINet, with
LESION_BENCH_WEIGHTS and LESION_BENCH_FIXTURE pointing to converted weights and
captured LINet inputs. ``denoiser_parity_case`` / ``parity_denoiser`` compare
the same denoiser inputs with native versus padded attention. Both parity
comparisons use the existing FP16 model bound: --atol 0.002 --rtol 0.002.
Models, inputs and conditioning are prepared outside timing; first calls include
compilation. Timed targets return arrays so perfkit measures completed GPU work.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from lesiondiffusion_mlx._typing import Array, Mlx


def _variant() -> bool:
    """Select one attention implementation for this benchmark process."""
    variant = os.environ.get("LESION_BENCH_VARIANT", "native")
    if variant not in {"native", "padded"}:
        raise ValueError("LESION_BENCH_VARIANT must be native or padded")
    return variant == "native"


def _host_inputs() -> tuple[
    npt.NDArray[np.float16], npt.NDArray[np.float16], npt.NDArray[np.float16], int
]:
    """Create seeded head-interleaved projections with the configured half magnitude."""
    length = int(os.environ.get("LESION_BENCH_LENGTH", "4096"))
    heads = int(os.environ.get("LESION_BENCH_HEADS", "16"))
    magnitude = float(os.environ.get("LESION_BENCH_MAGNITUDE", "1"))
    if length <= 0 or length % 32 or heads <= 0:
        raise ValueError("Attention requires a positive multiple-of-32 length and positive heads")
    if not np.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("Attention magnitude must be finite and positive")
    rng = np.random.default_rng(2701)
    shape = (1, length, heads * 32)
    q = (rng.normal(size=shape) * magnitude).astype(np.float16)
    k = (rng.normal(size=shape) * magnitude).astype(np.float16)
    v = rng.normal(size=shape).astype(np.float16)
    return q, k, v, heads


def _attention(q: Array, k: Array, v: Array, *, heads: int, native: bool, mx: Mlx) -> Array:
    """Use production direct-half attention or the prior padded fused MLX operation."""
    from lesiondiffusion_mlx._attention import flash_attention

    if native:
        return flash_attention(q, k, v, heads=heads, scale=32**-0.5, mx=mx)

    def split(x: Array) -> Array:
        return mx.pad(
            mx.transpose(mx.reshape(x, (x.shape[0], x.shape[1], heads, 32)), (0, 2, 1, 3)),
            [(0, 0), (0, 0), (0, 0), (0, 32)],
        )

    output = mx.fast.scaled_dot_product_attention(
        split(q), split(k), split(v), scale=32**-0.5, force_fused=True
    )[..., :32]
    return mx.reshape(mx.transpose(output, (0, 2, 1, 3)), tuple(q.shape))


def attention_case() -> dict[str, object]:
    """Prepare evaluated inputs and one compiled attention operation outside timing."""
    from medmlx_core.runtime import import_mlx

    from lesiondiffusion_mlx._attention import flash_eligible
    from lesiondiffusion_mlx._typing import ArrayConstructor

    mx = cast("Mlx", import_mlx())
    native = _variant()
    host_q, host_k, host_v, heads = _host_inputs()
    array = cast(ArrayConstructor, mx.array)
    q, k, v = array(host_q), array(host_k), array(host_v)
    if native and not flash_eligible(q, k, heads, mx):
        raise RuntimeError("Native half attention requires enabled NAX on this host")

    def forward(q: Array, k: Array, v: Array) -> Array:
        return _attention(q, k, v, heads=heads, native=native, mx=mx)

    mx.eval(q, k, v)
    return {
        "name": f"attention-{'native' if native else 'padded'}-{tuple(q.shape)}",
        "args": (mx.compile(forward), q, k, v),
    }


def run_attention(
    forward: Callable[[Array, Array, Array], Array], q: Array, k: Array, v: Array
) -> Array:
    """Return attention outputs for perfkit's synchronized completion boundary."""
    return forward(q, k, v)


def attention_parity_case() -> dict[str, object]:
    """Export identical host projections to independent native and padded workers."""
    q, k, v, heads = _host_inputs()
    return {"name": f"attention-half-{tuple(q.shape)}", "args": (q, k, v, heads)}


def parity_attention(
    q: npt.NDArray[np.float16],
    k: npt.NDArray[np.float16],
    v: npt.NDArray[np.float16],
    heads: int,
    *,
    native: bool = True,
) -> npt.NDArray[np.float32]:
    """Evaluate one attention implementation for comparison at the existing half bound."""
    from medmlx_core.runtime import import_mlx

    from lesiondiffusion_mlx._attention import flash_eligible
    from lesiondiffusion_mlx._typing import ArrayConstructor

    mx = cast("Mlx", import_mlx())
    array = cast(ArrayConstructor, mx.array)
    queries, keys, values = array(q), array(k), array(v)
    if native and not flash_eligible(queries, keys, heads, mx):
        raise RuntimeError("Native half attention requires enabled NAX on this host")
    output = _attention(queries, keys, values, heads=heads, native=native, mx=mx)
    return np.array(output, dtype=np.float32)


@contextmanager
def _padded_half_attention() -> Generator[None]:
    """Trace the prior LINet path while retaining native convolutions and FP32 attention.

    This benchmark-only binding lasts through the compiled callable's tracing
    and restores the module immediately afterward; workers run in separate
    processes. The benchmark uses no shared production configuration changes.
    """
    from lesiondiffusion_mlx import mlx_model

    original = mlx_model.flash_eligible

    def eligible(q: Array, k: Array, heads: int, mx: Mlx) -> bool:
        return q.dtype != mx.float16 and original(q, k, heads, mx)

    mlx_model.flash_eligible = eligible
    try:
        yield
    finally:
        mlx_model.flash_eligible = original


def _denoiser(
    weights_path: Path, context: Array, *, native: bool, mx: Mlx
) -> Callable[[Array, Array], Array]:
    """Load one conditioned LINet and bind evaluated text projections before timing."""
    from lesiondiffusion_mlx._nax_conv import nax_available
    from lesiondiffusion_mlx.medmlx import load_stage
    from lesiondiffusion_mlx.mlx_model import LINET_CONFIG, LdmUnet

    if native and not nax_available(mx):
        raise RuntimeError("Native half attention requires enabled NAX on this host")
    model = LdmUnet(
        load_stage(weights_path, "linet", mx, dtype=mx.float16),
        mx,
        replace(LINET_CONFIG, skip_context=False),
        source_arithmetic=False,
        fused_attention=True,
        native_kernels=True,
    )
    forward = mx.compile(model.prepare_denoiser(context))
    mx.eval(*model.weights.values())
    if native:
        return forward

    def padded(x: Array, timestep: Array) -> Array:
        with _padded_half_attention():
            return forward(x, timestep)

    return padded


def denoiser_case() -> dict[str, object]:
    """Load a real captured LINet input and the selected compiled denoiser."""
    from medmlx_core.runtime import import_mlx

    mx = cast("Mlx", import_mlx())
    native = _variant()
    arrays = mx.load(os.environ["LESION_BENCH_FIXTURE"], format="safetensors")
    forward = _denoiser(
        Path(os.environ["LESION_BENCH_WEIGHTS"]), arrays["context"], native=native, mx=mx
    )
    x, timestep = arrays["network_input"], arrays["timestep"]
    mx.eval(x, timestep)
    return {
        "name": f"linet-attention-{'native' if native else 'padded'}-{tuple(x.shape)}",
        "args": (forward, x, timestep),
    }


def run_denoiser(forward: Callable[[Array, Array], Array], x: Array, timestep: Array) -> Array:
    """Return the denoiser output so perfkit includes completed GPU work."""
    return forward(x, timestep)


def denoiser_parity_case() -> dict[str, object]:
    """Export captured LINet inputs to independent reference and candidate workers."""
    from medmlx_core.runtime import import_mlx

    mx = cast("Mlx", import_mlx())
    arrays = mx.load(os.environ["LESION_BENCH_FIXTURE"], format="safetensors")
    return {
        "name": f"linet-attention-half-{tuple(arrays['network_input'].shape)}",
        "args": (
            np.array(arrays["network_input"]),
            np.array(arrays["timestep"]),
            np.array(arrays["context"]),
            os.environ["LESION_BENCH_WEIGHTS"],
        ),
    }


def parity_denoiser(
    x: npt.NDArray[np.floating],
    timestep: npt.NDArray[np.integer],
    context: npt.NDArray[np.floating],
    weights_path: str,
    *,
    native: bool = True,
) -> npt.NDArray[np.float32]:
    """Compare real LINet predictions when only half self-attention changes."""
    from medmlx_core.runtime import import_mlx

    from lesiondiffusion_mlx._typing import ArrayConstructor

    mx = cast("Mlx", import_mlx())
    array = cast(ArrayConstructor, mx.array)
    forward = _denoiser(Path(weights_path), array(context), native=native, mx=mx)
    return np.array(forward(array(x), array(timestep)), dtype=np.float32)
