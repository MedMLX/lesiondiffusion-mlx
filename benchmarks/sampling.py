"""Compare DDIM execution on retained real inputs through perfkit.

Set LESION_BENCH_COMPONENT (lmnet or linet), LESION_BENCH_WEIGHTS and
LESION_BENCH_FIXTURE, then use ``sampling:sample_case`` as the input factory
and ``sampling:run_sample`` as the timed callable. The variant argument selects
baseline, cached, or compiled; LESION_BENCH_ITERATIONS bounds the reverse steps
(default 20 of the unchanged 200-step schedule). Cadence and asynchronous
submission are experimental benchmark arguments, not inference options.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Mlx


def sample_case() -> dict[str, object]:
    """Load one fixed reverse trajectory with all setup outside the timed region."""
    from medmlx_core.runtime import import_mlx

    from lesiondiffusion_mlx.medmlx import load_stage
    from lesiondiffusion_mlx.mlx_model import LINET_CONFIG, LMNET_CONFIG, LdmUnet
    from lesiondiffusion_mlx.mlx_sampling import DdimSampler

    mx = cast("Mlx", import_mlx())
    stage = os.environ["LESION_BENCH_COMPONENT"]
    iterations = int(os.environ.get("LESION_BENCH_ITERATIONS", "20"))
    if stage not in {"lmnet", "linet"}:
        raise ValueError("Choose lmnet or linet")
    if not 1 <= iterations <= 200:
        raise ValueError("LESION_BENCH_ITERATIONS must be between 1 and 200")
    arrays = mx.load(os.environ["LESION_BENCH_FIXTURE"], format="safetensors")
    is_lmnet = stage == "lmnet"
    model = LdmUnet(
        load_stage(
            Path(os.environ["LESION_BENCH_WEIGHTS"]),
            stage,
            mx,
            dtype=mx.float32 if is_lmnet else mx.float16,
        ),
        mx,
        LMNET_CONFIG if is_lmnet else replace(LINET_CONFIG, skip_context=False),
        source_arithmetic=is_lmnet,
        fused_attention=True,
        native_kernels=True,
    )
    condition, conditioning = arrays["context"], arrays["conditioning_image"]
    mask = arrays.get("mask")
    sampler = DdimSampler(model.weights["alphas_cumprod"], mx, 200)
    forward = mx.compile(model.__call__)

    def unbound_denoiser(x: Array, timestep: Array) -> Array:
        return forward(x, timestep, condition)

    bound = model.prepare_denoiser(condition)

    def make_step(denoiser: Callable[[Array, Array], Array]) -> Callable[..., Array]:
        def step(
            x: Array,
            timestep: Array,
            alpha: Array,
            previous: Array,
            sigma: Array,
            sqrt_one_minus: Array,
            noise: Array,
        ) -> Array:
            if mask is not None:
                x = x * mask + (1 - mask) * conditioning
                network_input = mx.concatenate([x, mask], axis=1)
            else:
                network_input = mx.concatenate([x, conditioning], axis=1)
            eps = denoiser(network_input, timestep)
            if mask is not None:
                eps = eps[:, : conditioning.shape[1]]
            predicted = (x - sqrt_one_minus * eps) / mx.sqrt(alpha)
            direction = mx.sqrt(1 - previous - sigma**2) * eps
            return mx.sqrt(previous) * predicted + direction + sigma * noise

        return step

    advances = {
        "baseline": make_step(unbound_denoiser),
        "cached": make_step(mx.compile(bound)),
        "compiled": mx.compile(make_step(bound)),
    }
    x = arrays["network_input"][:, : 1 if is_lmnet else conditioning.shape[1]]
    mx.random.seed(23)
    noises = tuple(mx.random.normal(x.shape) for _ in range(iterations))
    indices = list(reversed(range(len(sampler.timesteps))))[:iterations]
    timesteps = tuple(mx.full((x.shape[0],), sampler.timesteps[i], dtype=mx.int32) for i in indices)
    coefficients = tuple(
        (sampler.alphas[i], sampler.previous[i], sampler.sigmas[i], sampler.sqrt_one_minus[i])
        for i in indices
    )
    mx.eval(x, *noises, *timesteps, *(c for row in coefficients for c in row))

    def sample(
        x: Array, *, variant: str = "baseline", cadence: int = 1, asynchronous: bool = False
    ) -> Array:
        advance = advances[variant]
        if cadence < 1:
            raise ValueError("cadence must be positive")
        for iteration, (timestep, coefficient, noise) in enumerate(
            zip(timesteps, coefficients, noises, strict=True), 1
        ):
            x = advance(x, timestep, *coefficient, noise)
            if asynchronous:
                mx.async_eval(x)
            if iteration % cadence == 0:
                mx.eval(x)
        if mask is not None:
            x = x * mask + (1 - mask) * conditioning
        mx.eval(x)
        return x

    return {"name": f"{stage}-{iterations}-steps", "args": (sample, x)}


def run_sample(
    sample: Callable[..., Array],
    x: Array,
    *,
    variant: str = "baseline",
    cadence: int = 1,
    asynchronous: bool = False,
) -> Array:
    """Return a completed fixed-noise trajectory for latency or parity comparison."""
    return sample(x, variant=variant, cadence=cadence, asynchronous=asynchronous)
