"""Deterministic DPM-Solver++ 2M with the Diffusers midpoint update.

The numerical reference is Diffusers 0.42.1, DPMSolverMultistepScheduler with
epsilon prediction, solver order 2, midpoint updates, linspace timesteps,
no thresholding and a final zero sigma. This is an alternative inference
trajectory, distinct from the stochastic source DDIM sampler.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from lesiondiffusion_mlx._schedules import dpm_solver_timesteps
from lesiondiffusion_mlx._typing import ArrayConstructor

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Mlx


class Dpmpp2mSampler:
    """Sample FP32 VP state with epsilon prediction and a two-step x0 history.

    Checkpoint alpha products retain their stored rounding before conversion
    to FP32. Known inpainting voxels are reset before every denoiser call and
    in the final output. Only the initial noise is random; this solver has no
    per-step stochastic draw or eta parameter.
    """

    def __init__(self, alphas_cumprod: Array, mx: Mlx, steps: int):
        """Prepare the reference linspace schedule and FP32 update coefficients."""
        self.mx = mx
        self.timesteps = dpm_solver_timesteps(steps, int(alphas_cumprod.shape[0]))
        indices = cast(ArrayConstructor, mx.array)(self.timesteps, dtype=mx.int32)
        abar = alphas_cumprod[indices].astype(mx.float32)
        # The reference first computes sigma/alpha, then reconstructs VP
        # alpha and sigma from that ratio. Retain those rounding boundaries.
        ratios = mx.sqrt((1 - abar) / abar)
        alpha = 1 / mx.sqrt(ratios**2 + 1)
        sigma = ratios * alpha
        log_snr = mx.log(alpha) - mx.log(sigma)
        mx.eval(alpha, sigma, log_snr)
        if not bool(mx.all(mx.isfinite(log_snr)).item()):
            raise ValueError(
                "DPM-Solver++ needs finite sampled alphas strictly between zero and one"
            )
        zero = cast(ArrayConstructor, mx.array)(0, dtype=mx.float32)
        one = cast(ArrayConstructor, mx.array)(1, dtype=mx.float32)
        coefficients: list[Array] = []
        for index in range(len(self.timesteps)):
            if index == len(self.timesteps) - 1:
                # Zero terminal sigma forces first order in the reference.
                # Its finite limit is x0; avoid log(0) and inf/inf in history.
                ratio, clean, history = zero, one, zero
            else:
                h = log_snr[index + 1] - log_snr[index]
                ratio = sigma[index + 1] / sigma[index]
                clean = -alpha[index + 1] * mx.expm1(-h)
                history = (
                    zero if index == 0 else 0.5 * clean * h / (log_snr[index] - log_snr[index - 1])
                )
            coefficients.append(mx.stack([alpha[index], sigma[index], ratio, clean, history]))
        self.coefficients = tuple(coefficients)
        mx.eval(*self.coefficients)

    def __call__(
        self,
        denoiser: Callable[[Array, Array, Array | None], Array],
        condition: Array | None,
        conditioning_image: Array,
        *,
        initial_noise: Array | None = None,
        mask: Array | None = None,
    ) -> Array:
        """Run the uncompiled reference path, optionally from supplied initial noise."""

        def bound(x: Array, timestep: Array) -> Array:
            return denoiser(x, timestep, condition)

        return self._bind(bound, conditioning_image, mask=mask, compiled=False)(initial_noise)

    def prepare(
        self,
        denoiser: Callable[[Array, Array], Array],
        conditioning_image: Array,
        *,
        mask: Array | None = None,
    ) -> Callable[[], Array]:
        """Compile one context-bound solver step and retain bounded FP32 history.

        The binding owns its fixed denoiser, conditioning and mask. It draws
        one fresh initial noise tensor on each call. Evaluating both returned
        tensors each step prevents the x0 history from retaining activations.
        """
        sample = self._bind(denoiser, conditioning_image, mask=mask, compiled=True)

        def draw() -> Array:
            return sample(None)

        return draw

    def _bind(
        self,
        denoiser: Callable[[Array, Array], Array],
        conditioning_image: Array,
        *,
        mask: Array | None,
        compiled: bool,
    ) -> Callable[[Array | None], Array]:
        """Bind fixed inputs to the same midpoint update for both execution paths."""
        mx = self.mx
        shape = (
            (conditioning_image.shape[0], 1, *conditioning_image.shape[2:])
            if mask is None
            else conditioning_image.shape
        )
        timesteps = tuple(mx.full((shape[0],), value, dtype=mx.int32) for value in self.timesteps)
        mx.eval(*timesteps)

        def step(x: Array, previous: Array, timestep: Array, coeff: Array) -> tuple[Array, Array]:
            if mask is not None:
                x = x * mask + (1 - mask) * conditioning_image
                network_input = mx.concatenate([x, mask], axis=1)
            else:
                network_input = mx.concatenate([x, conditioning_image], axis=1)
            eps = denoiser(network_input, timestep)
            if mask is not None:
                eps = eps[:, : conditioning_image.shape[1]]
            # Solver state/history are FP32 even when LINet predicts FP16 noise.
            eps = eps.astype(mx.float32)
            predicted = (x - coeff[1] * eps) / coeff[0]
            updated = coeff[2] * x + coeff[3] * predicted + coeff[4] * (predicted - previous)
            return updated, predicted

        advance = mx.compile(step) if compiled else step

        def sample(initial_noise: Array | None) -> Array:
            x = (
                mx.random.normal(shape)
                if initial_noise is None
                else initial_noise.astype(mx.float32)
            )
            if tuple(x.shape) != tuple(shape):
                raise ValueError(f"DPM-Solver++ initial noise shape {x.shape} differs from {shape}")
            previous = x  # First-order coefficient ignores history on the first step.
            for timestep, coefficient in zip(timesteps, self.coefficients, strict=True):
                x, previous = advance(x, previous, timestep, coefficient)
                mx.eval(x, previous)
            if mask is not None:
                x = x * mask + (1 - mask) * conditioning_image
            return x

        return sample
