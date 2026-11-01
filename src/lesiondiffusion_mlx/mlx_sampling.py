"""DDIM sampler for the upstream mask and inpainting stages, executed on Metal."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from lesiondiffusion_mlx._typing import ArrayConstructor

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Mlx

from lesiondiffusion_mlx._schedules import lesion_diffusion_timesteps


class DdimSampler:
    """Uniform DDIM sampler with upstream's schedule, covering both pipeline stages.

    One class serves LMNet mask generation (no ``mask`` argument) and LINet latent
    inpainting (with ``mask``), mirroring upstream's separate mask and inpainting
    samplers. All tensors are channels-first ``(B, C, D, H, W)``. Timestep indices
    come from ``lesion_diffusion_timesteps`` and are visited from the largest down.
    ``eta=1`` (the default) matches upstream's stochastic sampling.
    """

    def __init__(self, alphas_cumprod: Array, mx: Mlx, steps: int, *, eta: float = 1.0):
        """Precompute per-step coefficients.

        Args:
            alphas_cumprod: Cumulative product of the training noise schedule, shape
                ``(T,)``, as stored in the converted checkpoint (float16 for LINet).
            mx: The imported ``mlx.core`` module.
            steps: Requested DDIM step count; see ``lesion_diffusion_timesteps``.
            eta: DDIM stochasticity; 0 removes the noise contribution, 1 is the
                upstream setting. Values are not range-checked.

        Raises:
            ValueError: If ``steps`` does not give a valid schedule for ``T``.
        """
        self.mx = mx
        self.timesteps = lesion_diffusion_timesteps(steps, alphas_cumprod.shape[0])
        # Source LINet stores its schedule in fp16, then creates fp32 scalar
        # broadcast tensors in p_sample_ddim. Preserve that rounded schedule.
        alphas = alphas_cumprod[cast(ArrayConstructor, mx.array)(self.timesteps, dtype=mx.int32)]
        previous = mx.concatenate([alphas_cumprod[:1], alphas[:-1]])
        self.alphas = alphas.astype(mx.float32)
        self.previous = previous.astype(mx.float32)
        self.sqrt_one_minus = mx.sqrt(1 - alphas).astype(mx.float32)
        # NumPy numerator / Torch denominator dispatches through Torch's
        # reciprocal in the denominator dtype before promotion. This rounding
        # is observable in LINet's fp16 schedule.
        inverse_variance = mx.reciprocal(1 - alphas)
        mx.eval(inverse_variance)
        self.sigmas = eta * mx.sqrt(
            (1 - self.previous)
            * inverse_variance.astype(mx.float32)
            * (1 - self.alphas / self.previous)
        )
        mx.eval(self.alphas, self.previous, self.sqrt_one_minus, self.sigmas)

    def __call__(
        self,
        denoiser: Callable[[Array, Array, Array | None], Array],
        condition: Array | None,
        conditioning_image: Array,
        *,
        initial_noise: Array | None = None,
        step_noise: tuple[Array, ...] | None = None,
        mask: Array | None = None,
    ) -> Array:
        """Run the reverse process and return the final sample.

        Without ``mask`` (LMNet), ``conditioning_image`` of shape ``(B, C, D, H, W)``
        is concatenated to the noise on every step and the sample has shape
        ``(B, 1, D, H, W)``. With ``mask`` (LINet), ``conditioning_image`` is the
        clean latent ``(B, C, D, H, W)`` and ``mask`` is ``(B, 1, D, H, W)`` with 1
        marking voxels to generate; voxels where ``mask`` is 0 are reset to
        ``conditioning_image`` on every step and in the result, and the sample has
        the shape of ``conditioning_image``.

        Args:
            denoiser: Network called as ``denoiser(input, timestep, condition)`` with
                ``timestep`` an int32 array of shape ``(B,)``. Its output is the
                noise prediction; in the masked case only the first ``C`` channels
                are used and the network's extra mask channel is ignored.
            condition: Cross-attention context ``(B, N, D_ctx)``, or ``None``.
            conditioning_image: Conditioning volume or latent, see above.
            initial_noise: Starting noise with the sample's shape; drawn from MLX's
                RNG when omitted. Mainly for reference comparisons.
            step_noise: One noise array per actual timestep, in the order the steps
                run (largest timestep first); drawn from MLX's RNG when omitted,
                including when ``eta=0``. Individual noise shapes are not checked
                here and follow MLX broadcasting rules.
            mask: Region to generate, or ``None`` for unmasked sampling.

        Returns:
            The final sample. The lazy graph is evaluated after every step so it
            does not grow across steps.

        Raises:
            ValueError: If ``initial_noise`` has the wrong shape or ``step_noise``
                does not have one entry per timestep.
        """
        mx = self.mx
        shape = (
            (conditioning_image.shape[0], 1, *conditioning_image.shape[2:])
            if mask is None
            else conditioning_image.shape
        )
        x = mx.random.normal(shape) if initial_noise is None else initial_noise
        if tuple(x.shape) != tuple(shape):
            raise ValueError(f"DDIM initial noise shape {x.shape} differs from {shape}")
        if step_noise is not None and len(step_noise) != len(self.timesteps):
            raise ValueError("DDIM needs one noise tensor per actual source timestep")
        for iteration, index in enumerate(reversed(range(len(self.timesteps)))):
            timestep = mx.full((x.shape[0],), self.timesteps[index], dtype=mx.int32)
            if mask is not None:
                x = x * mask + (1 - mask) * conditioning_image
                network_input = mx.concatenate([x, mask], axis=1)
            else:
                network_input = mx.concatenate([x, conditioning_image], axis=1)
            eps = denoiser(network_input, timestep, condition)
            if mask is not None:
                # Source predicts a fifth mask channel but retains the supplied
                # mask: p_sample_ddim's mask_pred is deliberately unused.
                eps = eps[:, : conditioning_image.shape[1]]
            alpha, previous, sigma = self.alphas[index], self.previous[index], self.sigmas[index]
            predicted = (x - self.sqrt_one_minus[index] * eps) / mx.sqrt(alpha)
            direction = mx.sqrt(1 - previous - sigma**2) * eps
            noise = mx.random.normal(shape) if step_noise is None else step_noise[iteration]
            x = mx.sqrt(previous) * predicted + direction + sigma * noise
            mx.eval(x)
        if mask is not None:
            x = x * mask + (1 - mask) * conditioning_image
        return x
