"""Replay the pre-optimization conditioned execution in the current checkout.

The original UNet call, every-step DDIM loop and chunked codebook lookup remain
in production for source-compatible execution and explicit comparisons. This
benchmark-only context routes conditioned inference through those retained
implementations. It does not change weights, precision, model inputs or outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

from lesiondiffusion_mlx import medmlx, mlx_model
from lesiondiffusion_mlx.mlx_model import LdmUnet, VqAutoencoder
from lesiondiffusion_mlx.mlx_sampling import DdimSampler

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Mlx


class _BaselineDenoiser:
    """Retain dynamic context for both the compiled call and component capture."""

    def __init__(
        self,
        forward: Callable[[Array, Array, Array | None], Array],
        context: Array | None,
    ):
        self.forward = forward
        self.context = context

    def __call__(self, x: Array, timestep: Array) -> Array:
        """Pass context as an input to the original compiled denoiser."""
        return self.forward(x, timestep, self.context)


class _BaselineVq(VqAutoencoder):
    """Use the retained chunked lookup, recomputing norms as in the baseline."""

    def quantize(self, z: Array, *, chunk_size: int = 4096) -> Array:
        native = self.native_kernels
        cached = self._codebook_norm_cache
        self.native_kernels = False
        self._codebook_norm_cache = None
        try:
            return super().quantize(z, chunk_size=chunk_size)
        finally:
            self.native_kernels = native
            self._codebook_norm_cache = cached


@contextmanager
def baseline_execution() -> Generator[None]:
    """Temporarily use dynamic-context UNet compilation and the original DDIM loop."""
    original_denoiser = LdmUnet.prepare_denoiser
    original_sampler = DdimSampler.prepare
    original_vq = medmlx.VqAutoencoder
    original_flash_eligible = mlx_model.flash_eligible

    def flash_eligible(q: Array, k: Array, heads: int, mx: Mlx) -> bool:
        # Before the FP16 specialization, half-precision heads used padded MLX SDPA.
        return q.dtype == mx.float32 and original_flash_eligible(q, k, heads, mx)

    def prepare_denoiser(self: LdmUnet, context: Array | None) -> Callable[[Array, Array], Array]:
        # Keep context a dynamic compiled-function input as it was in the baseline;
        # compiling a closure over it could constant-fold its projections instead.
        forward = self.mx.compile(self.__call__)
        return _BaselineDenoiser(forward, context)

    def prepare_sampler(
        self: DdimSampler,
        denoiser: Callable[[Array, Array], Array],
        conditioning_image: Array,
        *,
        mask: Array | None = None,
    ) -> Callable[[], Array]:
        bound = cast("_BaselineDenoiser", denoiser)

        def forward(x: Array, timestep: Array, _context: Array | None) -> Array:
            return bound(x, timestep)

        def sample() -> Array:
            return self(forward, bound.context, conditioning_image, mask=mask)

        return sample

    LdmUnet.prepare_denoiser = prepare_denoiser
    DdimSampler.prepare = prepare_sampler
    medmlx.VqAutoencoder = _BaselineVq
    mlx_model.flash_eligible = flash_eligible
    try:
        yield
    finally:
        LdmUnet.prepare_denoiser = original_denoiser
        DdimSampler.prepare = original_sampler
        medmlx.VqAutoencoder = original_vq
        mlx_model.flash_eligible = original_flash_eligible
