"""Metal spatial copies used by the source model's factor-two upsampling."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, MetalKernel, Mlx


_UPSAMPLE = r"""
uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
uint slice = thread_position_in_grid.z;
if (x >= OW || y >= OH || ulong(slice) >= ulong(B) * C * OD) return;
uint z = slice % OD;
uint c = (slice / OD) % C;
uint n = slice / (OD * C);
// Signed offsets retain reversed views; the input need not be row contiguous.
long loc = long(n) * values_strides[0] + long(c) * values_strides[1]
         + long(z / 2) * values_strides[2] + long(y / 2) * values_strides[3]
         + long(x / 2) * values_strides[4];
ulong dst = (ulong(slice) * OH + y) * OW + x;
output[dst] = values[loc];
"""


@lru_cache(maxsize=1)
def _upsample_kernel(mx: Mlx) -> MetalKernel:
    """Reuse the kernel object so repeated denoising does not create Metal libraries."""
    return mx.fast.metal_kernel(
        name="lesion_nearest_upsample2",
        input_names=["values"],
        output_names=["output"],
        source=_UPSAMPLE,
        ensure_row_contiguous=False,
    )


def upsample_nearest2(x: Array, *, mx: Mlx) -> Array:
    """Duplicate NCDHW voxels twice along each spatial axis in one copy pass.

    The caller supplies a nonempty five-dimensional input. Integer division
    implements the source's floor(i / 2) rule exactly. Input strides are read
    directly; output is contiguous NCDHW with the input dtype and values.
    """
    batch, channels, d, h, w = x.shape
    od, oh, ow = 2 * d, 2 * h, 2 * w
    return _upsample_kernel(mx)(
        inputs=[x],
        template=[("B", batch), ("C", channels), ("OD", od), ("OH", oh), ("OW", ow)],
        grid=(ow, oh, batch * channels * od),
        threadgroup=(32, 4, 2),
        output_shapes=[(batch, channels, od, oh, ow)],
        output_dtypes=[x.dtype],
    )[0]


def resize_nearest(x: Array, shape: tuple[int, ...], *, mx: Mlx) -> Array:
    """Resize NCDHW spatial axes with the source's floor(i * in_size / out_size) rule."""
    if len(shape) != 3:
        raise ValueError("LesionDiffusion nearest resizing needs three spatial axes")
    if x.ndim == 5 and min(x.shape) > 0 and shape == tuple(2 * s for s in x.shape[2:]):
        return upsample_nearest2(x, mx=mx)
    for axis, target in enumerate(shape, start=2):
        indices = mx.floor(mx.arange(target) * (x.shape[axis] / target))
        x = mx.take(x, indices.astype(mx.int32), axis=axis)
    return x
