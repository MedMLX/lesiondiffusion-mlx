"""Four-channel VQ search without a latent-position by codebook distance matrix."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, MetalKernel, Mlx


_SEARCH = r"""
// One SIMD group searches one latent vector. Each lane visits every 32nd
// code; the final reduction selects the smallest index on equal distances.
const uint position = thread_position_in_grid.x / 32;
const uint lane = thread_index_in_simdgroup;
if (position >= P) return;
T best = T(INFINITY);
uint best_index = 0xffffffffu;
for (uint code = lane; code < E; code += 32) {
  // Match the four-term FP32 GEMM accumulator, then its output rounding.
  float dot = float(z[position * 4]) * float(codebook[code * 4]);
  dot = metal::fma(float(z[position * 4 + 1]), float(codebook[code * 4 + 1]), dot);
  dot = metal::fma(float(z[position * 4 + 2]), float(codebook[code * 4 + 2]), dot);
  dot = metal::fma(float(z[position * 4 + 3]), float(codebook[code * 4 + 3]), dot);
  // The array expression rounds the norm sum, scaled product and subtraction
  // separately. Keep these boundaries, especially for FP16 near ties.
  const T summed = T(z_norm[position] + codebook_norm[code]);
  const T doubled = T(T(2) * T(dot));
  const T distance = T(summed - doubled);
  if ((isnan(distance) && (!isnan(best) || code < best_index)) ||
      (!isnan(best) &&
       (distance < best || (distance == best && code < best_index)))) {
    best = distance;
    best_index = code;
  }
}
for (ushort offset = 16; offset > 0; offset /= 2) {
  const T other = simd_shuffle_down(best, offset);
  const uint other_index = simd_shuffle_down(best_index, offset);
  if ((isnan(other) && (!isnan(best) || other_index < best_index)) ||
      (!isnan(best) &&
       (other < best || (other == best && other_index < best_index)))) {
    best = other;
    best_index = other_index;
  }
}
if (lane == 0) indices[position] = best_index;
"""


@lru_cache(maxsize=1)
def _search_kernel(mx: Mlx) -> MetalKernel:
    """Compile once; explicit fma handles the dot while other operations stay separate."""
    return mx.fast.metal_kernel(
        name="lesion_vq_nearest4",
        input_names=["z", "codebook", "z_norm", "codebook_norm"],
        output_names=["indices"],
        source=_SEARCH,
        header="#pragma clang fp contract(off)\n",
    )


def nearest_codebook4(z: Array, codebook: Array, codebook_norm: Array, mx: Mlx) -> Array:
    """Return nearest indices for ``(positions, 4)`` vectors and a nonempty codebook.

    Inputs share FP16 or FP32 dtype. Norms retain the array implementation's
    rounded squares and FP32 accumulation. The search retains each distance's
    dtype rounding, chooses the first code on a tie, and gives NaNs the same
    first-index priority as MLX argmin. Scratch storage grows with positions,
    rather than positions times codebook entries.
    """
    z_norm = mx.sum((z * z).astype(mx.float32), axis=1).astype(z.dtype)
    positions = int(z.shape[0])
    return _search_kernel(mx)(
        inputs=[z, codebook, z_norm, codebook_norm],
        template=[("T", z.dtype), ("P", positions), ("E", int(codebook.shape[0]))],
        grid=(positions * 32, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(positions,)],
        output_dtypes=[mx.int32],
    )[0]
