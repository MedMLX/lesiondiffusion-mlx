"""Native FP16 NAX attention for 32-wide self-attention heads.

Q/K and probability/value products accumulate in FP32. Scores, the online
softmax maximum and denominator, and running output rows remain FP32; only
probabilities supplied to NAX and final outputs round to FP16. Inputs retain
the projection layout, avoiding padded heads and head-layout copies.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast

from lesiondiffusion_mlx._nax_conv import NAX_MMA_HEADER
from lesiondiffusion_mlx._typing import ArrayConstructor

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, MetalKernel, Mlx

_SIMDGROUPS = 4
_ROWS = 16

_SOURCE = r"""
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint tid = sg * 32 + lane;
const uint qid = lane >> 2;
const uint fm = (qid & 4) | ((lane >> 1) & 3);
const uint fn = ((qid & 2) | (lane & 1)) * 4;
const uint b = threadgroup_position_in_grid.y / H, h = threadgroup_position_in_grid.y % H;
constexpr uint LD = H * 32;
const uint row0 = threadgroup_position_in_grid.x * (NSG * 16) + sg * 16;
const ulong qbase = ulong(b) * L * LD + h * 32;
const ulong kbase = ulong(b) * LK * LD + h * 32;
hfrag q0, q1;
for (short i = 0; i < 2; i++) {
  const uint r = min(uint(row0 + fm + 8 * i), uint(L - 1));
  for (short j = 0; j < 4; j++) {
    const ulong at = qbase + ulong(r) * LD + fn + j;
    q0[4 * i + j] = q[at]; q1[4 * i + j] = q[at + 16];
  }
}
constexpr uint KP = 40;
threadgroup half K[32 * KP], V[32 * KP];
facc o0 = facc(0), o1 = facc(0);
float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
for (uint kb = 0; kb < LK; kb += 32) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < 32 * 8; e += NSG * 32) {
    const uint kr = e / 8, kc = (e % 8) * 4;
    const ulong at = kbase + ulong(kb + kr) * LD + kc;
    const half4 kv = *reinterpret_cast<const device half4*>(k + at);
    for (short t = 0; t < 4; t++) K[(kc + t) * KP + kr] = kv[t];
    *reinterpret_cast<threadgroup half4*>(V + kr * KP + kc) =
        *reinterpret_cast<const device half4*>(v + at);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  facc s0 = facc(0), s1 = facc(0);
  for (short dc = 0; dc < 2; dc++) {
    hfrag k0, k1;
    for (short i = 0; i < 2; i++) {
      const uint d = dc * 16 + fm + 8 * i;
      const half4 x0 = *reinterpret_cast<threadgroup half4*>(K + d * KP + fn);
      const half4 x1 = *reinterpret_cast<threadgroup half4*>(K + d * KP + 16 + fn);
      for (short j = 0; j < 4; j++) { k0[4*i+j] = x0[j]; k1[4*i+j] = x1[j]; }
    }
    const thread hfrag& a = dc == 0 ? q0 : q1;
    mma_16x32x16(s0, s1, a, k0, k1);
  }
  // Scale scores in FP32; scaling Q into half would introduce another rounding.
  s0 *= scale[0] * M_LOG2E_F;
  s1 *= scale[0] * M_LOG2E_F;
  hfrag p0, p1;
  for (short i = 0; i < 2; i++) {
    float top = -INFINITY;
    for (short j = 0; j < 4; j++) top = max(top, max(s0[4*i+j], s1[4*i+j]));
    top = max(top, simd_shuffle_xor(top, 1));
    top = max(top, simd_shuffle_xor(top, 8));
    const float next = max(m[i], top);
    const float alpha = metal::precise::exp2(m[i] - next);
    float total = 0;
    for (short j = 0; j < 4; j++) {
      const float e0 = metal::precise::exp2(s0[4*i+j] - next);
      const float e1 = metal::precise::exp2(s1[4*i+j] - next);
      total += e0 + e1;
      p0[4*i+j] = half(e0); p1[4*i+j] = half(e1);
      o0[4*i+j] *= alpha; o1[4*i+j] *= alpha;
    }
    total += simd_shuffle_xor(total, 1);
    total += simd_shuffle_xor(total, 8);
    l[i] = l[i] * alpha + total; m[i] = next;
  }
  for (short kc = 0; kc < 2; kc++) {
    hfrag v0, v1;
    for (short i = 0; i < 2; i++) {
      const uint kr = kc * 16 + fm + 8 * i;
      const half4 x0 = *reinterpret_cast<threadgroup half4*>(V + kr * KP + fn);
      const half4 x1 = *reinterpret_cast<threadgroup half4*>(V + kr * KP + 16 + fn);
      for (short j = 0; j < 4; j++) { v0[4*i+j] = x0[j]; v1[4*i+j] = x1[j]; }
    }
    const thread hfrag& a = kc == 0 ? p0 : p1;
    mma_16x32x16(o0, o1, a, v0, v1);
  }
}
for (short i = 0; i < 2; i++) {
  const uint r = row0 + fm + 8 * i;
  if (r >= L) continue;
  for (short j = 0; j < 4; j++) {
    const ulong at = qbase + ulong(r) * LD + fn + j;
    out[at] = half(o0[4*i+j] / l[i]);
    out[at + 16] = half(o1[4*i+j] / l[i]);
  }
}
"""


@lru_cache(maxsize=1)
def _kernel(mx: Mlx) -> MetalKernel:
    """Compile the direct-half NAX kernel once per runtime."""
    return mx.fast.metal_kernel(
        name="lesion_flash_attention_nax_half_d32",
        input_names=["q", "k", "v", "scale"],
        output_names=["out"],
        source=_SOURCE,
        header=NAX_MMA_HEADER,
    )


def flash_attention_half(
    q: Array, k: Array, v: Array, *, heads: int, scale: float, mx: Mlx
) -> Array:
    """Attend eligible FP16 projections and return their shape, layout and dtype.

    The caller owns NAX availability and shape dispatch: query/key lengths are
    equal and divisible by 32, with 32 channels per head. A partial final
    64-query threadgroup repeats the last query internally and discards it.
    """
    batch, length = int(q.shape[0]), int(q.shape[1])
    groups = -(-length // (_ROWS * _SIMDGROUPS))
    return _kernel(mx)(
        inputs=[q, k, v, cast(ArrayConstructor, mx.array)([scale], dtype=mx.float32)],
        template=[("L", length), ("LK", int(k.shape[1])), ("H", heads), ("NSG", _SIMDGROUPS)],
        grid=(32 * _SIMDGROUPS * groups, batch * heads, 1),
        threadgroup=(32 * _SIMDGROUPS, 1, 1),
        output_shapes=[tuple(q.shape)],
        output_dtypes=[mx.float16],
    )[0]
