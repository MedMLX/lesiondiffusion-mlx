"""FP32 flash attention for 32-wide heads, kept in the projection layout.

MLX 0.32 has no fused full-attention kernel for 32-wide FP32 heads. Using its
64-wide kernel requires zero-padding heads to 64, doing twice the arithmetic.
With TF32 disabled (medmlx-core's default, which keeps LMNet in full FP32) that
padded kernel dominates LMNet's step time. This kernel computes the same
``softmax(Q K^T * scale) V`` in FP32 without padding:

* Each simdgroup owns 8 query rows; 32-key blocks of K and V are staged in
  threadgroup memory and shared by the threadgroup's simdgroups.
* ``Q K^T`` and ``P V`` use FP32 ``simdgroup_float8x8`` multiply-accumulate.
* The online softmax keeps a running row maximum and sum in FP32, using
  ``exp2`` of log2(e)-scaled scores (folded into Q once) with precise exp2.
  Running output rows are rescaled by a diagonal 8x8 matrix multiply.

Results agree with MLX's padded FP32 kernel to FP32 rounding; summation order
differs, so outputs are not bitwise identical.

On M5-class GPUs (``nax_available``) a second kernel runs both products on the
neural accelerators in the exact split-FP16 form of ``_nax_conv``: Q (scaled),
K, V and each softmax probability ``p`` are split into an FP16 ``hi`` and a
``2^11``-scaled FP16 ``lo``; ``hi x hi`` and ``hi x lo + lo x hi`` go to
separate FP32 accumulators. Only the ``lo x lo`` term is dropped, so errors
stay at FP32 level (measured below the SIMD kernel's), at three FP16 products
per FP32 product. Each simdgroup owns 16 query rows; a ``16 x 32`` score tile
already has the register layout of ``P V``'s left operand, so probabilities
are split in registers. It runs at about the accelerators' FP16 throughput,
roughly 1.5x the SIMD kernel on LMNet's 23104-token layers.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING, cast

from lesiondiffusion_mlx._nax_conv import NAX_MMA_HEADER, nax_available
from lesiondiffusion_mlx._typing import ArrayConstructor

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, MetalKernel, Mlx

_SIMDGROUPS = 4
_BLOCK = 32  # query rows per threadgroup and keys per staged block
_NAX_ROWS = 16  # query rows per simdgroup in the NAX kernel

_SOURCE = r"""
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint tid = sg * 32 + lane;
const uint b = threadgroup_position_in_grid.y / H, h = threadgroup_position_in_grid.y % H;
const uint ld = H * 32;
const uint row0 = threadgroup_position_in_grid.x * (NSG * 8) + sg * 8;
const device float* Q = q + ulong(b) * L * ld + h * 32;
const device float* K = k + ulong(b) * LK * ld + h * 32;
const device float* V = v + ulong(b) * LK * ld + h * 32;

threadgroup float Ks[32 * 32];
threadgroup float Vs[32 * 32];
threadgroup float Ss[NSG][8 * 32];
threadgroup float Dg[NSG][64];

for (uint i = lane; i < 64; i += 32) Dg[sg][i] = 0.0f;
const float sc = scale[0] * M_LOG2E_F;
simdgroup_float8x8 qf[4];
simdgroup_float8x8 o[4];
for (short j = 0; j < 4; j++) {
  simdgroup_load(qf[j], Q + ulong(row0) * ld + j * 8, ld);
  qf[j].thread_elements()[0] *= sc;
  qf[j].thread_elements()[1] *= sc;
  o[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}
// Lane owns row r of the simdgroup's 8 rows and 8 of the block's 32 scores.
const uint r = lane / 4, c0 = (lane % 4) * 8;
float m = -INFINITY, l = 0.0f;

for (uint kb = 0; kb < LK; kb += 32) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 32 * 8; i += NSG * 32) {
    const uint kr = i / 8, kc = (i % 8) * 4;
    *reinterpret_cast<threadgroup float4*>(Ks + kr * 32 + kc) =
        *reinterpret_cast<const device float4*>(K + ulong(kb + kr) * ld + kc);
    *reinterpret_cast<threadgroup float4*>(Vs + kr * 32 + kc) =
        *reinterpret_cast<const device float4*>(V + ulong(kb + kr) * ld + kc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (short n = 0; n < 4; n++) {
    simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (short j = 0; j < 4; j++) {
      simdgroup_float8x8 kt;
      simdgroup_load(kt, Ks + n * 8 * 32 + j * 8, 32, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(s, qf[j], kt, s);
    }
    simdgroup_store(s, Ss[sg] + n * 8, 32);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float* row = Ss[sg] + r * 32 + c0;
  float values[8];
  float local = -INFINITY;
  for (short j = 0; j < 8; j++) { values[j] = row[j]; local = max(local, values[j]); }
  local = max(local, simd_shuffle_xor(local, 1));
  local = max(local, simd_shuffle_xor(local, 2));
  const float m_new = max(m, local);
  const float alpha = metal::precise::exp2(m - m_new);
  float total = 0.0f;
  for (short j = 0; j < 8; j++) {
    const float p = metal::precise::exp2(values[j] - m_new);
    total += p;
    row[j] = p;
  }
  total += simd_shuffle_xor(total, 1);
  total += simd_shuffle_xor(total, 2);
  l = l * alpha + total;
  m = m_new;
  if ((lane % 4) == 0) Dg[sg][r * 9] = alpha;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 dg;
  simdgroup_load(dg, Dg[sg], 8);
  for (short d = 0; d < 4; d++) { simdgroup_float8x8 t; simdgroup_multiply(t, dg, o[d]); o[d] = t; }
  for (short kk = 0; kk < 4; kk++) {
    simdgroup_float8x8 pf;
    simdgroup_load(pf, Ss[sg] + kk * 8, 32);
    for (short d = 0; d < 4; d++) {
      simdgroup_float8x8 vf;
      simdgroup_load(vf, Vs + kk * 8 * 32 + d * 8, 32);
      simdgroup_multiply_accumulate(o[d], pf, vf, o[d]);
    }
  }
}
simdgroup_barrier(mem_flags::mem_threadgroup);
if ((lane % 4) == 0) Dg[sg][r * 9] = 1.0f / l;
simdgroup_barrier(mem_flags::mem_threadgroup);
simdgroup_float8x8 dg;
simdgroup_load(dg, Dg[sg], 8);
for (short d = 0; d < 4; d++) {
  simdgroup_float8x8 t;
  simdgroup_multiply(t, dg, o[d]);
  simdgroup_store(t, out + ulong(b) * L * ld + ulong(row0) * ld + h * 32 + d * 8, ld);
}
"""


_NAX_SOURCE = r"""
// Fragment element 4 i + j is row fm + 8 i, column fn + j of a 16x16 tile, for
// left and right operands and the accumulator alike (see mma_16x32x16).
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

// Q hi/lo for head dims 0-15 (q*0) and 16-31 (q*1); rows past L repeat the
// last row and are not stored.
hfrag qh0, qh1, ql0, ql1;
for (short i = 0; i < 2; i++) {
  const uint r = min(uint(row0 + fm + 8 * i), uint(L - 1));
  for (short j = 0; j < 4; j++) {
    const ulong at = qbase + ulong(r) * LD + fn + j;
    qh0[4 * i + j] = qh[at]; qh1[4 * i + j] = qh[at + 16];
    ql0[4 * i + j] = ql[at]; ql1[4 * i + j] = ql[at + 16];
  }
}
// 32-key blocks: K transposed (head dim major) and V key major, padded rows.
constexpr uint KP = 40;
threadgroup half Kh[32 * KP], Kl[32 * KP], Vh[32 * KP], Vl[32 * KP];
// o: hi x hi output sums; p: 2^11-scaled cross terms.
facc o0 = facc(0), o1 = facc(0), p0 = facc(0), p1 = facc(0);
float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

for (uint kb = 0; kb < LK; kb += 32) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < 32 * 8; e += NSG * 32) {
    const uint kr = e / 8, kc = (e % 8) * 4;
    const ulong at = kbase + ulong(kb + kr) * LD + kc;
    const half4 k_h = *reinterpret_cast<const device half4*>(kh + at);
    const half4 k_l = *reinterpret_cast<const device half4*>(kl + at);
    for (short t = 0; t < 4; t++) {
      Kh[(kc + t) * KP + kr] = k_h[t];
      Kl[(kc + t) * KP + kr] = k_l[t];
    }
    *reinterpret_cast<threadgroup half4*>(Vh + kr * KP + kc) =
        *reinterpret_cast<const device half4*>(vh + at);
    *reinterpret_cast<threadgroup half4*>(Vl + kr * KP + kc) =
        *reinterpret_cast<const device half4*>(vl + at);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Scores of keys kb + 0-15 (s0) and kb + 16-31 (s1), log2(e)-scaled via Q.
  facc s0 = facc(0), s1 = facc(0), t0 = facc(0), t1 = facc(0);
  for (short dc = 0; dc < 2; dc++) {
    hfrag bh0, bh1, bl0, bl1;
    for (short i = 0; i < 2; i++) {
      const uint d = dc * 16 + fm + 8 * i;
      const half4 x0 = *reinterpret_cast<threadgroup half4*>(Kh + d * KP + fn);
      const half4 x1 = *reinterpret_cast<threadgroup half4*>(Kh + d * KP + 16 + fn);
      const half4 y0 = *reinterpret_cast<threadgroup half4*>(Kl + d * KP + fn);
      const half4 y1 = *reinterpret_cast<threadgroup half4*>(Kl + d * KP + 16 + fn);
      for (short j = 0; j < 4; j++) {
        bh0[4 * i + j] = x0[j]; bh1[4 * i + j] = x1[j];
        bl0[4 * i + j] = y0[j]; bl1[4 * i + j] = y1[j];
      }
    }
    const thread hfrag& a_h = dc == 0 ? qh0 : qh1;
    const thread hfrag& a_l = dc == 0 ? ql0 : ql1;
    mma_16x32x16(s0, s1, a_h, bh0, bh1);
    mma_16x32x16(t0, t1, a_h, bl0, bl1);
    mma_16x32x16(t0, t1, a_l, bh0, bh1);
  }
  s0 += t0 * (1.0f / 2048.0f);
  s1 += t1 * (1.0f / 2048.0f);
  // Online softmax per row fm + 8 i; a row's 32 scores span lanes xor 1 and 8.
  hfrag ph0, ph1, pl0, pl1;
  for (short i = 0; i < 2; i++) {
    float top = -INFINITY;
    for (short j = 0; j < 4; j++) top = max(top, max(s0[4 * i + j], s1[4 * i + j]));
    top = max(top, simd_shuffle_xor(top, 1));
    top = max(top, simd_shuffle_xor(top, 8));
    const float m_new = max(m[i], top);
    const float alpha = metal::precise::exp2(m[i] - m_new);
    float total = 0.0f;
    for (short j = 0; j < 4; j++) {
      const float e0 = metal::precise::exp2(s0[4 * i + j] - m_new);
      const float e1 = metal::precise::exp2(s1[4 * i + j] - m_new);
      total += e0 + e1;
      const half h0 = half(e0), h1 = half(e1);
      ph0[4 * i + j] = h0; ph1[4 * i + j] = h1;
      pl0[4 * i + j] = half((e0 - float(h0)) * 2048.0f);
      pl1[4 * i + j] = half((e1 - float(h1)) * 2048.0f);
    }
    total += simd_shuffle_xor(total, 1);
    total += simd_shuffle_xor(total, 8);
    l[i] = l[i] * alpha + total;
    m[i] = m_new;
    for (short j = 0; j < 4; j++) {
      o0[4 * i + j] *= alpha; o1[4 * i + j] *= alpha;
      p0[4 * i + j] *= alpha; p1[4 * i + j] *= alpha;
    }
  }
  // O += P V over keys kb + 16 kc + 0-15.
  for (short kc = 0; kc < 2; kc++) {
    hfrag vh0, vh1, vl0, vl1;
    for (short i = 0; i < 2; i++) {
      const uint kr = kc * 16 + fm + 8 * i;
      const half4 x0 = *reinterpret_cast<threadgroup half4*>(Vh + kr * KP + fn);
      const half4 x1 = *reinterpret_cast<threadgroup half4*>(Vh + kr * KP + 16 + fn);
      const half4 y0 = *reinterpret_cast<threadgroup half4*>(Vl + kr * KP + fn);
      const half4 y1 = *reinterpret_cast<threadgroup half4*>(Vl + kr * KP + 16 + fn);
      for (short j = 0; j < 4; j++) {
        vh0[4 * i + j] = x0[j]; vh1[4 * i + j] = x1[j];
        vl0[4 * i + j] = y0[j]; vl1[4 * i + j] = y1[j];
      }
    }
    const thread hfrag& a_h = kc == 0 ? ph0 : ph1;
    const thread hfrag& a_l = kc == 0 ? pl0 : pl1;
    mma_16x32x16(o0, o1, a_h, vh0, vh1);
    mma_16x32x16(p0, p1, a_h, vl0, vl1);
    mma_16x32x16(p0, p1, a_l, vh0, vh1);
  }
}
for (short i = 0; i < 2; i++) {
  const uint r = row0 + fm + 8 * i;
  if (r >= L) continue;
  const float inv = 1.0f / l[i];
  for (short j = 0; j < 4; j++) {
    const ulong at = qbase + ulong(r) * LD + fn + j;
    out[at] = (o0[4 * i + j] + p0[4 * i + j] * (1.0f / 2048.0f)) * inv;
    out[at + 16] = (o1[4 * i + j] + p1[4 * i + j] * (1.0f / 2048.0f)) * inv;
  }
}
"""


def flash_eligible(q: Array, k: Array, heads: int, mx: Mlx) -> bool:
    """Whether projections ``(B, L, heads * 32)`` fit the kernel.

    Requires FP32, 32-wide heads, and query and key lengths divisible by 32.
    """
    return (
        q.dtype == mx.float32
        and k.dtype == mx.float32
        and int(q.shape[-1]) == heads * 32
        and int(k.shape[-1]) == heads * 32
        and int(q.shape[1]) % _BLOCK == 0
        and int(k.shape[1]) % _BLOCK == 0
    )


@lru_cache(maxsize=1)
def _kernel(mx: Mlx) -> MetalKernel:
    """Build the FP32 SIMD kernel once; used when the NAX kernel is unavailable."""
    return mx.fast.metal_kernel(
        name="lesion_flash_attention_fp32_d32",
        input_names=["q", "k", "v", "scale"],
        output_names=["out"],
        source=_SOURCE,
    )


@lru_cache(maxsize=1)
def _nax_kernel(mx: Mlx) -> MetalKernel:
    """Build the NAX kernel once; it takes hi/lo FP16 splits of q, k and v (see ``_split``)."""
    return mx.fast.metal_kernel(
        name="lesion_flash_attention_nax_split_d32",
        input_names=["qh", "ql", "kh", "kl", "vh", "vl"],
        output_names=["out"],
        source=_NAX_SOURCE,
        header=NAX_MMA_HEADER,
    )


def _split(x: Array, mx: Mlx) -> tuple[Array, Array]:
    """FP32 ``x`` as FP16 ``hi = half(x)`` and ``lo = half((x - hi) * 2^11)``."""
    hi = x.astype(mx.float16)
    return hi, ((x - hi.astype(mx.float32)) * 2048.0).astype(mx.float16)


def flash_attention(q: Array, k: Array, v: Array, *, heads: int, scale: float, mx: Mlx) -> Array:
    """Multi-head attention on head-interleaved projections.

    Args:
        q: Queries ``(B, L, heads * 32)``; head ``h`` is columns ``32 h`` to ``32 h + 31``.
        k: Keys ``(B, LK, heads * 32)``.
        v: Values ``(B, LK, heads * 32)``.
        heads: Number of heads.
        scale: Multiplier applied to ``Q K^T`` before the softmax.
        mx: The imported ``mlx.core`` module.

    Returns:
        ``(B, L, heads * 32)`` FP32 attention output in the same head layout,
        ready for the output projection. On M5-class GPUs the NAX split-FP16
        kernel computes it, which needs ``|q * scale * log2(e)|``, ``|k|`` and
        ``|v|`` below 65504; these ranges are not checked. When ``nax_available``
        is false, the FP32 SIMD kernel computes it.
    """
    batch, length = int(q.shape[0]), int(q.shape[1])
    if nax_available(mx):
        (qh, ql), (kh, kl), (vh, vl) = (
            _split(q * (scale * math.log2(math.e)), mx),
            _split(k, mx),
            _split(v, mx),
        )
        groups = -(-length // (_NAX_ROWS * _SIMDGROUPS))
        return _nax_kernel(mx)(
            inputs=[qh, ql, kh, kl, vh, vl],
            template=[("L", length), ("LK", int(k.shape[1])), ("H", heads), ("NSG", _SIMDGROUPS)],
            grid=(32 * _SIMDGROUPS * groups, batch * heads, 1),
            threadgroup=(32 * _SIMDGROUPS, 1, 1),
            output_shapes=[tuple(q.shape)],
            output_dtypes=[mx.float32],
        )[0]
    return _kernel(mx)(
        inputs=[q, k, v, cast(ArrayConstructor, mx.array)([scale], dtype=mx.float32)],
        template=[("L", length), ("LK", int(k.shape[1])), ("H", heads), ("NSG", _SIMDGROUPS)],
        grid=(32 * _SIMDGROUPS * (length // _BLOCK), batch * heads, 1),
        threadgroup=(32 * _SIMDGROUPS, 1, 1),
        output_shapes=[tuple(q.shape)],
        output_dtypes=[mx.float32],
    )[0]
