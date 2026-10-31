"""Implicit-GEMM 3x3x3 float16 convolution on the M5 GPU neural accelerators.

MLX 0.32's ``conv3d`` reaches roughly a third of the FP16 matrix throughput
that its own NAX matmul reaches on M5-class GPUs, and LINet/VQ spend most of
their time in 3x3x3 convolutions. This module computes one as a GEMM over a
zero-padded, flattened NDHWC copy of the input: with row pitch ``Q = W + 2``
and slice pitch ``P = (H + 2) * Q``, kernel tap ``(kd, kh, kw)`` of the
window starting at padded row ``r`` reads row ``r + kd * P + kh * Q + kw``, so
no im2col buffer exists. GEMM rows are output voxels, each with its own
window start, so padding rows are never computed.

A pack kernel builds the padded copy from NCDHW input in one pass and can
apply a per-channel affine (group-norm statistics) and SiLU on the way, so a
normalize-SiLU-convolve sequence never stores the normalized tensor. Each
simdgroup of the GEMM owns a 32x32 output tile and accumulates all products
in FP32 through Metal Performance Primitives ``matmul2d``; bias, an optional
per-batch shift and an optional residual are added to the FP32 sum, and the
result is rounded to FP16 once and stored as NCDHW. MLX's FP16 conv followed
by separate additions rounds after each, so results differ from ``mx.conv3d``
by FP16 rounding, not by accumulation precision.

The same kernel handles stride 2 and, with weights from
``pack_upsample_weight``, a convolution of a nearest 2x upsampling without
forming it.

FP32 inputs and weights use an exact mode instead, for networks kept in FP32
(LMNet): each value ``v`` is split into ``hi = half(v)`` and ``lo =
half((v - hi) * 2^11)``, and the GEMM sums ``hi x hi`` in one FP32 accumulator
and ``hi x lo + lo x hi`` in another, scaled back by ``2^-11`` before the FP32
store. FP16 products are exact in FP32, so the only error beyond FP32
accumulation is the dropped ``lo x lo`` term and the 22-bit split, about
``2^-22`` relative: near FP32, unlike TF32's ``2^-11``. It costs three FP16
products per FP32 product, still faster than MLX's FP32 ``conv3d``. Exact mode
needs ``|v| < 65504``, the FP16 range of ``hi``.
"""

from __future__ import annotations

import math
import os
import platform
import re
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, MetalKernel, Mlx

# Set to "0" to force MLX's own conv3d and the SIMD FP32 attention kernel, e.g.
# to compare against the NAX kernels.
DISABLE_ENV = "LESIONDIFFUSION_NAX_CONV"

# Shared with the NAX attention kernel in _attention.
NAX_MMA_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
typedef vec<float, 8> facc;
typedef vec<half, 8> hfrag;

// One 16x32x16 multiply-accumulate. Fragment element 4 i + j of a 16x16 tile
// is row fm + 8 i, column fn + j for the lane coordinates computed below; this
// is the register layout MLX's NAX GEMM feeds to the same descriptor.
METAL_FUNC void mma_16x32x16(thread facc& c0, thread facc& c1, thread const hfrag& a,
                             thread const hfrag& b0, thread const hfrag& b1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, false, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
  auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
  auto cc = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
  for (short i = 0; i < 8; i++) {
    ca[i] = a[i]; cb[i] = b0[i]; cb[8 + i] = b1[i]; cc[i] = c0[i]; cc[8 + i] = c1[i];
  }
  op.run(ca, cb, cc);
  for (short i = 0; i < 8; i++) { c0[i] = cc[i]; c1[i] = cc[8 + i]; }
}

// 32x16x16 for 16-column tiles (MPP needs a 32 among M, N, K): the left
// operand and destination stack two 16-row fragments of the layout above.
METAL_FUNC void mma_32x16x16(thread facc& c0, thread facc& c1, thread const hfrag& a0,
                             thread const hfrag& a1, thread const hfrag& b) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      32, 16, 16, false, false, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
  auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
  auto cc = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
  for (short i = 0; i < 8; i++) {
    ca[i] = a0[i]; ca[8 + i] = a1[i]; cb[i] = b[i]; cc[i] = c0[i]; cc[8 + i] = c1[i];
  }
  op.run(ca, cb, cc);
  for (short i = 0; i < 8; i++) { c0[i] = cc[i]; c1[i] = cc[8 + i]; }
}

METAL_FUNC hfrag load_frag(const device half* p, uint ld) {
  hfrag f;
  for (short i = 0; i < 2; i++) {
    half4 v = *reinterpret_cast<const device half4*>(p + i * 8 * ld);
    f[i * 4 + 0] = v.x; f[i * 4 + 1] = v.y; f[i * 4 + 2] = v.z; f[i * 4 + 3] = v.w;
  }
  return f;
}
"""

_PACK = r"""
// Threadgroup (padded line, channel chunk, batch item) writes the Q padded rows
// of one (d, h) line for CB of the C >= CIN packed channels; channels from CIN
// up are zero. Interior lines stage the W voxels through threadgroup memory:
// reads walk W along NCDHW rows, writes walk channels along NDHWC rows, so both
// device accesses are contiguous. With EXACT a row holds 2 C halves: FP16 hi =
// half(y) in channels [0, C) and lo = half((y - hi) * 2^11) in [C, 2 C).
const uint line = threadgroup_position_in_grid.x, c0 = threadgroup_position_in_grid.y * CB;
const uint b = threadgroup_position_in_grid.z, tid = thread_position_in_threadgroup.x;
const uint d = line / (H + 2), h = line % (H + 2);
const bool interior = d >= 1 && d <= D && h >= 1 && h <= H;
constexpr uint LD = CB + 2;
constexpr uint XC = EXACT ? 2 * C : C;
threadgroup half tile[W * LD];
threadgroup half tile_lo[EXACT ? W * LD : 1];
if (interior) {
  const ulong base = ulong(b * CIN + c0) * (D * H * W) + ((d - 1) * H + (h - 1)) * W;
  for (uint e = tid; e < CB * W; e += TG) {
    const uint c = e / W, ww = e % W;
    float y = 0.0f;
    if (c0 + c < CIN) {
      y = float(x[base + ulong(c) * (D * H * W) + ww]);
      if (AFFINE) {
        y = y * affine[(b * CIN + c0 + c) * 2] + affine[(b * CIN + c0 + c) * 2 + 1];
      }
      if (SILU) y = y / (1.0f + metal::precise::exp(-y));
    }
    const half hi = half(y);
    tile[ww * LD + c] = hi;
    if (EXACT) tile_lo[ww * LD + c] = half((y - float(hi)) * 2048.0f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint e = tid; e < Q * (CB / 8); e += TG) {
  const uint j = e / (CB / 8), cv = (e % (CB / 8)) * 8;
  vec<half, 8> v = vec<half, 8>(0), v_lo = vec<half, 8>(0);
  if (interior && j >= 1 && j <= W) {
    for (short k = 0; k < 8; k++) v[k] = tile[(j - 1) * LD + cv + k];
    if (EXACT) {
      for (short k = 0; k < 8; k++) v_lo[k] = tile_lo[(j - 1) * LD + cv + k];
    }
  }
  device half* row = out + (ulong(b) * ROWS + line * Q + j) * XC + c0 + cv;
  *reinterpret_cast<device vec<half, 8>*>(row) = v;
  if (EXACT) *reinterpret_cast<device vec<half, 8>*>(row + C) = v_lo;
}
"""

_SOURCE = r"""
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint qid = lane >> 2;
const uint fm = (qid & 4) | ((lane >> 1) & 3);
const uint fn = ((qid & 2) | (lane & 1)) * 4;
const uint z = threadgroup_position_in_grid.z;
const uint row0 = threadgroup_position_in_grid.y * (WM * 32) + (sg / WN) * 32;
// TN: simdgroup tile width, 32 columns, or 16 when O' is 16.
const uint col0 = threadgroup_position_in_grid.x * (WN * TN) + (sg % WN) * TN;
constexpr uint S = D * H * W;
// UP: grid z enumerates (batch item, output parity) and each tap axis has K = 2
// taps; otherwise z is the batch item, K = 3 and the parity is zero. With
// SPLIT > 1, z also enumerates SPLIT equal ranges of taps.
const uint split = z % SPLIT, zb = z / SPLIT;
const uint b = UP ? zb / 8 : zb, phase = UP ? zb % 8 : 0;
const uint pd = phase >> 2, ph = (phase >> 1) & 1, pw = phase & 1;

// Output rows are voxels v = (d * H + h) * W + w of the D x H x W output grid.
// A simdgroup owns 32 rows: fragment rows fm + 8 i, i = 0..3, start their
// windows at padded input row (STRIDE d + pd) * P + (STRIDE h + ph) * Q +
// STRIDE w + pw; rows past the volume reuse the last voxel and are not stored.
// EXACT input rows hold hi then lo channels (pitch 2 C); EXACT weights hold
// the hi plane then the lo plane, K^3 C rows each, per parity.
constexpr uint XC = EXACT ? 2 * C : C;
constexpr ulong WLO = ulong(K * K * K * C) * O;
const device half* xb = x + ulong(b) * ROWS * XC;
const device half* a[4];
for (short i = 0; i < 4; i++) {
  const uint v = min(row0 + fm + i * 8, S - 1);
  const uint row = (STRIDE * (v / (H * W)) + pd) * P + (STRIDE * ((v / W) % H) + ph) * Q
                   + STRIDE * (v % W) + pw;
  a[i] = xb + ulong(row) * XC + fn;
}
// Named pointers and accumulators: indexing arrays of them inside the loop
// spills to the stack and runs several times slower.
const device half* a0 = a[0];
const device half* a1 = a[1];
const device half* a2 = a[2];
const device half* a3 = a[3];
const device half* wp = w + ulong(phase) * (EXACT ? 2 : 1) * WLO;
facc c00 = facc(0), c01 = facc(0), c10 = facc(0), c11 = facc(0);
// EXACT: hi x lo and lo x hi products, both 2^11 times their true value.
facc l00 = facc(0), l01 = facc(0), l10 = facc(0), l11 = facc(0);
constexpr uint SPLIT_TAPS = K * K * K / SPLIT;
// Accumulate both 16-row A fragments times B into the 32x32 (or 32x16) tile.
#define MMA(t00, t01, t10, t11, x0, x1, y0, y1) \
  if (TN == 32) {                                \
    mma_16x32x16(t00, t01, x0, y0, y1);          \
    mma_16x32x16(t10, t11, x1, y0, y1);          \
  } else {                                       \
    mma_32x16x16(t00, t10, x0, x1, y0);          \
  }
for (uint tap = split * SPLIT_TAPS; tap < (split + 1) * SPLIT_TAPS; tap++) {
  const uint offset = ((tap / (K * K)) * P + ((tap / K) % K) * Q + (tap % K)) * XC;
  const device half* bw = wp + ulong(tap * C + fm) * O + col0 + fn;
  for (uint k = 0; k < C; k += 16) {
    // Both 16-row A fragments reuse each 16x32 B fragment.
    const hfrag fb0 = load_frag(bw + ulong(k) * O, O);
    const hfrag fb1 = TN == 32 ? load_frag(bw + ulong(k) * O + 16, O) : hfrag(0);
    const half4 v0 = *reinterpret_cast<const device half4*>(a0 + offset + k);
    const half4 v1 = *reinterpret_cast<const device half4*>(a1 + offset + k);
    const half4 v2 = *reinterpret_cast<const device half4*>(a2 + offset + k);
    const half4 v3 = *reinterpret_cast<const device half4*>(a3 + offset + k);
    const hfrag f0 = hfrag(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
    const hfrag f1 = hfrag(v2.x, v2.y, v2.z, v2.w, v3.x, v3.y, v3.z, v3.w);
    MMA(c00, c01, c10, c11, f0, f1, fb0, fb1);
    if (EXACT) {
      const hfrag lb0 = load_frag(bw + WLO + ulong(k) * O, O);
      const hfrag lb1 = TN == 32 ? load_frag(bw + WLO + ulong(k) * O + 16, O) : hfrag(0);
      const half4 u0 = *reinterpret_cast<const device half4*>(a0 + offset + C + k);
      const half4 u1 = *reinterpret_cast<const device half4*>(a1 + offset + C + k);
      const half4 u2 = *reinterpret_cast<const device half4*>(a2 + offset + C + k);
      const half4 u3 = *reinterpret_cast<const device half4*>(a3 + offset + C + k);
      const hfrag g0 = hfrag(u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w);
      const hfrag g1 = hfrag(u2.x, u2.y, u2.z, u2.w, u3.x, u3.y, u3.z, u3.w);
      MMA(l00, l01, l10, l11, f0, f1, lb0, lb1);
      MMA(l00, l01, l10, l11, g0, g1, fb0, fb1);
    }
  }
}

// NCDHW store of the first OUT of O columns: for each column, the 8 fm lanes
// write 8 consecutive voxels (every other voxel of a doubled row when UP).
// Bias (per batch item when BATCH_BIAS) and the optional residual are added
// to the FP32 sum before the single FP16 rounding. With SPLIT > 1 each tap
// range instead stores its raw FP32 sum to plane split of the partials; the
// reduction kernel adds the planes, bias and residual. EXACT outputs are
// FP32 (OT), with the scaled lo products folded in before any addition.
constexpr uint SO = UP ? 8 * S : S;
const auto bias_b = bias + (BATCH_BIAS ? b * OUT : 0);
for (short i = 0; i < 4; i++) {
  const uint v = row0 + fm + i * 8;
  if (v >= S) continue;
  const uint o = UP ? ((2 * (v / (H * W)) + pd) * (2 * H) + 2 * ((v / W) % H) + ph) * (2 * W)
                          + 2 * (v % W) + pw
                    : v;
  const thread facc& lo = i < 2 ? c00 : c10;
  const thread facc& hi = i < 2 ? c01 : c11;
  const thread facc& lo_s = i < 2 ? l00 : l10;
  const thread facc& hi_s = i < 2 ? l01 : l11;
  const ulong index = ulong(b * OUT + col0 + fn) * SO + o;
  const ulong plane = ulong(split) * NB * OUT * SO;
  for (short j = 0; j < 4; j++) {
    for (short half_index = 0; half_index < TN / 16; half_index++) {
      const uint col = col0 + half_index * 16 + fn + j;
      if (OUT != O && col >= OUT) continue;
      const ulong at = index + ulong(half_index * 16 + j) * SO;
      float sum = (half_index == 0 ? lo : hi)[(i % 2) * 4 + j];
      if (EXACT) sum += (half_index == 0 ? lo_s : hi_s)[(i % 2) * 4 + j] * (1.0f / 2048.0f);
      if (SPLIT > 1) {
        out[plane + at] = sum;
      } else {
        out[at] = OT(sum + float(bias_b[col]) + (RESIDUAL ? float(res[at]) : 0.0f));
      }
    }
  }
}
"""

_REDUCE = r"""
// Four consecutive outputs per thread: sum the SPLIT FP32 partial planes, add
// bias and the optional residual, and round once to the output type OT.
const uint i = thread_position_in_grid.x * 4;
if (i >= N) return;
const uint o = (i / SPATIAL) % OUT, b = i / (SPATIAL * OUT);
float4 sum = float4(float(bias[(BATCH_BIAS ? b * OUT : 0) + o]));
for (uint k = 0; k < SPLIT; k++) {
  sum += *reinterpret_cast<const device float4*>(partials + ulong(k) * N + i);
}
if (RESIDUAL) sum += float4(res[i], res[i + 1], res[i + 2], res[i + 3]);
*reinterpret_cast<device vec<OT, 4>*>(out + i) = vec<OT, 4>(sum);
"""

_SIMDGROUP_ROWS = 8
# Split the 27 taps three ways when fewer threadgroups than this would cover
# the output and C is large; M5 Max's 40 GPU cores are otherwise underfilled
# at LINet's 16^3 level (64 threadgroups), and the reduction pass is cheap
# next to C >= 512 products per output.
_SPLIT_BELOW_THREADGROUPS = 96
_SPLIT_MIN_CHANNELS = 512
_PACK_THREADS = 256


def nax_available(mx: Mlx) -> bool:
    """Return whether this process should use the NAX convolution.

    Requires an Apple GPU of generation 17 or newer (M5 family), which carries
    the neural accelerators ``matmul2d`` targets, and macOS 26.2 or newer for
    Metal 4 tensor operations. ``LESIONDIFFUSION_NAX_CONV=0`` disables it and
    the NAX attention kernel.
    """
    if os.environ.get(DISABLE_ENV, "1") == "0" or not mx.metal.is_available():
        return False
    match = re.fullmatch(r"applegpu_g(\d+)\w*", str(mx.device_info().get("architecture", "")))
    return match is not None and int(match.group(1)) >= 17 and _macos_release() >= (26, 2)


@lru_cache(maxsize=1)
def _macos_release() -> tuple[int, ...]:
    """Major and minor macOS version; ``platform.mac_ver`` reads a plist on every call."""
    return tuple(int(part) for part in (platform.mac_ver()[0] or "0").split(".")[:2])


def nax_eligible(weight: Array) -> bool:
    """Whether a Torch-layout ``(O, C, KD, KH, KW)`` FP16 or FP32 weight fits the kernel.

    The kernel handles 3x3x3 filters with padding 1 and stride 1 or 2 (or 1
    after a nearest 2x upsampling); packing zero-pads ``C`` to a multiple of
    16, and ``O`` to 16 when ``O <= 16`` or otherwise to a multiple of 32.
    """
    return weight.ndim == 5 and tuple(weight.shape[2:]) == (3, 3, 3)


def _pad_channels(weight: Array, mx: Mlx) -> Array:
    """Zero-pad an ``(O, C, 3, 3, 3)`` weight to the GEMM tile sizes (C to 16, O as below)."""
    out_channels, in_channels = int(weight.shape[0]), int(weight.shape[1])
    # At most 16 outputs use 16-column tiles; LMNet's full-resolution layers
    # have 16 (and its last layer 1), which 32 columns would half waste.
    out_padding = 16 - out_channels if out_channels <= 16 else -out_channels % 32
    return mx.pad(
        weight,
        [(0, out_padding), (0, -in_channels % 16), (0, 0), (0, 0), (0, 0)],
    )


def _split_planes(taps: Array, mx: Mlx) -> Array:
    """Stack FP32 ``(..., R, O)`` as FP16 ``(..., 2 R, O)``: hi rows, then 2^11-scaled lo rows."""
    hi = taps.astype(mx.float16)
    lo = ((taps - hi.astype(mx.float32)) * 2048.0).astype(mx.float16)
    return mx.concatenate([hi, lo], axis=-2)


def pack_weight(weight: Array, mx: Mlx) -> Array:
    """Reorder a Torch ``(O, C, 3, 3, 3)`` weight to contiguous ``(27 * C', O')`` taps.

    ``C'`` rounds ``C`` up to a multiple of 16; ``O'`` is 16 when ``O <= 16`` and
    otherwise rounds ``O`` up to a multiple of 32. Padding uses zero weights.
    Row ``tap * C' + c`` holds input channel ``c`` of tap
    ``kd * 9 + kh * 3 + kw``. An FP16 weight packs to FP16 of that shape; an
    FP32 weight packs for exact mode to FP16 ``(2 * 27 * C', O')``, the hi
    plane followed by the ``2^11``-scaled lo plane.
    """
    exact = weight.dtype == mx.float32
    weight = _pad_channels(weight, mx)
    out_channels, in_channels = int(weight.shape[0]), int(weight.shape[1])
    taps = mx.reshape(mx.transpose(weight, (2, 3, 4, 1, 0)), (27 * in_channels, out_channels))
    return mx.contiguous(_split_planes(taps, mx) if exact else taps)


def pack_upsample_weight(weight: Array, mx: Mlx) -> Array:
    """Fold a Torch ``(O, C, 3, 3, 3)`` weight into eight 2x2x2 parity filters.

    After nearest 2x upsampling, output index ``2 m + p`` of one axis reads
    upsampled ``2 m + p - 1 + k`` for taps ``k = 0, 1, 2``, i.e. source index
    ``m - 1, m, m`` for ``p = 0`` and ``m, m, m + 1`` for ``p = 1``. Taps that
    read the same source voxel merge, so each parity is a 2x2x2 convolution
    of the source over padded offsets ``p, p + 1``: 8/27 of the products.

    Merged weights are summed in FP32 and rounded to FP16 once, so they can
    differ from the exact sum by half an FP16 ulp. An FP32 weight instead
    keeps the FP32 sums and splits each parity into hi and lo planes, as
    ``pack_weight`` does.

    Returns:
        Contiguous ``(8, 8 * C', O')`` float16, channels padded as by
        ``pack_weight``; parity ``pd * 4 + ph * 2 + pw``, row
        ``(td * 4 + th * 2 + tw) * C' + c``. For an FP32 weight,
        ``(8, 2 * 8 * C', O')`` with each parity's hi rows before its lo rows.
    """
    exact = weight.dtype == mx.float32
    weight = _pad_channels(weight, mx)
    out_channels, in_channels = int(weight.shape[0]), int(weight.shape[1])

    # Fold one leading 3-tap axis into leading (parity p, tap t) axes: tap k of
    # parity p reads padded source offset p + t. Plain FP32 additions, not an
    # einsum, whose matmul could run in TF32 under MLX_ENABLE_TF32.
    def fold(taps: Array) -> Array:
        """Fold one upsampling tap into packed convolution weights."""
        k0, k1, k2 = taps[0], taps[1], taps[2]
        return mx.stack([mx.stack([k0, k1 + k2]), mx.stack([k0 + k1, k2])])

    folded = mx.transpose(weight.astype(mx.float32), (2, 3, 4, 1, 0))  # (kd, kh, kw, C, O)
    folded = fold(folded)  # (pd, td, kh, kw, C, O)
    folded = fold(mx.transpose(folded, (2, 0, 1, 3, 4, 5)))  # (ph, th, pd, td, kw, C, O)
    folded = fold(mx.transpose(folded, (4, 0, 1, 2, 3, 5, 6)))  # (pw, tw, ph, th, pd, td, ...)
    folded = mx.transpose(folded, (4, 2, 0, 5, 3, 1, 6, 7))  # (pd, ph, pw, td, th, tw, C, O)
    folded = mx.reshape(folded, (8, 8 * in_channels, out_channels))
    return mx.contiguous(_split_planes(folded, mx) if exact else folded.astype(mx.float16))


@lru_cache(maxsize=1)
def _kernels(mx: Mlx) -> tuple[MetalKernel, MetalKernel, MetalKernel]:
    """Reuse the kernel objects; MLX caches their specializations by template."""
    return (
        mx.fast.metal_kernel(
            name="lesion_nax_conv_pack",
            input_names=["x", "affine"],
            output_names=["out"],
            source=_PACK,
        ),
        mx.fast.metal_kernel(
            name="lesion_nax_conv3x3x3",
            input_names=["x", "w", "bias", "res"],
            output_names=["out"],
            source=_SOURCE,
            header=NAX_MMA_HEADER,
        ),
        mx.fast.metal_kernel(
            name="lesion_nax_conv_reduce",
            input_names=["partials", "bias", "res"],
            output_names=["out"],
            source=_REDUCE,
        ),
    )


def nax_conv3d(
    x: Array,
    packed: Array,
    bias: Array,
    *,
    affine: Array | None = None,
    silu: bool = False,
    stride: int = 1,
    upsample: bool = False,
    residual: Array | None = None,
    mx: Mlx,
) -> Array:
    """Padding-1 3x3x3 convolution of NCDHW ``x``, in its dtype.

    Float16 ``x`` rounds its prepared input and output to FP16 once each.
    Float32 ``x`` uses the exact split-FP16 mode described in the module
    docstring and returns float32.

    Args:
        x: Input ``(B, C, D, H, W)``, float16 or float32.
        packed: Weight from ``pack_weight``, or with ``upsample`` from
            ``pack_upsample_weight``, packed from an FP16 weight for FP16
            ``x`` and from an FP32 weight for FP32 ``x``.
        bias: Per-output-channel bias ``(O,)``, or ``(B, O)`` for a bias per
            batch item; its last axis is the output channel count. Added in
            FP32 before rounding.
        affine: Optional FP32 ``(B * C * 2)`` per-channel ``(scale, offset)``
            pairs, e.g. group-norm statistics from ``group_norm_affine``,
            applied to ``x`` in FP32 before the convolution.
        silu: Apply SiLU after ``affine``. FP16 mode rounds the prepared
            input once to FP16, as a separate normalize-SiLU kernel would.
        stride: 1 or 2, on every spatial axis.
        upsample: Convolve the nearest-neighbour 2x upsampling of ``x``
            without materializing it (stride 1 only).
        residual: Optional NCDHW tensor of the output's shape, added in FP32
            before rounding.
        mx: The imported ``mlx.core`` module.

    Returns:
        Contiguous NCDHW output ``(B, O, D', H', W')`` in ``x``'s dtype: the
        input size for stride 1, ``(n - 1) // 2 + 1`` per axis for stride 2,
        and twice the input size with ``upsample``.

    Raises:
        ValueError: If ``packed`` does not match ``x``'s channels and dtype, or
            ``residual`` does not have the output's shape.
    """
    batch, in_channels, d, h, w = (int(size) for size in x.shape)
    exact = x.dtype == mx.float32
    dtype = mx.float32 if exact else mx.float16
    taps = 8 if upsample else 27
    channels = in_channels + -in_channels % 16
    if int(packed.shape[-2]) != taps * channels * (2 if exact else 1):
        raise ValueError(
            f"packed weight rows {int(packed.shape[-2])} do not fit {in_channels} "
            f"{'float32' if exact else 'float16'} input channels"
        )
    width = int(packed.shape[-1])
    out_channels = int(bias.shape[-1])
    q, p = w + 2, (h + 2) * (w + 2)
    rows = (d + 2) * p
    # The conv kernel's row grid: output voxels, or source voxels when UP.
    grid = (d, h, w) if stride == 1 else tuple((n - 1) // 2 + 1 for n in (d, h, w))
    pack_kernel, conv_kernel, reduce_kernel = _kernels(mx)
    # Largest power-of-two channel chunk (at least 8) whose staged line of
    # W x (chunk + 2) halves fits in 16 KiB of threadgroup memory.
    chunk = 8
    while channels % (2 * chunk) == 0 and w * (2 * chunk + 2) * (4 if exact else 2) <= 16384:
        chunk *= 2
    flat = pack_kernel(
        inputs=[x, affine if affine is not None else mx.zeros((1,), dtype=mx.float32)],
        template=[
            ("T", x.dtype),
            ("C", channels),
            ("CIN", in_channels),
            ("D", d),
            ("H", h),
            ("W", w),
            ("Q", q),
            ("ROWS", rows),
            ("CB", chunk),
            ("TG", _PACK_THREADS),
            ("AFFINE", affine is not None),
            ("SILU", silu),
            ("EXACT", exact),
        ],
        grid=(_PACK_THREADS * (d + 2) * (h + 2), channels // chunk, batch),
        threadgroup=(_PACK_THREADS, 1, 1),
        output_shapes=[(batch, rows, channels * (2 if exact else 1))],
        output_dtypes=[mx.float16],
    )[0]
    # Four 32-column simdgroups per threadgroup when O' allows; this was the
    # fastest shape for LINet's 128-512 channel layers on M5 Max.
    columns = 4 if width % 128 == 0 else 1
    tile_width = 16 if width == 16 else 32
    tiles = -(-math.prod(grid) // (_SIMDGROUP_ROWS * 32))
    scale = 2 if upsample else 1
    output_shape = (batch, out_channels, *(scale * n for n in grid))
    if residual is not None and tuple(residual.shape) != output_shape:
        raise ValueError(f"residual shape {tuple(residual.shape)} != output {output_shape}")
    threadgroups = width // (tile_width * columns) * tiles * batch * scale**3
    split = (
        3
        if not upsample
        and threadgroups < _SPLIT_BELOW_THREADGROUPS
        and channels >= _SPLIT_MIN_CHANNELS
        and math.prod(grid) % 4 == 0
        else 1
    )
    output = conv_kernel(
        inputs=[flat, packed, bias, residual if residual is not None else flat],
        template=[
            ("C", channels),
            ("O", width),
            ("OUT", out_channels),
            ("NB", batch),
            ("D", grid[0]),
            ("H", grid[1]),
            ("W", grid[2]),
            ("P", p),
            ("Q", q),
            ("ROWS", rows),
            ("STRIDE", stride),
            ("K", 2 if upsample else 3),
            ("UP", upsample),
            ("SPLIT", split),
            ("BATCH_BIAS", bias.ndim == 2),
            ("RESIDUAL", residual is not None),
            ("WM", _SIMDGROUP_ROWS),
            ("WN", columns),
            ("TN", tile_width),
            ("EXACT", exact),
            ("OT", dtype),
        ],
        grid=(32 * _SIMDGROUP_ROWS * width // tile_width, tiles, batch * scale**3 * split),
        threadgroup=(32 * _SIMDGROUP_ROWS * columns, 1, 1),
        output_shapes=[(split, *output_shape) if split > 1 else output_shape],
        output_dtypes=[mx.float32 if split > 1 else dtype],
    )[0]
    if split == 1:
        return output
    size = math.prod(output_shape)
    return reduce_kernel(
        inputs=[output, bias, residual if residual is not None else output],
        template=[
            ("N", size),
            ("SPATIAL", math.prod(grid)),
            ("OUT", out_channels),
            ("SPLIT", split),
            ("BATCH_BIAS", bias.ndim == 2),
            ("RESIDUAL", residual is not None),
            ("OT", dtype),
        ],
        grid=(size // 4, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[dtype],
    )[0]
