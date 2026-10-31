"""Metal FP16 layers with the pinned Torch CPU oracle's reduction ordering.

The convolution and Welford algorithms follow PyTorch 2.13 CPU BlasKernel.cpp
and moments_utils.h. See root THIRD_PARTY_NOTICES.md. Inputs and outputs remain on Metal.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, MetalKernel, Mlx

from medmlx_core.layout import require_ncdhw

from lesiondiffusion_mlx._typing import ArrayConstructor

_CONV = r"""
uint tid = thread_position_in_threadgroup.x;
ulong row_base = ulong(threadgroup_position_in_grid.y) * 64;
uint channel_base = threadgroup_position_in_grid.z * 32;
const uint size = C * KD * KH * KW;
const uint complete = size - size % 4;
const ulong positions = ulong(OD) * OH * OW;
threadgroup float input_tile[64 * 32];
threadgroup float4 weight_tile[32 * 8];
uint row_slot = tid / 8;
uint channel_slot = tid % 8;
uint oc = channel_base + channel_slot * 4;
float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
float4 b0 = 0, b1 = 0, b2 = 0, b3 = 0;
for (uint base = 0; base < complete; base += 32) {
    for (uint i = tid; i < 64 * 32; i += 256) {
        ulong row = row_base + i / 32;
        uint k = base + i % 32;
        float value = 0;
        if (row < ulong(B) * positions && k < complete) {
            uint c = k / (KD * KH * KW);
            uint rem = k % (KD * KH * KW);
            int z = int((row / (OH * OW)) % OD) * S + int(rem / (KH * KW)) - P;
            int y = int((row / OW) % OH) * S + int((rem / KW) % KH) - P;
            int x = int(row % OW) * S + int(rem % KW) - P;
            if (z >= 0 && z < D && y >= 0 && y < H && x >= 0 && x < W) {
                ulong n = row / positions;
                value = float(values[(((n * C + c) * D + z) * H + y) * W + x]);
            }
        }
        input_tile[i] = value;
    }
    uint k = base + tid / 8;
    uint first = channel_base + (tid % 8) * 4;
    float4 w = 0;
    if (k < complete) {
        for (uint j = 0; j < 4; ++j)
            if (first + j < O) w[j] = float(weights[ulong(first + j) * size + k]);
    }
    weight_tile[tid] = w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Each partial visits precisely k % 4 in channel-major source order.
    for (uint k = 0; k < 32; k += 4) {
        float4 w0 = weight_tile[k * 8 + channel_slot];
        float4 w1 = weight_tile[(k + 1) * 8 + channel_slot];
        float4 w2 = weight_tile[(k + 2) * 8 + channel_slot];
        float4 w3 = weight_tile[(k + 3) * 8 + channel_slot];
        a0 += input_tile[row_slot * 32 + k] * w0;
        a1 += input_tile[row_slot * 32 + k + 1] * w1;
        a2 += input_tile[row_slot * 32 + k + 2] * w2;
        a3 += input_tile[row_slot * 32 + k + 3] * w3;
        b0 += input_tile[(row_slot + 32) * 32 + k] * w0;
        b1 += input_tile[(row_slot + 32) * 32 + k + 1] * w1;
        b2 += input_tile[(row_slot + 32) * 32 + k + 2] * w2;
        b3 += input_tile[(row_slot + 32) * 32 + k + 3] * w3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// The source appends every remaining tap to its first partial sum.
for (uint k = complete; k < size; ++k) {
    float4 w = 0;
    for (uint j = 0; j < 4; ++j)
        if (oc + j < O) w[j] = float(weights[ulong(oc + j) * size + k]);
    for (uint r = 0; r < 2; ++r) {
        ulong row = row_base + row_slot + r * 32;
        float value = 0;
        if (row < ulong(B) * positions) {
            uint c = k / (KD * KH * KW);
            uint rem = k % (KD * KH * KW);
            int z = int((row / (OH * OW)) % OD) * S + int(rem / (KH * KW)) - P;
            int y = int((row / OW) % OH) * S + int((rem / KW) % KH) - P;
            int x = int(row % OW) * S + int(rem % KW) - P;
            if (z >= 0 && z < D && y >= 0 && y < H && x >= 0 && x < W) {
                ulong n = row / positions;
                value = float(values[(((n * C + c) * D + z) * H + y) * W + x]);
            }
        }
        if (r == 0) a0 += value * w;
        else b0 += value * w;
    }
}
float4 total_a = ((a0 + a1) + a2) + a3;
float4 total_b = ((b0 + b1) + b2) + b3;
if constexpr (HAS_BIAS) {
    float4 offset = 0;
    for (uint j = 0; j < 4; ++j)
        if (oc + j < O) offset[j] = float(bias[oc + j]);
    total_a += offset;
    total_b += offset;
}
for (uint r = 0; r < 2; ++r) {
    ulong row = row_base + row_slot + r * 32;
    if (row >= ulong(B) * positions) continue;
    ulong n = row / positions;
    ulong position = row % positions;
    float4 total = r == 0 ? total_a : total_b;
    for (uint j = 0; j < 4; ++j)
        if (oc + j < O) result[(n * O + oc + j) * positions + position] = total[j];
}
"""

_MOMENTS_HEADER = r"""
#pragma clang fp contract(off)
struct Moments { uint count; float4 mean; float4 m2; };
inline void merge(thread Moments& a, Moments b) {
    uint count = a.count + b.count;
    float ratio = count == 0 ? 0 : float(b.count) / float(count);
    float4 delta = b.mean - a.mean;
    float4 correction = ratio * delta;
    float4 prior = delta * float(a.count);
    a.mean = a.mean + correction;
    a.m2 = metal::fma(prior, correction, a.m2 + b.m2);
    a.count = count;
}
"""

_PARTIAL_MOMENTS = r"""
uint index = thread_position_in_grid.x;
if (index >= B * G * CHUNKS) return;
uint group = index / CHUNKS;
uint chunk = index % CHUNKS;
uint vectors = SIZE / 8;
uint count = min(uint(16), vectors - chunk * 16);
float4 mean0 = 0, mean1 = 0, m20 = 0, m21 = 0;
for (uint j = 0; j < count; ++j) {
    ulong base = ulong(group) * SIZE + (chunk * 16 + j) * 8;
    float4 x0 = float4(values[base], values[base+1], values[base+2], values[base+3]);
    float4 x1 = float4(values[base+4], values[base+5], values[base+6], values[base+7]);
    float ratio = 1.0f / float(j + 1);
    float4 delta0 = x0 - mean0;
    float4 delta1 = x1 - mean1;
    mean0 = metal::fma(delta0, ratio, mean0);
    mean1 = metal::fma(delta1, ratio, mean1);
    m20 = metal::fma(delta0, x0 - mean0, m20);
    m21 = metal::fma(delta1, x1 - mean1, m21);
}
for (uint lane = 0; lane < 4; ++lane) {
    partials[index * 16 + lane] = mean0[lane];
    partials[index * 16 + 4 + lane] = m20[lane];
    partials[index * 16 + 8 + lane] = mean1[lane];
    partials[index * 16 + 12 + lane] = m21[lane];
}
"""

_CASCADE = r"""
uint index = thread_position_in_grid.x;
if (index >= B * G * BLOCKS) return;
uint group = index / BLOCKS;
uint first = (index % BLOCKS) * BLOCK_CHUNKS;
uint vectors = SIZE / 8;
Moments stack[BLOCK_DEPTH];
for (uint level = 0; level < BLOCK_DEPTH; ++level)
    stack[level] = {0, float4(0), float4(0)};
for (uint chunk = 0; chunk < BLOCK_CHUNKS; ++chunk) {
    uint count = min(uint(16), vectors - (first + chunk) * 16);
    ulong base = (ulong(group) * CHUNKS + first + chunk) * 16;
    Moments lo = {count, float4(0), float4(0)};
    Moments hi = {count, float4(0), float4(0)};
    for (uint lane = 0; lane < 4; ++lane) {
        lo.mean[lane] = partials[base + lane];
        lo.m2[lane] = partials[base + 4 + lane];
        hi.mean[lane] = partials[base + 8 + lane];
        hi.m2[lane] = partials[base + 12 + lane];
    }
    merge(stack[0], lo);
    merge(stack[0], hi);
    uint mask = chunk + 1;
    for (uint level = 1; level < BLOCK_DEPTH && (mask & 1) == 0; ++level) {
        merge(stack[level], stack[level - 1]);
        stack[level - 1] = {0, float4(0), float4(0)};
        mask >>= 1;
    }
}
// Stop before the final carry. Its empty/populated destination belongs to the
// complete group cascade and is merged by the next kernel in source order.
Moments result = stack[BLOCK_DEPTH - 1];
for (uint lane = 0; lane < 4; ++lane) {
    cascades[index * 8 + lane] = result.mean[lane];
    cascades[index * 8 + 4 + lane] = result.m2[lane];
}
"""

_MOMENTS = r"""
uint group = thread_position_in_grid.x;
if (group >= B * G) return;
uint vectors = SIZE / 8;
Moments stack[DEPTH];
for (uint level = 0; level < DEPTH; ++level) stack[level] = {0, float4(0), float4(0)};
if constexpr (BLOCKS > 0) {
    for (uint block = 0; block < BLOCKS; ++block) {
        uint count = min(uint(BLOCK_CHUNKS * 16), vectors - block * BLOCK_CHUNKS * 16) * 2;
        ulong base = (ulong(group) * BLOCKS + block) * 8;
        Moments part = {count, float4(0), float4(0)};
        for (uint lane = 0; lane < 4; ++lane) {
            part.mean[lane] = cascades[base + lane];
            part.m2[lane] = cascades[base + 4 + lane];
        }
        merge(stack[BLOCK_DEPTH], part);
        uint mask = block + 1;
        for (uint level = BLOCK_DEPTH + 1; level < DEPTH && (mask & 1) == 0; ++level) {
            merge(stack[level], stack[level - 1]);
            stack[level - 1] = {0, float4(0), float4(0)};
            mask >>= 1;
        }
    }
}
for (uint chunk = BLOCKS * BLOCK_CHUNKS; chunk < CHUNKS; ++chunk) {
    uint count = min(uint(16), vectors - chunk * 16);
    ulong base = (ulong(group) * CHUNKS + chunk) * 16;
    Moments lo = {count, float4(0), float4(0)};
    Moments hi = {count, float4(0), float4(0)};
    for (uint lane = 0; lane < 4; ++lane) {
        lo.mean[lane] = partials[base + lane];
        lo.m2[lane] = partials[base + 4 + lane];
        hi.mean[lane] = partials[base + 8 + lane];
        hi.m2[lane] = partials[base + 12 + lane];
    }
    merge(stack[0], lo);
    merge(stack[0], hi);
    uint mask = chunk + 1;
    for (uint level = 1; level < DEPTH && (mask & 1) == 0; ++level) {
        merge(stack[level], stack[level - 1]);
        stack[level - 1] = {0, float4(0), float4(0)};
        mask >>= 1;
    }
}
for (uint level = 1; level < DEPTH; ++level) merge(stack[0], stack[level]);
uint count = 0;
float mean = 0, m2 = 0;
for (uint i = vectors * 8; i < SIZE; ++i) {
    float x = float(values[ulong(group) * SIZE + i]);
    float delta = x - mean;
    ++count;
    mean += delta / float(count);
    m2 = metal::fma(delta, x - mean, m2);
}
uint added = vectors * 2;
for (uint lane = 0; lane < 4; ++lane) {
    uint n = count + added;
    float ratio = n == 0 ? 0 : float(added) / float(n);
    float delta = stack[0].mean[lane] - mean;
    mean = metal::fma(ratio, delta, mean);
    m2 += metal::fma(delta * delta * ratio, float(count), stack[0].m2[lane]);
    count = n;
}
"""

_NORM = r"""
uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
uint slice = thread_position_in_grid.z;
if (x >= W || y >= H || ulong(slice) >= ulong(B) * C * D) return;
ulong index = (ulong(slice) * H + y) * W + x;
uint group = uint(index / ulong(SIZE));
uint channel = (slice / D) % C;
float mean = stats[group * 2];
float var = max(stats[group * 2 + 1], 0.0f);
// Compensate the reciprocal root before rounding: Torch adds epsilon and
// takes the root in double, then stores the FP32 reciprocal.
float rstd = metal::precise::rsqrt(var + epsilon[0]);
float square = rstd * rstd;
float square_error = metal::fma(rstd, rstd, -square);
float residual = metal::fma(-var, square, 1.0f) - var * square_error - epsilon[0] * square;
rstd = metal::fma(0.5f * rstd, residual, rstd);
float scale = rstd * float(weight[channel]);
float offset = metal::fma(-scale, mean, float(bias[channel]));
output[index] = scale * float(values[index]) + offset;
"""


_NORM_PARAMETERS = r"""
// Append to the source-ordered moments kernel. Its mean and variance are FP32,
// exactly as when written to/read from the standalone statistics buffer.
float var = max(m2 / float(SIZE), 0.0f);
// Retain _NORM's compensated reciprocal root, once per group.
float rstd = metal::precise::rsqrt(var + epsilon[0]);
float square = rstd * rstd;
float square_error = metal::fma(rstd, rstd, -square);
float residual = metal::fma(-var, square, 1.0f) - var * square_error - epsilon[0] * square;
rstd = metal::fma(0.5f * rstd, residual, rstd);
for (uint c = 0; c < C / G; ++c) {
    uint channel = group * (C / G) + c;
    float scale = rstd * float(weight[channel % C]);
    float offset = metal::fma(-scale, mean, float(bias[channel % C]));
    affine[channel * 2] = scale;
    affine[channel * 2 + 1] = offset;
}
"""

_NORM_VECTOR = r"""
uint vector = thread_position_in_grid.x;
uint channel = thread_position_in_grid.y;
if (ulong(vector) * 4 >= SPATIAL || channel >= B * C) return;
ulong base = ulong(channel) * SPATIAL + ulong(vector) * 4;
float scale = affine[channel * 2];
float offset = affine[channel * 2 + 1];
// SPATIAL is divisible by four, so each aligned load stays in its channel.
half4 value = *reinterpret_cast<const device half4*>(values + base);
float4 result = scale * float4(value) + offset;
*reinterpret_cast<device half4*>(output + base) = half4(result);
"""


@lru_cache(maxsize=1)
def _norm_affine_kernels(mx: Mlx) -> tuple[MetalKernel, MetalKernel]:
    """Fuse group moments with channel affines; apply without contracting multiply/add."""
    return (
        mx.fast.metal_kernel(
            name="lesion_source_half_moments_affine",
            input_names=["values", "partials", "cascades", "weight", "bias", "epsilon"],
            output_names=["affine"],
            source=_MOMENTS + _NORM_PARAMETERS,
            header=_MOMENTS_HEADER,
        ),
        mx.fast.metal_kernel(
            name="lesion_source_half_norm_vector",
            input_names=["values", "affine"],
            output_names=["output"],
            source=_NORM_VECTOR,
            header="#pragma clang fp contract(off)\n",
        ),
    )


@lru_cache(maxsize=1)
def _kernels(mx: Mlx) -> tuple[MetalKernel, MetalKernel, MetalKernel, MetalKernel, MetalKernel]:
    """Build or reuse the source-order half-precision Metal kernels."""
    def kernel(
        name: str, inputs: list[str], output: str, source: str, header: str = ""
    ) -> MetalKernel:
        """Construct one named source-order Metal kernel."""
        return mx.fast.metal_kernel(
            name="lesion_source_half_" + name,
            input_names=inputs,
            output_names=[output],
            header=header,
            source=source,
            ensure_row_contiguous=True,
        )

    return (
        kernel("conv3d_tiled", ["values", "weights", "bias"], "result", _CONV),
        kernel("partial_moments", ["values"], "partials", _PARTIAL_MOMENTS, _MOMENTS_HEADER),
        kernel("cascade", ["partials"], "cascades", _CASCADE, _MOMENTS_HEADER),
        kernel(
            "moments",
            ["values", "partials", "cascades"],
            "stats",
            _MOMENTS + "stats[group * 2] = mean;\nstats[group * 2 + 1] = m2 / float(SIZE);\n",
            _MOMENTS_HEADER,
        ),
        kernel(
            "norm",
            ["values", "stats", "weight", "bias", "epsilon"],
            "output",
            _NORM,
            "#pragma clang fp contract(off)\n",
        ),
    )


def conv3d_half(
    x: Array, weight: Array, bias: Array | None, *, padding: int, stride: int, mx: Mlx
) -> Array:
    """Accumulate four channel-major partial sums, add bias, then round once.

    Args:
        x: Input ``(B, C, D, H, W)``; float16 in the runner.
        weight: Kernel ``(O, C, KD, KH, KW)``.
        bias: Per-output-channel bias ``(O,)``, or ``None``.
        padding: Symmetric zero padding applied to all three spatial axes.
        stride: Stride applied to all three spatial axes.
        mx: The imported ``mlx.core`` module.

    Returns:
        Output ``(B, O, OD, OH, OW)`` in ``x``'s dtype.

    Raises:
        ValueError: If the channel counts or bias shape disagree, or the kernel is
            larger than the padded input.
    """
    require_ncdhw(x, name="conv3d input")
    batch, channels, depth, height, width = x.shape
    outputs, inputs, kd, kh, kw = weight.shape
    if inputs != channels:
        raise ValueError("conv3d input channels differ from source weights")
    if bias is not None and bias.shape != (outputs,):
        raise ValueError("conv3d bias must have one value per output channel")
    od, oh, ow = (
        (d + 2 * padding - k) // stride + 1
        for d, k in zip((depth, height, width), (kd, kh, kw), strict=True)
    )
    if min(od, oh, ow) < 1:
        raise ValueError("conv3d kernel exceeds the padded input")
    shape = (batch, outputs, od, oh, ow)
    dimensions = dict(
        B=batch,
        C=channels,
        D=depth,
        H=height,
        W=width,
        O=outputs,
        KD=kd,
        KH=kh,
        KW=kw,
        OD=od,
        OH=oh,
        OW=ow,
        S=stride,
        P=padding,
        HAS_BIAS=bias is not None,
    )
    return _kernels(mx)[0](
        inputs=[x, weight, bias if bias is not None else mx.zeros((1,), dtype=x.dtype)],
        template=list(dimensions.items()),
        grid=(256, (batch * od * oh * ow + 63) // 64, (outputs + 31) // 32),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[x.dtype],
    )[0]


def group_norm_half(
    x: Array, weight: Array, bias: Array, *, groups: int, eps: float, mx: Mlx
) -> Array:
    """Cascade source-ordered Welford moments and apply the FP32 affine.

    Args:
        x: Input ``(B, C, D, H, W)``; float16 in the runner.
        weight: Per-channel scale ``(C,)``.
        bias: Per-channel shift ``(C,)``.
        groups: Number of normalization groups; must divide ``C``.
        eps: Variance epsilon.
        mx: The imported ``mlx.core`` module.

    Returns:
        Normalized array with ``x``'s shape and dtype.

    Raises:
        ValueError: If ``C`` is not divisible by ``groups`` or the affine shapes
            differ from ``(C,)``.
    """
    require_ncdhw(x, name="group_norm input")
    batch, channels, d, h, w = x.shape
    if channels % groups:
        raise ValueError(f"group_norm channels {channels} must be divisible by num_groups {groups}")
    if weight.shape != (channels,) or bias.shape != (channels,):
        raise ValueError("group_norm affine must have one value per channel")
    # Each Metal kernel otherwise makes its own contiguous copy of a strided
    # activation. Share one graph-visible copy across moments and affine passes.
    x = mx.contiguous(x)
    positions = d * h * w
    size = channels // groups * positions
    chunks = max(1, (size // 8 + 15) // 16)
    depth = max(1, (chunks - 1).bit_length())
    # Power-of-two blocks replay complete low-level source cascades in parallel.
    # The final carry and residual fold retain the original group-wide order.
    block_depth = 7
    block_chunks = 1 << block_depth
    blocks = chunks // block_chunks if chunks > block_chunks else 0
    _, partial_kernel, cascade_kernel, moments_kernel, norm_kernel = _kernels(mx)
    dimensions = [("B", batch), ("G", groups), ("SIZE", size), ("CHUNKS", chunks)]
    partials = partial_kernel(
        inputs=[x],
        template=dimensions,
        grid=(batch * groups * chunks, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(batch * groups * chunks, 16)],
        output_dtypes=[mx.float32],
    )[0]
    cascade_dimensions = [
        *dimensions,
        ("BLOCKS", blocks),
        ("BLOCK_CHUNKS", block_chunks),
        ("BLOCK_DEPTH", block_depth),
    ]
    cascades = (
        cascade_kernel(
            inputs=[partials],
            template=cascade_dimensions,
            grid=(batch * groups * blocks, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(batch * groups * blocks, 8)],
            output_dtypes=[mx.float32],
        )[0]
        if blocks
        else mx.zeros((1,), dtype=mx.float32)
    )
    epsilon = cast(ArrayConstructor, mx.array)([eps], dtype=mx.float32)
    if x.dtype == mx.float16 and positions >= 4096 and positions % 4 == 0:
        # Large aligned volumes use a vector copy with channel affines prepared
        # by the moments kernel. Small/tail shapes retain the scalar path.
        parameters_kernel, vector_kernel = _norm_affine_kernels(mx)
        affine = parameters_kernel(
            inputs=[x, partials, cascades, weight, bias, epsilon],
            template=[*cascade_dimensions, ("DEPTH", depth), ("C", channels)],
            grid=(batch * groups, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(batch * channels, 2)],
            output_dtypes=[mx.float32],
        )[0]
        return vector_kernel(
            inputs=[x, affine],
            template=[("B", batch), ("C", channels), ("SPATIAL", positions)],
            grid=(positions // 4, batch * channels, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]
    stats = moments_kernel(
        inputs=[x, partials, cascades],
        template=[*cascade_dimensions, ("DEPTH", depth)],
        grid=(batch * groups, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch * groups, 2)],
        output_dtypes=[mx.float32],
    )[0]
    return norm_kernel(
        inputs=[x, stats, weight, bias, epsilon],
        template=[("B", batch), ("C", channels), ("SIZE", size), ("D", d), ("H", h), ("W", w)],
        grid=(w, h, batch * channels * d),
        threadgroup=(16, 4, 4),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]
