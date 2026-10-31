"""Conditioned-mode fast paths: NAX and pointwise convolution, dense, group norm, attention.

Convolution bounds separate the error sources. Each FP16 rounding adds at most
half a unit of FP16 spacing; FP32 accumulation adds a small multiple of FP32
epsilon times the sum of absolute products, which an FP16 accumulator would
exceed by about four orders of magnitude. An FP16 rounding of every operand of
one factor (folded upsampling weights, a separately rounded normalized input)
adds at most ``2**-11`` times that sum. FP32 exact mode splits each FP32
operand into two FP16 parts and drops only their product of low parts, adding
a small multiple of ``2**-22`` times the same sum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from medmlx_core.group_norm import native_group_norm, native_group_norm_eligible
from medmlx_core.ops import conv3d_ncdhw

from lesiondiffusion_mlx._arrays import NumericArray
from lesiondiffusion_mlx._attention import flash_attention, flash_eligible
from lesiondiffusion_mlx._nax_conv import (
    DISABLE_ENV,
    nax_available,
    nax_conv3d,
    nax_eligible,
    pack_upsample_weight,
    pack_weight,
)
from lesiondiffusion_mlx._typing import ArrayConstructor
from lesiondiffusion_mlx.mlx_model import LdmUnet, SourceLayers, UnetConfig

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Mlx

_ACCUMULATION_EPS = 8 * 2.0**-23
_OPERAND_ROUNDING = 2.0**-11
_SPLIT_ERROR = 4 * 2.0**-22


def _host(mx: Mlx, array: Array) -> NumericArray:
    """Convert a native array to a float64 host comparison array."""
    return cast(NumericArray, np.array(array.astype(mx.float32))).astype(np.float64)


def _half_spacing(values: NumericArray) -> NumericArray:
    """Compute float16 spacing around host values."""
    return np.spacing(np.abs(values).astype(np.float16)).astype(np.float64) / 2


def _case(
    mx: Mlx, shape: tuple[int, ...], out_channels: int, kernel: int, seed: int
) -> tuple[Array, Array, Array]:
    """Create seeded native convolution inputs and weights."""
    rng = np.random.default_rng(seed)
    create = cast(ArrayConstructor, mx.array)
    channels = shape[1]
    x = create(rng.normal(size=shape).astype(np.float16))
    scale = (channels * kernel**3) ** -0.5
    weight = create(
        (rng.normal(size=(out_channels, channels, kernel, kernel, kernel)) * scale).astype(
            np.float16
        )
    )
    bias = create((rng.normal(size=(out_channels,)) * 0.1).astype(np.float16))
    return x, weight, bias


def _reference(
    mx: Mlx, x: Array, weight: Array, bias: Array | None, padding: int, stride: int = 1
) -> tuple[NumericArray, NumericArray, NumericArray]:
    """FP32 sum without bias, FP32 result and sum of absolute terms (TF32 is disabled)."""

    def convolve(values: Array, kernel: Array, offset: Array | None) -> NumericArray:
        """Run the reference convolution for a numerical case."""
        result = conv3d_ncdhw(
            values.astype(mx.float32),
            kernel.astype(mx.float32),
            None if offset is None else offset.astype(mx.float32),
            padding=padding,
            stride=stride,
            mx=mx,
        )
        return _host(mx, result)

    products = convolve(x, weight, None)
    result = products if bias is None else convolve(x, weight, bias)

    def absolute(value: Array) -> Array:
        """Return absolute host values as a native array."""
        return cast(ArrayConstructor, mx.array)(np.abs(_host(mx, value)).astype(np.float32))

    magnitude = convolve(absolute(x), absolute(weight), None if bias is None else absolute(bias))
    return products, result, magnitude


def _assert_single_rounding(
    mx: Mlx, actual: Array, x: Array, weight: Array, bias: Array, stride: int = 1
) -> None:
    """Check native half output against the single-rounding reference."""
    _, expected, magnitude = _reference(mx, x, weight, bias, padding=1, stride=stride)
    error = np.abs(_host(mx, actual) - expected)
    assert np.all(error <= _half_spacing(expected) + _ACCUMULATION_EPS * magnitude)


def _skip_without_nax(mx: Mlx) -> None:
    """Skip neural-accelerator cases when the required device is unavailable."""
    if not nax_available(mx):
        pytest.skip("NAX convolution needs an M5-class GPU and macOS 26.2")


@pytest.mark.parametrize(
    ("shape", "out_channels", "stride"),
    [
        ((1, 16, 5, 7, 9), 32, 1),  # single-column threadgroups, odd extents
        ((2, 32, 4, 6, 3), 128, 1),  # four-column threadgroups, batched
        ((1, 48, 9, 3, 5), 64, 1),
        ((1, 5, 6, 4, 7), 4, 1),  # zero-padded input and output channels
        ((2, 24, 7, 6, 9), 40, 2),  # stride 2, odd and even extents
        ((1, 512, 4, 4, 4), 128, 1),  # few threadgroups, large C: split taps
    ],
)
def test_nax_conv_accumulates_in_fp32_and_rounds_once(
    mx: Mlx, shape: tuple[int, ...], out_channels: int, stride: int
) -> None:
    """Verify FP32 convolution accumulation and one final half rounding."""
    _skip_without_nax(mx)
    x, weight, bias = _case(mx, shape, out_channels, 3, seed=sum(shape))
    assert nax_eligible(weight)
    # Feed a transposed view, as layers may, not a contiguous copy.
    view = mx.transpose(mx.contiguous(mx.transpose(x, (0, 2, 3, 4, 1))), (0, 4, 1, 2, 3))
    actual = nax_conv3d(view, pack_weight(weight, mx), bias, stride=stride, mx=mx)
    mx.eval(actual)
    assert actual.dtype == mx.float16
    extents = tuple((size - 1) // stride + 1 for size in shape[2:])
    assert tuple(actual.shape) == (shape[0], out_channels, *extents)
    _assert_single_rounding(mx, actual, x, weight, bias, stride=stride)


@pytest.mark.parametrize("split", [False, True])
def test_nax_conv_adds_batch_bias_and_residual_before_rounding(mx: Mlx, split: bool) -> None:
    """Add batch bias and residual before the final rounding step."""
    _skip_without_nax(mx)
    channels = 512 if split else 32
    x, weight, _ = _case(mx, (2, channels, 4, 4, 4), 64, 3, seed=channels)
    rng = np.random.default_rng(1)
    create = cast(ArrayConstructor, mx.array)
    bias = create(rng.normal(size=(2, 64)).astype(np.float32))
    residual = create(rng.normal(size=(2, 64, 4, 4, 4)).astype(np.float16))
    actual = nax_conv3d(x, pack_weight(weight, mx), bias, residual=residual, mx=mx)
    mx.eval(actual)
    products, _, magnitude = _reference(mx, x, weight, None, padding=1)
    expected = products + _host(mx, bias)[:, :, None, None, None] + _host(mx, residual)
    error = np.abs(_host(mx, actual) - expected)
    assert np.all(error <= _half_spacing(expected) + _ACCUMULATION_EPS * magnitude)


def test_nax_upsample_conv_matches_conv_of_nearest_upsampling(mx: Mlx) -> None:
    """Compare fused upsample convolution with explicit nearest upsampling."""
    _skip_without_nax(mx)
    x, weight, bias = _case(mx, (2, 32, 3, 5, 4), 64, 3, seed=21)
    actual = nax_conv3d(x, pack_upsample_weight(weight, mx), bias, upsample=True, mx=mx)
    mx.eval(actual)
    upsampled = x
    for axis in (2, 3, 4):
        upsampled = mx.repeat(upsampled, 2, axis=axis)
    _, expected, magnitude = _reference(mx, upsampled, weight, bias, padding=1)
    assert tuple(actual.shape) == expected.shape
    error = np.abs(_host(mx, actual) - expected)
    # Merged taps are rounded to FP16 once when folded.
    bound = _half_spacing(expected) + (_ACCUMULATION_EPS + _OPERAND_ROUNDING) * magnitude
    assert np.all(error <= bound)


def _conv_float64(
    x: NumericArray, weight: NumericArray, stride: int = 1
) -> tuple[NumericArray, NumericArray]:
    """Padding-1 3x3x3 convolution in float64, and the same of absolute values."""

    def convolve(values: NumericArray, kernel: NumericArray) -> NumericArray:
        """Compute a float64 reference convolution for one kernel case."""
        padded = np.pad(values, [(0, 0), (0, 0), (1, 1), (1, 1), (1, 1)])
        windows = cast(Any, np.lib.stride_tricks).sliding_window_view(
            padded, (3, 3, 3), axis=(2, 3, 4)
        )
        windows = windows[:, :, ::stride, ::stride, ::stride]
        return np.einsum("bcdhwijk,ocijk->bodhw", windows, kernel)

    return convolve(x, weight), convolve(np.abs(x), np.abs(weight))


def _fp32_case(
    shape: tuple[int, ...], out_channels: int, seed: int
) -> tuple[NumericArray, NumericArray, NumericArray]:
    """Create a seeded FP32 convolution case."""
    rng = np.random.default_rng(seed)
    # Channel scales over several decades put many low parts near FP16's
    # subnormal range, which the 2^11 scaling of the low part must survive.
    scales = np.exp(rng.normal(size=(1, shape[1], 1, 1, 1)) * 2)
    x = (rng.normal(size=shape) * scales).astype(np.float32).astype(np.float64)
    weight = rng.normal(size=(out_channels, shape[1], 3, 3, 3)) * (27 * shape[1]) ** -0.5
    bias = rng.normal(size=(out_channels,)) * 0.1
    return (
        x,
        weight.astype(np.float32).astype(np.float64),
        bias.astype(np.float32).astype(np.float64),
    )


def _assert_near_fp32(
    actual: NumericArray, expected: NumericArray, magnitude: NumericArray
) -> None:
    """Check FP32 kernel output against the magnitude-scaled error bound."""
    bound = (_ACCUMULATION_EPS + _SPLIT_ERROR) * magnitude + 2.0**-23 * np.abs(expected)
    assert np.all(np.abs(actual - expected) <= bound)


@pytest.mark.parametrize(
    ("shape", "out_channels", "stride"),
    [
        ((1, 4, 5, 7, 9), 16, 1),  # padded input channels, 16-column tiles
        ((1, 16, 6, 5, 4), 1, 1),  # a single output channel
        ((2, 24, 7, 6, 9), 40, 2),  # stride 2, padded output channels
        ((1, 64, 4, 6, 5), 128, 1),  # four-column threadgroups
    ],
)
def test_nax_conv_fp32_exact_mode_matches_float64(
    mx: Mlx, shape: tuple[int, ...], out_channels: int, stride: int
) -> None:
    """Compare split-FP16 FP32 convolution with a float64 reference."""
    _skip_without_nax(mx)
    x, weight, bias = _fp32_case(shape, out_channels, seed=sum(shape))
    create = cast(ArrayConstructor, mx.array)
    packed = pack_weight(create(weight.astype(np.float32)), mx)
    actual = nax_conv3d(
        create(x.astype(np.float32)), packed, create(bias.astype(np.float32)), stride=stride, mx=mx
    )
    mx.eval(actual)
    assert actual.dtype == mx.float32
    products, magnitude = _conv_float64(x, weight, stride)
    expected = products + bias[None, :, None, None, None]
    assert tuple(actual.shape) == expected.shape
    _assert_near_fp32(
        _host(mx, actual), expected, magnitude + np.abs(bias)[None, :, None, None, None]
    )


def test_nax_conv_fp32_exact_mode_upsamples_with_bias_and_residual(mx: Mlx) -> None:
    """Compare fused FP32 upsampling, bias, and residual with a reference."""
    _skip_without_nax(mx)
    x, weight, _ = _fp32_case((2, 32, 3, 5, 4), 16, seed=23)
    rng = np.random.default_rng(23)
    bias = rng.normal(size=(2, 16)).astype(np.float32).astype(np.float64)
    residual = rng.normal(size=(2, 16, 6, 10, 8)).astype(np.float32).astype(np.float64)
    create = cast(ArrayConstructor, mx.array)
    actual = nax_conv3d(
        create(x.astype(np.float32)),
        pack_upsample_weight(create(weight.astype(np.float32)), mx),
        create(bias.astype(np.float32)),
        upsample=True,
        residual=create(residual.astype(np.float32)),
        mx=mx,
    )
    mx.eval(actual)
    upsampled = x.repeat(2, axis=2).repeat(2, axis=3).repeat(2, axis=4)
    products, magnitude = _conv_float64(upsampled, weight)
    expected = products + bias[:, :, None, None, None] + residual
    # Folded taps are FP32 sums of up to eight weights before the split.
    magnitude = magnitude + np.abs(bias)[:, :, None, None, None] + np.abs(residual)
    _assert_near_fp32(_host(mx, actual), expected, 2 * magnitude)


def test_nax_conv_rejects_weights_packed_for_the_other_precision(mx: Mlx) -> None:
    """Reject packed weights for an incompatible precision mode."""
    _skip_without_nax(mx)
    x, weight, bias = _case(mx, (1, 16, 3, 3, 3), 16, 3, seed=2)
    with pytest.raises(ValueError, match="packed weight rows"):
        nax_conv3d(x.astype(mx.float32), pack_weight(weight, mx), bias, mx=mx)


def test_native_layers_dispatch_fp32_convolutions_under_source_arithmetic(mx: Mlx) -> None:
    """Verify native layer FP32 convolution dispatch under source arithmetic."""
    rng = np.random.default_rng(31)
    create = cast(ArrayConstructor, mx.array)
    x, weight, bias = _fp32_case((1, 32, 4, 5, 6), 16, seed=31)
    weights = {
        "norm.weight": create(rng.normal(size=(32,)).astype(np.float32)),
        "norm.bias": create(rng.normal(size=(32,)).astype(np.float32)),
        "conv.weight": create(weight.astype(np.float32)),
        "conv.bias": create(bias.astype(np.float32)),
    }
    # LMNet's conditioned configuration: FP32 weights, source arithmetic,
    # native kernels. Source arithmetic only governs FP16 operations.
    native = SourceLayers(weights, mx, source_arithmetic=True, native_kernels=True)
    assert set(native.nax_weights) == ({"conv"} if nax_available(mx) else set())
    fallback = SourceLayers(weights, mx, source_arithmetic=True)
    values = create(x.astype(np.float32))
    shift = create(rng.normal(size=(1, 16)).astype(np.float32))
    residual = create(rng.normal(size=(1, 16, 4, 5, 6)).astype(np.float32))
    conv = native.conv(values, "conv")
    fused = native.norm_silu_conv(
        values, "norm", "conv", groups=8, eps=1e-5, shift=shift, residual=residual
    )
    unfused = fallback.norm_silu_conv(
        values, "norm", "conv", groups=8, eps=1e-5, shift=shift, residual=residual
    )
    mx.eval(conv, fused, unfused)
    assert conv.dtype == fused.dtype == mx.float32
    products, magnitude = _conv_float64(x, weight)
    _assert_near_fp32(_host(mx, conv), products + bias[None, :, None, None, None], magnitude + 1)
    np.testing.assert_allclose(_host(mx, fused), _host(mx, unfused), rtol=0, atol=1e-5)


def test_native_layers_dispatch_eligible_fp16_convolutions(mx: Mlx) -> None:
    """Verify dispatch for eligible native FP16 convolutions."""
    x, weight, bias = _case(mx, (1, 32, 4, 5, 6), 64, 3, seed=3)
    weights = {"conv.weight": weight, "conv.bias": bias}
    native = SourceLayers(weights, mx, source_arithmetic=False)
    assert not SourceLayers(weights, mx, source_arithmetic=True).nax_weights
    assert bool(native.nax_weights) == nax_available(mx)
    for stride in (1, 2):

        def convolve(value: Array, stride: int = stride) -> Array:
            """Apply the named native convolution with the selected stride."""
            return native.conv(value, "conv", stride=stride)

        actual = mx.compile(convolve)(x)
        mx.eval(actual)
        if native.nax_weights:
            _assert_single_rounding(mx, actual, x, weight, bias, stride=stride)
        else:
            expected = conv3d_ncdhw(x, weight, bias, padding=1, stride=stride, mx=mx)
            mx.eval(expected)
            assert _host(mx, actual).tobytes() == _host(mx, expected).tobytes()


def test_native_norm_silu_conv_fuses_normalization_shift_and_residual(mx: Mlx) -> None:
    """Compare fused normalization, activation, shift, and residual with the reference."""
    rng = np.random.default_rng(13)
    create = cast(ArrayConstructor, mx.array)
    x, weight, bias = _case(mx, (2, 32, 4, 5, 6), 64, 3, seed=13)
    weights = {
        "norm.weight": create(rng.normal(size=(32,)).astype(np.float16)),
        "norm.bias": create(rng.normal(size=(32,)).astype(np.float16)),
        "conv.weight": weight,
        "conv.bias": bias,
    }
    shift = create(rng.normal(size=(2, 64)).astype(np.float16))
    residual = create(rng.normal(size=(2, 64, 4, 5, 6)).astype(np.float16))
    native = SourceLayers(weights, mx, source_arithmetic=False)
    actual = native.norm_silu_conv(
        x, "norm", "conv", groups=8, eps=1e-5, shift=shift, residual=residual
    )
    normalized = native.norm_silu(x, "norm", groups=8, eps=1e-5)
    mx.eval(actual, normalized)
    _, expected, magnitude = _reference(mx, normalized, weight, bias, padding=1)
    expected = expected + _host(mx, shift)[:, :, None, None, None] + _host(mx, residual)
    error = np.abs(_host(mx, actual) - expected)
    if native.nax_weights:
        # One rounding of the sum; the fused input may round differently by
        # an FP16 ulp from the separately normalized tensor.
        bound = _half_spacing(expected) + (_ACCUMULATION_EPS + _OPERAND_ROUNDING) * magnitude
    else:
        # Rounded after the convolution, the shift and the residual.
        bound = 3 * _half_spacing(expected) + _ACCUMULATION_EPS * magnitude
    assert np.all(error <= bound)


def test_native_fp16_dense_rounds_once(mx: Mlx) -> None:
    """Verify native FP16 dense layers round after accumulation."""
    rng = np.random.default_rng(17)
    create = cast(ArrayConstructor, mx.array)
    x = rng.normal(size=(2, 33, 48)).astype(np.float16)
    weight = (rng.normal(size=(40, 48)) / 7).astype(np.float16)
    bias = rng.normal(size=(40,)).astype(np.float16)
    weights = {"fc.weight": create(weight), "fc.bias": create(bias)}
    actual = SourceLayers(weights, mx, source_arithmetic=False).dense(create(x), "fc")
    mx.eval(actual)
    assert actual.dtype == mx.float16
    terms = x.astype(np.float64) @ weight.astype(np.float64).T
    expected = terms + bias.astype(np.float64)
    magnitude = np.abs(x.astype(np.float64)) @ np.abs(weight.astype(np.float64)).T + np.abs(
        bias.astype(np.float64)
    )
    error = np.abs(_host(mx, actual) - expected)
    assert np.all(error <= _half_spacing(expected) + _ACCUMULATION_EPS * magnitude)


@pytest.mark.parametrize("with_bias", [True, False])
def test_pointwise_fp16_convolution_rounds_product_then_bias(mx: Mlx, with_bias: bool) -> None:
    """Verify pointwise half convolution rounding around bias addition."""
    x, weight, bias = _case(mx, (2, 48, 3, 5, 7), 40, 1, seed=11)
    weights = {"proj.weight": weight} | ({"proj.bias": bias} if with_bias else {})
    actual = SourceLayers(weights, mx, source_arithmetic=False).conv(x, "proj", padding=0)
    mx.eval(actual)
    assert tuple(actual.shape) == (2, 40, 3, 5, 7)
    assert actual.dtype == mx.float16
    products, expected, magnitude = _reference(
        mx, x, weight, bias if with_bias else None, padding=0
    )
    # The FP16 product is rounded, then the bias sum is rounded again.
    bound = _half_spacing(products) + _ACCUMULATION_EPS * magnitude
    if with_bias:
        bound = bound + _half_spacing(expected)
    assert np.all(np.abs(_host(mx, actual) - expected) <= bound)


def _attention_reference(
    q: NumericArray, k: NumericArray, v: NumericArray, heads: int
) -> NumericArray:
    """Compute a float64 attention reference."""
    result = np.zeros_like(q)
    for b in range(q.shape[0]):
        for h in range(heads):
            columns = slice(32 * h, 32 * h + 32)
            scores = q[b, :, columns] @ k[b, :, columns].T / np.sqrt(32)
            weights = np.exp(scores - scores.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True)
            result[b, :, columns] = weights @ v[b, :, columns]
    return result


@pytest.mark.parametrize("kernel", ["nax", "simd"])
@pytest.mark.parametrize(
    ("batch", "length", "keys", "heads"),
    [(1, 96, 64, 2), (2, 32, 160, 1), (1, 1056, 96, 2)],  # 1056: partial NAX threadgroup
)
def test_flash_attention_matches_float64_softmax(
    mx: Mlx,
    monkeypatch: pytest.MonkeyPatch,
    kernel: str,
    batch: int,
    length: int,
    keys: int,
    heads: int,
) -> None:
    """Compare flash attention with float64 softmax reference values."""
    if kernel == "nax":
        _skip_without_nax(mx)
    else:
        monkeypatch.setenv(DISABLE_ENV, "0")
    rng = np.random.default_rng(length + keys)
    create = cast(ArrayConstructor, mx.array)
    # Scaled inputs give peaked softmax rows, exercising the running maximum.
    q, k, v = (
        rng.normal(size=(batch, size, heads * 32)) * scale
        for size, scale in ((length, 2.0), (keys, 2.0), (keys, 1.0))
    )
    arrays = [create(value.astype(np.float32)) for value in (q, k, v)]
    assert flash_eligible(arrays[0], arrays[1], heads, mx)
    actual = flash_attention(*arrays, heads=heads, scale=32**-0.5, mx=mx)
    mx.eval(actual)
    expected = _attention_reference(q, k, v, heads)
    np.testing.assert_allclose(_host(mx, actual), expected, rtol=0, atol=2e-5)


def test_fused_fp32_attention_dispatches_to_flash_kernel(mx: Mlx) -> None:
    """Verify fused FP32 attention dispatches to the flash kernel."""
    rng = np.random.default_rng(5)
    create = cast(ArrayConstructor, mx.array)
    weights = {
        f"attn.{name}.weight": create((rng.normal(size=(64, 64)) / 8).astype(np.float32))
        for name in ("to_q", "to_k", "to_v", "to_out.0")
    } | {"attn.to_out.0.bias": create(np.zeros(64, dtype=np.float32))}
    x = create(rng.normal(size=(1, 64, 64)).astype(np.float32))
    config = UnetConfig(64, 1, ())
    fused = LdmUnet(weights, mx, config, fused_attention=True).attention(x, None, "attn", 2)
    explicit = LdmUnet(weights, mx, config).attention(x, None, "attn", 2)
    mx.eval(fused, explicit)
    np.testing.assert_allclose(_host(mx, fused), _host(mx, explicit), rtol=0, atol=1e-5)


@pytest.mark.parametrize("precision", ["float16", "float32"])
@pytest.mark.parametrize("silu", [False, True])
@pytest.mark.parametrize(
    ("shape", "groups"),
    [
        ((2, 8, 3, 4, 6), 4),  # one partial chunk per group
        ((1, 4, 34, 32, 32), 2),  # 69632 values per group: four full chunks and a partial
    ],
)
def test_native_group_norm_matches_float64_statistics(
    mx: Mlx, precision: str, silu: bool, shape: tuple[int, ...], groups: int
) -> None:
    """Compare native group normalization with float64 statistics."""
    rng = np.random.default_rng(len(shape) + groups)
    dtype = np.float16 if precision == "float16" else np.float32
    create = cast(ArrayConstructor, mx.array)
    # An offset mean exercises the shifted moments.
    host = (rng.normal(size=shape) * 3 + 20).astype(dtype)
    weight, bias = (rng.normal(size=(shape[1],)).astype(dtype) for _ in range(2))
    x = create(host)
    assert native_group_norm_eligible(x, groups)
    actual = native_group_norm(
        x, create(weight), create(bias), groups=groups, eps=1e-5, silu=silu, mx=mx
    )
    mx.eval(actual)
    assert actual.dtype == x.dtype
    grouped = host.astype(np.float64).reshape(shape[0], groups, -1)
    normalized = (grouped - grouped.mean(axis=2, keepdims=True)) / np.sqrt(
        grouped.var(axis=2, keepdims=True) + 1e-5
    )
    channel = (-1, 1, 1, 1)
    expected = normalized.reshape(shape) * weight.astype(np.float64).reshape(channel) + bias.astype(
        np.float64
    ).reshape(channel)
    if silu:
        expected = expected / (1 + np.exp(-expected))
    error = np.abs(_host(mx, actual) - expected)
    if precision == "float16":
        # FP32 statistics, then one FP16 rounding of the result.
        assert np.all(error <= _half_spacing(expected) + 1e-5 * np.abs(expected) + 1e-6)
    else:
        assert float(np.max(error)) <= 1e-5


def test_native_layers_fuse_silu_only_with_native_kernels(mx: Mlx) -> None:
    """Fuse SiLU only when the native-kernel path is active."""
    rng = np.random.default_rng(9)
    create = cast(ArrayConstructor, mx.array)
    weights = {
        "n.weight": create(rng.normal(size=(8,)).astype(np.float16)),
        "n.bias": create(rng.normal(size=(8,)).astype(np.float16)),
    }
    x = create(rng.normal(size=(1, 8, 2, 4, 4)).astype(np.float16))
    source = SourceLayers(weights, mx, source_arithmetic=True)
    native = SourceLayers(weights, mx, source_arithmetic=True, native_kernels=True)
    unfused = source.silu(source.norm(x, "n", groups=4, eps=1e-6))
    separate = source.norm_silu(x, "n", groups=4, eps=1e-6)
    fused = native.norm_silu(x, "n", groups=4, eps=1e-6)
    mx.eval(unfused, separate, fused)
    # Without native kernels the separate rounding steps are unchanged.
    assert _host(mx, separate).tobytes() == _host(mx, unfused).tobytes()
    np.testing.assert_allclose(_host(mx, fused), _host(mx, unfused), rtol=0, atol=4e-3)
