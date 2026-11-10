"""Exact code selection and straight-through arithmetic for native VQ search."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from lesiondiffusion_mlx._typing import ArrayConstructor
from lesiondiffusion_mlx._vq import nearest_codebook4
from lesiondiffusion_mlx.mlx_model import VqAutoencoder

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Mlx


def _nearest_reference(flat: Array, codebook: Array, mx: Mlx) -> tuple[Array, Array]:
    """Use the source distance expression, retaining each dtype rounding boundary."""
    norms = mx.sum((codebook * codebook).astype(mx.float32), axis=1).astype(codebook.dtype)
    latent_norms = mx.sum((flat * flat).astype(mx.float32), axis=1, keepdims=True).astype(
        flat.dtype
    )
    indices = mx.argmin(latent_norms + norms[None] - 2 * (flat @ codebook.T), axis=1)
    return indices, norms


@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
@pytest.mark.parametrize("native_kernels", [True, False], ids=["native", "chunked"])
def test_vq_preserves_quantized_values_on_strided_batches(
    dtype_name: str, native_kernels: bool, mx: Mlx
) -> None:
    """Fused search must keep code selection and source straight-through rounding."""
    rng = np.random.default_rng(2701)
    dtype = getattr(mx, dtype_name)
    array = cast(ArrayConstructor, mx.array)
    codebook = array(rng.normal(size=(2048, 4)).astype(np.float32), dtype=dtype)
    z = array(rng.normal(size=(2, 4, 9, 8, 14)).astype(np.float32), dtype=dtype)[..., ::2]
    flat = mx.reshape(mx.transpose(z, (0, 2, 3, 4, 1)), (-1, 4))
    indices, _ = _nearest_reference(flat, codebook, mx)
    expected = flat + (codebook[indices] - flat)
    expected = mx.transpose(mx.reshape(expected, (2, 9, 8, 7, 4)), (0, 4, 1, 2, 3))
    model = VqAutoencoder(
        {"quantize.embedding.weight": codebook}, mx,
        source_arithmetic=False, native_kernels=native_kernels,
    )
    actual = model.quantize(z, chunk_size=37)
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(
        np.array(actual, dtype=np.float32), np.array(expected, dtype=np.float32)
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
@pytest.mark.parametrize(
    "positions,entries", [(1, 2048), (2, 33), (3, 1), (31, 2), (32, 32), (33, 31)]
)
def test_nearest_codebook_preserves_first_index_at_rounding_ties(
    dtype_name: str, positions: int, entries: int, mx: Mlx
) -> None:
    """Duplicate entries, midpoint neighbors and NaNs retain MLX argmin's index order."""
    rng = np.random.default_rng(317)
    host_codebook = rng.normal(size=(entries, 4)).astype(np.float32)
    if entries > 3:
        host_codebook[-1] = host_codebook[0]
        host_codebook[3] = -host_codebook[2]
    host_points = np.concatenate(
        [(host_codebook[:-1] + host_codebook[1:]) / 2, host_codebook, np.zeros((1, 4))]
    ).astype(np.float32)
    host_points = np.tile(host_points, (positions // len(host_points) + 1, 1))[:positions]
    dtype = getattr(mx, dtype_name)
    array = cast(ArrayConstructor, mx.array)
    codebook = array(host_codebook, dtype=dtype)
    flat = array(host_points, dtype=dtype)
    expected, norms = _nearest_reference(flat, codebook, mx)
    actual = nearest_codebook4(flat, codebook, norms, mx)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(
        np.array(actual, dtype=np.int32), np.array(expected, dtype=np.int32)
    )

    # FP16 squared norms can overflow on finite values. Both paths must give
    # the resulting NaN distances the same first-index priority.
    special = array(np.array([[np.nan, 0, 0, 0], [300, 1, -2, 3]], np.float32), dtype=dtype)
    expected, norms = _nearest_reference(special, codebook, mx)
    actual = nearest_codebook4(special, codebook, norms, mx)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(
        np.array(actual, dtype=np.int32), np.array(expected, dtype=np.int32)
    )


def test_replacing_codebook_updates_cached_norms(mx: Mlx) -> None:
    """A reused model must select from the current codebook after weight replacement."""
    array = cast(ArrayConstructor, mx.array)
    weights = {
        "quantize.embedding.weight": array(np.array([[0, 0, 0, 0], [2, 0, 0, 0]], np.float32))
    }
    model = VqAutoencoder(weights, mx, source_arithmetic=False)
    z = array(np.array([1.75, 0, 0, 0], np.float32).reshape(1, 4, 1, 1, 1))
    first = model.quantize(z)
    mx.eval(first)
    weights["quantize.embedding.weight"] = array(
        np.array([[1, 0, 0, 0], [20, 0, 0, 0]], np.float32)
    )
    actual = model.quantize(z)
    mx.eval(actual)
    np.testing.assert_array_equal(
        np.array(actual, dtype=np.float32).reshape(4), np.array([1, 0, 0, 0], np.float32)
    )
