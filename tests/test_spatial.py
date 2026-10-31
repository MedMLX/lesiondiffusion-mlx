"""Voxel-copy contracts for the model's fused factor-two spatial resize."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from lesiondiffusion_mlx._arrays import NumericArray
from lesiondiffusion_mlx._typing import ArrayConstructor
from lesiondiffusion_mlx.mlx_model import SourceLayers

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Mlx


@pytest.mark.parametrize("precision", ["float16", "float32"])
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "reversed", "sliced"])
@pytest.mark.parametrize("compiled", [False, True])
def test_nearest_upsample_copies_strided_voxels_exactly(
    mx: Mlx, precision: str, layout: str, compiled: bool
) -> None:
    """Copy nearest-neighbor upsampled voxels exactly for strided inputs."""
    dtype = np.float16 if precision == "float16" else np.float32
    host = np.random.default_rng(57).normal(size=(2, 3, 3, 5, 7)).astype(dtype)
    # Copies must preserve signed zero and non-finite values too.
    host[0, 0, 0, 0, :4] = [-0.0, np.inf, -np.inf, np.nan]
    create = cast(ArrayConstructor, mx.array)
    x = create(host)
    if layout == "channels_last":
        x = mx.transpose(
            create(np.ascontiguousarray(host.transpose(0, 2, 3, 4, 1))), (0, 4, 1, 2, 3)
        )
    elif layout == "reversed":
        x, host = x[..., ::-1], host[..., ::-1]
    elif layout == "sliced":
        x, host = x[..., ::2], host[..., ::2]
    model = SourceLayers({}, mx)
    resize = mx.compile(model.resize_nearest) if compiled else model.resize_nearest
    actual = resize(x, tuple(2 * size for size in host.shape[2:]))
    mx.eval(actual)
    expected = host.repeat(2, axis=2).repeat(2, axis=3).repeat(2, axis=4)
    assert np.array(actual).tobytes() == expected.tobytes()


@pytest.mark.parametrize("shape", [(1, 1, 1, 1, 1), (1, 3, 1, 7, 1), (2, 1, 3, 1, 5)])
def test_nearest_upsample_handles_unit_axes(mx: Mlx, shape: tuple[int, ...]) -> None:
    """Handle unit-length spatial axes in nearest upsampling."""
    host = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    actual = SourceLayers({}, mx).resize_nearest(
        cast(ArrayConstructor, mx.array)(host), tuple(2 * size for size in shape[2:])
    )
    mx.eval(actual)
    np.testing.assert_array_equal(
        cast(NumericArray, np.array(actual)),
        host.repeat(2, axis=2).repeat(2, axis=3).repeat(2, axis=4),
    )


def test_other_resize_ratios_keep_the_floor_index_rule(mx: Mlx) -> None:
    """Use the source floor-index rule for other resize ratios."""
    host = np.arange(2 * 3 * 5, dtype=np.float32).reshape(1, 1, 2, 3, 5)
    actual = SourceLayers({}, mx).resize_nearest(cast(ArrayConstructor, mx.array)(host), (3, 2, 7))
    mx.eval(actual)
    expected = host[:, :, [0, 0, 1]][:, :, :, [0, 1]][:, :, :, :, [0, 0, 1, 2, 2, 3, 4]]
    np.testing.assert_array_equal(cast(NumericArray, np.array(actual)), expected)
