"""Independent source orientation, edge crop, and CT intensity-knot expectations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Mlx

import numpy as np
import pytest
from medmlx_core.errors import InvalidInputError

from lesiondiffusion_mlx._arrays import NumericArray
from lesiondiffusion_mlx._typing import ArrayConstructor
from lesiondiffusion_mlx.mlx_pipeline import piecewise_ct, ras_to_source, source_crop, source_to_ras


def test_ras_source_mapping_roundtrips_asymmetric_landmarks() -> None:
    """Preserve asymmetric landmarks through the RAS source mapping."""
    image = np.arange(7 * 11 * 13).reshape(7, 11, 13)
    source = ras_to_source(image)
    assert source.shape == (13, 11, 7)
    assert source[2, 5, 0] == image[-1, 5, 2]
    np.testing.assert_array_equal(source_to_ras(source), image)


def test_source_crop_clamps_to_edges_without_moving_grid() -> None:
    """Clamp source crops at image edges without shifting the grid."""
    mask = np.zeros((140, 200, 210), dtype=bool)
    mask[1:4, 195:199, 205:209] = True
    crop = source_crop(mask, (128, 128, 128))
    assert crop == (slice(0, 128), slice(72, 200), slice(82, 210))
    small = np.zeros((130, 100, 210), dtype=bool)
    small[1, 1, 1] = True
    with pytest.raises(InvalidInputError, match=">= 152"):
        source_crop(small, (64, 152, 152))


def test_native_ct_normalization_matches_source_knots(mx: Mlx) -> None:
    """Compare native CT normalization at source intensity knots."""
    values = cast(ArrayConstructor, mx.array)([-1500, -1000, -600, -200, 0, 200, 600, 1000, 1500])
    actual = piecewise_ct(values, mx)
    inverse = piecewise_ct(actual, mx, inverse=True)
    mx.eval(actual, inverse)
    np.testing.assert_allclose(
        cast(NumericArray, np.array(actual)), [0, 0, 0.1, 0.2, 0.5, 0.8, 0.9, 1, 1], atol=1e-7
    )
    np.testing.assert_allclose(
        cast(NumericArray, np.array(inverse)),
        [-1000, -1000, -600, -200, 0, 200, 600, 1000, 1000],
        atol=1e-3,
    )
