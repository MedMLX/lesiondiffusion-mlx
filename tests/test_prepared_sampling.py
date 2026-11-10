"""Condition binding and stochastic DDIM contracts for compiled inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from lesiondiffusion_mlx._typing import ArrayConstructor
from lesiondiffusion_mlx.mlx_model import LdmUnet, UnetConfig
from lesiondiffusion_mlx.mlx_sampling import DdimSampler
from tests._fixture_models import sampling_denoiser, sampling_inputs, unet_weights

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Dtype, Mlx
    from tests._fixture_models import ArrayFactory


@pytest.mark.parametrize("inpainting", [False, True])
@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_prepared_sampling_keeps_rng_progression_and_unmasked_voxels(
    inpainting: bool, eta: float, mx: Mlx
) -> None:
    """Compiled attempts retain the source draws, schedule and inpainting boundary."""
    fixture = sampling_inputs(71, inpainting=inpainting)
    array = cast(ArrayConstructor, mx.array)
    condition, image = array(fixture["condition"]), array(fixture["image"])
    mask = array(fixture["mask"]) if inpainting else None
    sampler = DdimSampler(array(fixture["alphas"]), mx, 4, eta=eta)
    denoiser = sampling_denoiser(mx)

    def bound(x: Array, timestep: Array) -> Array:
        return denoiser(x, timestep, condition)

    prepared = sampler.prepare(bound, image, mask=mask)
    mx.random.seed(123)
    expected = [
        np.array(sampler(denoiser, condition, image, mask=mask), dtype=np.float32) for _ in range(2)
    ]
    next_expected = np.array(mx.random.normal((7,)), dtype=np.float32)
    mx.random.seed(123)
    actual = [np.array(prepared(), dtype=np.float32) for _ in range(2)]
    next_actual = np.array(mx.random.normal((7,)), dtype=np.float32)
    for result, reference in zip(actual, expected, strict=True):
        # Existing FP32 DDIM oracle bounds; fused pointwise arithmetic can change
        # final float32 rounding, but neither discrete masks nor RNG state may drift.
        np.testing.assert_allclose(result, reference, atol=1e-5, rtol=1e-5)
        if inpainting:
            outside = np.broadcast_to(fixture["mask"] == 0, result.shape)
            np.testing.assert_array_equal(result[outside], fixture["image"][outside])
    np.testing.assert_array_equal(next_actual, next_expected)


@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_context_bindings_do_not_cache_evolving_image_or_other_request(
    dtype_name: str, mx: Mlx
) -> None:
    """Two bindings on one model retain their own context while the image evolves."""
    dtype = getattr(mx, dtype_name)
    weights = unet_weights(17, cast("ArrayFactory[Array, Dtype]", mx), dtype, skip_context=False)
    model = LdmUnet(
        weights,
        mx,
        UnetConfig(32, 1, (1, 2), channel_mult=(1, 2)),
        source_arithmetic=False,
        fused_attention=True,
    )
    rng = np.random.default_rng(31)
    array = cast(ArrayConstructor, mx.array)
    x = array(rng.normal(size=(1, 4, 8, 8, 8)).astype(np.float32))
    first = array(rng.normal(size=(1, 3, 6)).astype(np.float32))
    second = array(rng.normal(size=(1, 5, 6)).astype(np.float32))
    timestep = array([17], dtype=mx.int32)
    ordinary = mx.compile(model.__call__)
    bound_first = mx.compile(model.prepare_denoiser(first))
    bound_second = mx.compile(model.prepare_denoiser(second))
    for image, context, bound in (
        (x, first, bound_first),
        (x, second, bound_second),
        (x * 0.5, first, bound_first),
    ):
        expected = ordinary(image, timestep, context)
        actual = bound(image, timestep)
        mx.eval(expected, actual)
        np.testing.assert_array_equal(
            np.array(actual, dtype=np.float32), np.array(expected, dtype=np.float32)
        )
