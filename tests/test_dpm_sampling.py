"""DPM-Solver++ trajectories against the pinned independent Diffusers oracle."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from lesiondiffusion_mlx._arrays import HostArray
from lesiondiffusion_mlx._typing import ArrayConstructor
from lesiondiffusion_mlx.mlx_sampling import Dpmpp2mSampler
from tests._fixture_models import sampling_denoiser, sampling_inputs
from tests.test_equivalence import assert_numerical

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Mlx

FIXTURE = Path(__file__).parent / "fixtures/darwin-arm64/dpmpp_2m.npz"


def _host(array: Array) -> HostArray:
    """Materialize FP32 solver state for its CPU oracle comparison."""
    return np.array(array, dtype=np.float32)


@pytest.mark.parametrize("stage", ["lmnet", "linet"])
@pytest.mark.parametrize("steps", [4, 8, 25])
def test_dpm_midpoint_matches_diffusers_trajectory(stage: str, steps: int, mx: Mlx) -> None:
    """Preserve first/second/final order and known regions on the real alpha schedules."""
    with np.load(FIXTURE, allow_pickle=False) as archive:
        prefix = f"{stage}_{steps}."
        data = cast(
            dict[str, HostArray],
            {
                key.removeprefix(prefix): archive[key]
                for key in archive.files
                if key.startswith(prefix)
            },
        )
    array = cast(ArrayConstructor, mx.array)
    sampler = Dpmpp2mSampler(array(data["alphas_cumprod"]), mx, steps)
    np.testing.assert_array_equal(np.array(sampler.timesteps), data["timesteps"])
    image = array(data["conditioning_image"])
    mask = array(data["mask"]) if stage == "linet" else None
    predictions = data["predictions"]
    observed: list[HostArray] = []

    def denoiser(x: Array, timestep: Array, context: Array | None) -> Array:
        index = len(observed)
        np.testing.assert_array_equal(
            np.array(timestep, dtype=np.int32), data["timesteps"][index : index + 1]
        )
        observed.append(_host(x[:, : image.shape[1]]))
        return array(predictions[index])

    actual = sampler(denoiser, None, image, initial_noise=array(data["initial_noise"]), mask=mask)
    mx.eval(actual)
    assert actual.dtype == mx.float32
    assert_numerical(actual, data["output"])
    assert_numerical(np.stack(observed), data["projected_inputs"])
    assert np.isfinite(_host(actual)).all()
    if mask is not None:
        outside = np.broadcast_to(data["mask"] == 0, actual.shape)
        np.testing.assert_array_equal(_host(actual)[outside], data["conditioning_image"][outside])


@pytest.mark.parametrize("inpainting", [False, True])
def test_prepared_dpm_keeps_history_rng_and_known_region(inpainting: bool, mx: Mlx) -> None:
    """Compilation retains an evolving prediction history and exactly one noise draw."""
    fixture = sampling_inputs(71, inpainting=inpainting)
    array = cast(ArrayConstructor, mx.array)
    image, condition = array(fixture["image"]), array(fixture["condition"])
    mask = array(fixture["mask"]) if inpainting else None
    sampler = Dpmpp2mSampler(array(fixture["alphas"]), mx, 4)
    denoiser = sampling_denoiser(mx)

    def bound(x: Array, timestep: Array) -> Array:
        return denoiser(x, timestep, condition)

    prepared = sampler.prepare(bound, image, mask=mask)
    mx.random.seed(123)
    expected = [_host(sampler(denoiser, condition, image, mask=mask)) for _ in range(2)]
    expected_next = _host(mx.random.normal((7,)))
    mx.random.seed(123)
    actual = [_host(prepared()) for _ in range(2)]
    actual_next = _host(mx.random.normal((7,)))
    for result, reference in zip(actual, expected, strict=True):
        assert_numerical(result, reference)
        if inpainting:
            outside = np.broadcast_to(fixture["mask"] == 0, result.shape)
            np.testing.assert_array_equal(result[outside], fixture["image"][outside])
    np.testing.assert_array_equal(actual_next, expected_next)
