"""Native MLX comparisons against pinned upstream Torch CPU recordings."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from lesiondiffusion_mlx._arrays import HostArray, NumericArray
from lesiondiffusion_mlx._reference import REFERENCE

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Dtype, Mlx
    from tests._fixture_models import ArrayFactory

import json
from pathlib import Path

import numpy as np
import pytest
from medmlx_core.errors import InvalidInputError, ModelExecutionError

from lesiondiffusion_mlx._typing import ArrayConstructor
from lesiondiffusion_mlx.mlx_model import BiomedTextEncoder, LdmUnet, UnetConfig, VqAutoencoder
from lesiondiffusion_mlx.mlx_pipeline import (
    encode_attributes,
    encode_source,
    piecewise_ct,
    ras_to_source,
    source_crop,
    source_to_ras,
)
from lesiondiffusion_mlx.mlx_sampling import DdimSampler, lesion_diffusion_timesteps
from tests._fixture_models import sampling_denoiser, text_weights, unet_weights, vq_weights

FIXTURES = Path(__file__).parent / "fixtures/darwin-arm64"


def reference(name: str) -> dict[str, HostArray]:
    """Load a recorded Torch CPU fixture after checking it matches the pinned upstream revisions."""
    with np.load(FIXTURES / (name + ".npz"), allow_pickle=False) as fixture:
        assert fixture["schema_version"].item() == 1
        assert fixture["upstream_revision"].item() == REFERENCE["revision"]
        assert fixture["text_revision"].item() == REFERENCE["text_revision"]
        assert fixture["backend"].item() == "torch-cpu"
        assert fixture["recording_platform"].item() == "darwin-arm64"
        return cast(dict[str, HostArray], dict(fixture))


def assert_bits(actual: Array | HostArray, expected: HostArray) -> None:
    """Require identical dtype and values (for operations that are exact by construction)."""
    actual = np.array(actual)
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


def assert_numerical(actual: Array | HostArray, expected: HostArray) -> None:
    """Compare native arrays with a recorded numerical reference."""
    actual = np.array(actual)
    assert actual.dtype == expected.dtype
    # fp32: reduction order and fused multiply-add vary between Torch CPU and MLX.
    # fp16: each learned layer rounds to half; allow two output ulps at unit scale.
    # These cross-framework bounds are provisional (see docs/technical.md).
    atol, rtol = (2e-3, 2e-3) if expected.dtype == np.float16 else (2e-6, 2e-5)
    assert np.issubdtype(expected.dtype, np.number)
    np.testing.assert_allclose(
        cast(NumericArray, actual), cast(NumericArray, expected), atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("name", ["unet_mask", "unet_inpainting"])
def test_unet_reference(name: str, mx: Mlx) -> None:
    """Compare seeded native UNet output with the source recording."""
    fixture = reference(name)
    config_data = json.loads(fixture["config"].item())
    for key in ("attention_resolutions", "channel_mult"):
        config_data[key] = tuple(config_data[key])
    config = UnetConfig(**config_data)
    weights = unet_weights(
        int(fixture["weight_seed"]),
        cast("ArrayFactory[Array, Dtype]", mx),
        getattr(mx, fixture["dtype"].item()),
        skip_context=config.skip_context,
    )
    model = LdmUnet(weights, mx, config)
    result = model(
        cast(ArrayConstructor, mx.array)(fixture["x"]),
        cast(ArrayConstructor, mx.array)(fixture["timesteps"]),
        cast(ArrayConstructor, mx.array)(fixture["context"]),
    )
    assert_numerical(result, fixture["output"])


@pytest.mark.parametrize("name", ["silu", "softmax"])
def test_unet_half_activations_reference(name: str, mx: Mlx) -> None:
    """Compare half-precision UNet activations with the source recording."""
    fixture = reference("unet_activations")
    expected = fixture[name + "_output"]
    model = LdmUnet({}, mx, UnetConfig(32, 1, (1, 2)))
    actual = np.array(
        getattr(model, name)(cast(ArrayConstructor, mx.array)(fixture[name + "_input"]))
    )
    assert actual.dtype == expected.dtype
    error = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    ulps = error / np.abs(np.spacing(expected)).astype(np.float32)
    # Single-round half outputs may straddle a rounding midpoint: at most one
    # ulp, with RMS <= 0.1 ulp (at most 1% one-ulp differences). Source SiLU vs
    # fp32 sigmoid/product gives RMS 0.015625; rounding the sigmoid and the product
    # to half separately gives 0.458.
    assert float(ulps.max()) <= 1
    assert float(np.sqrt(np.mean(ulps**2))) <= 0.1


@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_vq_reference(dtype_name: str, mx: Mlx) -> None:
    """Compare the native VQ network with recorded source output."""
    fixture = reference("vq_" + dtype_name)
    dtype = getattr(mx, dtype_name)
    model = VqAutoencoder(
        vq_weights(int(fixture["weight_seed"]), cast("ArrayFactory[Array, Dtype]", mx), dtype), mx
    )
    x, z = (
        cast(ArrayConstructor, mx.array)(fixture["x"], dtype=dtype),
        cast(ArrayConstructor, mx.array)(fixture["z"], dtype=dtype),
    )
    assert_numerical(model.encode(x), fixture["encoded"])
    assert_bits(model.quantize(z, chunk_size=3), fixture["quantized"])
    assert_numerical(model.decode(z), fixture["decoded"])
    assert_numerical(model.decode(z, quantize=False), fixture["decoded_unquantized"])


@pytest.fixture(scope="module")
def text_model(mx: Mlx) -> BiomedTextEncoder:
    """Provide the seeded native text model and fixture."""
    fixture = reference("text")
    return BiomedTextEncoder(
        text_weights(int(fixture["weight_seed"]), cast("ArrayFactory[Array, Dtype]", mx)), mx
    )


def test_text_reference(mx: Mlx, text_model: BiomedTextEncoder) -> None:
    """Compare native text encoding with the source recording."""
    fixture = reference("text")
    assert_numerical(
        text_model(cast(ArrayConstructor, mx.array)(fixture["tokens"])), fixture["output"]
    )


def test_attributes_reference(mx: Mlx, text_model: BiomedTextEncoder, tmp_path: Path) -> None:
    """Compare generated text attributes with the source recording."""
    fixture = reference("attributes")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text(fixture["tokenizer"].item())
    attributes = json.loads(fixture["attributes"].item())
    assert_numerical(
        encode_attributes(attributes, text_model, tokenizer_path, mx), fixture["output"]
    )


@pytest.mark.parametrize("name", ["ddim_mask", "ddim_inpainting"])
def test_ddim_reference(name: str, mx: Mlx) -> None:
    """Compare native DDIM sampling with the source recording."""
    fixture = reference(name)
    sampler = DdimSampler(
        cast(ArrayConstructor, mx.array)(fixture["alphas"]), mx, int(fixture["steps"])
    )
    assert np.array_equal(sampler.timesteps, fixture["timesteps"])
    for attr, key in (
        ("alphas", "alphas_selected"),
        ("previous", "previous"),
        ("sqrt_one_minus", "sqrt_one_minus"),
        ("sigmas", "sigmas"),
    ):
        assert_numerical(getattr(sampler, attr), fixture[key])
    mask = cast(ArrayConstructor, mx.array)(fixture["mask"]) if "mask" in fixture else None
    denoiser = sampling_denoiser(mx)
    condition, image = (
        cast(ArrayConstructor, mx.array)(fixture["condition"]),
        cast(ArrayConstructor, mx.array)(fixture["image"]),
    )
    result = sampler(
        denoiser,
        condition,
        image,
        initial_noise=cast(ArrayConstructor, mx.array)(fixture["initial"]),
        step_noise=tuple(cast(ArrayConstructor, mx.array)(value) for value in fixture["noise"]),
        mask=mask,
    )
    assert_numerical(result, fixture["output"])
    # Framework RNG streams differ. Repeatability is checked within MLX only.
    mx.random.seed(410)
    first = cast(HostArray, np.array(sampler(denoiser, condition, image, mask=mask)))
    mx.random.seed(410)
    assert_bits(sampler(denoiser, condition, image, mask=mask), first)


def test_pipeline_reference(mx: Mlx) -> None:
    """Compare the native generation pipeline with the source recording."""
    fixture = reference("pipeline")
    assert_bits(ras_to_source(fixture["orientation"]), fixture["source"])
    assert_bits(source_to_ras(fixture["source"]), fixture["orientation"])
    assert_numerical(
        piecewise_ct(cast(ArrayConstructor, mx.array)(fixture["ct"]), mx), fixture["ct_output"]
    )
    # HU values require an absolute bound at zero. Four float32 ulps at the
    # 1000 HU clamp cover affine rounding; Metal's observed error was 1.53e-5 HU.
    np.testing.assert_allclose(
        cast(
            NumericArray,
            np.array(
                piecewise_ct(
                    cast(ArrayConstructor, mx.array)(fixture["normalized"]), mx, inverse=True
                )
            ),
        ),
        cast(NumericArray, fixture["inverse_output"]),
        atol=4 * np.spacing(np.float32(1000)),
        rtol=0,
    )
    model = VqAutoencoder(
        vq_weights(int(fixture["weight_seed"]), cast("ArrayFactory[Array, Dtype]", mx), mx.float16),
        mx,
    )
    for prefix in ("odd", "long"):
        x = cast(ArrayConstructor, mx.array)(fixture[prefix + "_input"], dtype=mx.float16)
        assert_numerical(encode_source(model, x, mx), fixture[prefix + "_output"])


@pytest.mark.parametrize(
    ("steps", "total", "message"),
    [
        (False, 1000, "steps must be a positive integer"),
        (0, 1000, "steps must be a positive integer"),
        (3, 2, "schedule must contain at least as many timesteps"),
        (6, 25, "produce out-of-bounds indices"),
    ],
)
def test_schedule_rejects_invalid_source_indices(steps: int, total: int, message: str) -> None:
    """Reject invalid source schedule indices."""
    with pytest.raises(ValueError, match=message):
        lesion_diffusion_timesteps(steps, total)


def test_ddim_rejects_mismatched_noise(mx: Mlx) -> None:
    """Reject diffusion noise with incompatible shape."""
    fixture = reference("ddim_mask")
    sampler = DdimSampler(cast(ArrayConstructor, mx.array)(fixture["alphas"]), mx, 4)
    image = cast(ArrayConstructor, mx.array)(fixture["image"])
    with pytest.raises(ValueError, match="initial noise shape"):
        sampler(sampling_denoiser(mx), None, image, initial_noise=mx.zeros((1, 2, 2, 2, 2)))
    with pytest.raises(ValueError, match="one noise tensor per actual source timestep"):
        sampler(sampling_denoiser(mx), None, image, step_noise=())


def test_crop_rejects_empty_or_undersized_input() -> None:
    """Reject empty or undersized source crops."""
    with pytest.raises(ModelExecutionError, match="cannot crop an empty lesion mask"):
        source_crop(np.zeros((4, 4, 4), bool), (4, 4, 4))
    with pytest.raises(InvalidInputError, match="source axis 0 >= 4; got 3"):
        source_crop(np.ones((3, 4, 4), bool), (4, 4, 4))


def test_crop_matches_upstream() -> None:
    """Compare source crop geometry with the upstream recording."""
    fixture = reference("pipeline")
    crop = source_crop(fixture["crop_mask"], tuple(fixture["crop_shape"]))
    assert np.array_equal([(axis.start, axis.stop) for axis in crop], fixture["crop"])
