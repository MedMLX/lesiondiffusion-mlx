"""MedMLX runner: upstream host-side references, rejection and end-to-end runs."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
import pytest
import safetensors.numpy as safe_numpy
from medmlx_core.errors import AssetNotReadyError, InvalidInputError

from lesiondiffusion_mlx import medmlx
from lesiondiffusion_mlx._arrays import BoolArray, FloatArray, HostArray, NumericArray
from lesiondiffusion_mlx._nifti import nib
from lesiondiffusion_mlx._typing import ArrayConstructor
from lesiondiffusion_mlx.medmlx import run
from lesiondiffusion_mlx.mlx_model import UnetConfig, VqAutoencoder
from lesiondiffusion_mlx.mlx_sampling import DdimSampler
from tests import _runner_cases as cases
from tests._fixture_models import text_weights, unet_weights, vq_weights
from tests.test_equivalence import reference

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Mlx


load_safe = cast("Callable[[str], dict[str, HostArray]]", vars(safe_numpy)["load_file"])
save_safe = cast("Callable[[dict[str, HostArray], str], None]", vars(safe_numpy)["save_file"])

TINY_LMNET = UnetConfig(32, 1, (1, 2), (1, 2), 16)
TINY_LINET = UnetConfig(32, 1, (1, 2), (1, 2), 16, skip_context=True)
BBOX_MM = (14.0, 14.0, 10.0)
RECORDED_CENTERS = [name for name, case in cases.CENTER_CASES.items() if case[3] is None]


def test_upstream_organ_vocabulary() -> None:
    """Compare the organ vocabulary with the source recording."""
    recorded = json.loads(reference("runner_host")["organ_labels"].item())
    for label, name in medmlx.SOURCE_ORGAN_LABELS.values():
        assert recorded[str(label)] == name


@pytest.mark.parametrize("name", RECORDED_CENTERS)
def test_upstream_box_placement(name: str) -> None:
    """Compare lesion-box placement with the source recording."""
    mode, shape, size, _requested, seed = cases.CENTER_CASES[name]
    organ = cases.organ_shape(shape) > 0
    result = medmlx.resolve_bbox_center(
        organ_mask=organ,
        bbox_size=size,
        placement=mode,
        requested_center=None,
        seed=seed,
        organ="colon" if mode in {"within_lumen", "wall_thickening"} else "liver",
    )
    recorded = reference("runner_host")
    assert list(result) == recorded[f"center.{name}"].tolist()
    if f"box.{name}" in recorded:
        assert np.array_equal(medmlx.bbox_mask(organ.shape, result, size), recorded[f"box.{name}"])


def test_upstream_threshold_selection() -> None:
    """Compare lesion-component threshold selection with the source."""
    fixture = reference("selection")
    samples, box = cast(NumericArray, fixture["samples"]), cast(BoolArray, fixture["box"])
    assert np.array_equal(medmlx.select_lesion_component(samples, box), fixture["selected"])
    assert np.array_equal(medmlx.select_lesion_component(samples * 0.1, box), fixture["empty"])


def test_lesion_output_cancels_reconstruction_offset_and_keeps_distant_input() -> None:
    """Lesion output adds only the inpainted change; voxels beyond reach keep the input."""
    rng = np.random.default_rng(7)
    original = rng.normal(40.0, 15.0, (24, 20, 18)).astype(np.float32)
    # Bone above the decoders' 1000 HU clip, beyond and within the lesion's reach.
    original[0, 0, 0] = original[8, 6, 5] = 1500.0
    lesion = np.zeros(original.shape, dtype=bool)
    lesion[10:13, 8:11, 7:10] = True
    # The autoencoder shifts every decoded voxel by -9 HU; inpainting darkens the
    # lesion by 30 HU and its first surrounding shell by 2 HU.
    reconstruction = np.clip(original, -1000, 1000) - 9.0
    generated = reconstruction.copy()
    shell = np.zeros(original.shape, dtype=bool)
    shell[9:14, 7:12, 6:11] = True
    generated[shell] -= 2.0
    generated[lesion] -= 28.0

    localized, facts = medmlx.apply_lesion_change(
        original=original,
        generated=generated,
        reconstruction=reconstruction,
        lesion=lesion,
        reach_voxels=3,
    )

    reached = np.zeros(original.shape, dtype=bool)
    reached[7:16, 5:14, 4:13] = True
    np.testing.assert_array_equal(localized[~reached], original[~reached])
    assert localized[8, 6, 5] == 1500.0
    np.testing.assert_allclose(localized[lesion], original[lesion] - 30.0, atol=1e-4)
    np.testing.assert_allclose(
        localized[shell & ~lesion], original[shell & ~lesion] - 2.0, atol=1e-4
    )
    assert facts["input_reconstruction_offset_hu"] == pytest.approx(-9.0)


def _host(arrays: dict[str, HostArray]) -> dict[str, HostArray]:
    """Convert a mapping of native fixture arrays to host arrays."""
    return {key: cast(NumericArray, np.array(value)) for key, value in arrays.items()}


class _HostArrayFactory:
    float32 = np.float32

    @staticmethod
    def array(value: HostArray, /, *, dtype: type[np.float32]) -> FloatArray:
        """Construct a host array for fixture execution."""
        return np.asarray(value, dtype=dtype)


@pytest.fixture(scope="module")
def tiny_weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Seeded tiny networks in the converted-directory format ``run`` loads."""
    mx = _HostArrayFactory()
    directory = tmp_path_factory.mktemp("lesiondiffusion_weights")
    # The source LDM "scaled linear" schedule; LINet keeps it in float16.
    betas = np.linspace(0.00085**0.5, 0.012**0.5, 1000, dtype=np.float64) ** 2
    alphas_cumprod = np.cumprod(1.0 - betas).astype(np.float32)
    stages = {
        "lmnet": {
            **_host(unet_weights(11, mx, mx.float32, skip_context=False)),
            "alphas_cumprod": alphas_cumprod,
        },
        "linet": {
            **_host(unet_weights(12, mx, mx.float32, skip_context=True)),
            "alphas_cumprod": alphas_cumprod,
        },
        "vq": _host(vq_weights(13, mx, mx.float32)),
        "lmnet_text": _host(text_weights(14, mx)),
        "linet_text": _host(text_weights(15, mx)),
    }
    packed = {
        f"{stage}.{name}": value
        for stage, arrays in stages.items()
        for name, value in arrays.items()
    }
    save_safe(packed, str(directory / "model.safetensors"))
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    words = ["[PAD]", "[UNK]", "liver", "lesion", "irregular", ",", "hypodense", "hyperdense"]
    tokenizer = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    return directory


@pytest.fixture(scope="module")
def conditioned_weights(tiny_weights: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny inpainting network with the checkpoint's cross-attention path."""
    directory = tmp_path_factory.mktemp("conditioned_weights")
    for path in tiny_weights.iterdir():
        shutil.copyfile(path, directory / path.name)
    weights = _host(unet_weights(12, _HostArrayFactory(), np.float32, skip_context=False))
    packed = load_safe(str(directory / "model.safetensors"))
    for key in (
        "input_blocks.0.0.weight",
        "input_blocks.0.0.bias",
        "out.2.weight",
        "out.2.bias",
        "alphas_cumprod",
    ):
        weights[key] = packed[f"linet.{key}"]
    packed = {key: value for key, value in packed.items() if not key.startswith("linet.")}
    packed.update({f"linet.{key}": value for key, value in weights.items()})
    save_safe(packed, str(directory / "model.safetensors"))
    return directory


@pytest.fixture
def tiny_networks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the package pipeline with tiny architectures and crops that fit them."""
    monkeypatch.setattr(medmlx, "LMNET_CONFIG", TINY_LMNET)
    monkeypatch.setattr(medmlx, "LINET_CONFIG", TINY_LINET)
    monkeypatch.setattr(medmlx, "LMNET_CROP", (8, 12, 12))
    monkeypatch.setattr(medmlx, "LINET_CROP", (16, 16, 16))


@pytest.fixture
def scan(tmp_path: Path) -> tuple[Path, Path]:
    """Write a synthetic CT and organ mask on a shared grid."""
    affine = cases.ras_affine((30.5, -42.0, 117.0))
    ct = tmp_path / "ct.nii.gz"
    organ = tmp_path / "liver.nii.gz"
    nib.save(nib.Nifti1Image(cases.ct_data(), affine), ct)
    nib.save(nib.Nifti1Image(cases.organ_block(), affine), organ)
    return ct, organ


def _arguments(
    scan: tuple[Path, Path], output: Path, weights: Path, **overrides: object
) -> cases.RunArguments:
    """Construct a valid runner request for fixture inputs."""
    ct, organ = scan
    # Malformed overrides deliberately exercise the public runner validation.
    return cast(
        cases.RunArguments,
        {
            "image_path": ct,
            "organ_mask_path": organ,
            "output_dir": output,
            "weights_path": weights,
            "organ": "liver",
            "lesion_type": "tumor",
            "bbox_size_mm": list(BBOX_MM),
            "placement": "inside_parenchyma",
            "seed": 4,
            "diffusion_steps": 10,
            "inference_mode": "source",
            **overrides,
        },
    )


@pytest.mark.parametrize(
    ("overrides", "existing"),
    [
        ({"image_path": "absent.nii.gz"}, None),
        ({"seed": -3}, None),
        ({"inference_mode": "unknown"}, None),
        ({"overwrite": "yes"}, None),
        ({}, "synthetic_ct.nii.gz"),
        ({}, "dangling:synthetic_ct.nii.gz"),
    ],
    ids=["path", "range", "choice", "boolean", "output-conflict", "dangling-output"],
)
def test_invalid_arguments_fail_before_downloading_weights(
    overrides: dict[str, object],
    existing: str | None,
    scan: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject every invalid argument and output conflict before any weight download."""

    def download(model_id: str) -> Path:
        """Fail the test: weights must not download for an invalid request."""
        raise AssertionError(f"downloaded {model_id} before checking the arguments")

    monkeypatch.setattr(medmlx, "weights_dir", download)
    output = tmp_path / "out"
    if existing is not None:
        output.mkdir()
        if existing.startswith("dangling:"):
            (output / existing.removeprefix("dangling:")).symlink_to(tmp_path / "absent.nii.gz")
        else:
            (output / existing).touch()
    paths = {key: tmp_path / str(path) for key, path in overrides.items() if key.endswith("_path")}
    arguments = _arguments(scan, output, tmp_path, **{**overrides, **paths})
    arguments.pop("weights_path")
    with pytest.raises(InvalidInputError):
        run(**arguments)
    if existing is not None and existing.startswith("dangling:"):
        assert (output / existing.removeprefix("dangling:")).is_symlink()


def test_missing_or_mismatched_weights_raise_asset_not_ready(
    scan: tuple[Path, Path], tiny_weights: Path, tmp_path: Path
) -> None:
    """Report a missing folder, file or stage, and an unreadable bundle, as asset errors."""
    output = tmp_path / "out"
    with pytest.raises(AssetNotReadyError):
        run(**_arguments(scan, output, tmp_path / "absent"))

    def copy(name: str) -> Path:
        """Copy fixture weights into a named temporary directory."""
        shutil.copytree(tiny_weights, tmp_path / name)
        return tmp_path / name

    weights = copy("missing_safetensors")
    (weights / "model.safetensors").unlink()
    with pytest.raises(AssetNotReadyError):
        run(**_arguments(scan, output, weights))
    weights = copy("no_linet_text")
    packed = load_safe(str(weights / "model.safetensors"))
    kept = {key: value for key, value in packed.items() if not key.startswith("linet_text.")}
    save_safe(kept, str(weights / "model.safetensors"))
    with pytest.raises(AssetNotReadyError):
        run(**_arguments(scan, output, weights, inference_mode="conditioned"))
    # Source mode does not condition LINet on text.
    assert medmlx.require_weights(weights, inference_mode="source") == weights
    weights = copy("corrupt_safetensors")
    (weights / "model.safetensors").write_bytes(b"not a safetensors header")
    with pytest.raises(AssetNotReadyError):
        run(**_arguments(scan, output, weights))


@pytest.fixture
def upstream_inpainting(mx: Mlx, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix discontinuous learned stages while exercising the runner's wiring."""
    fixture = reference("generation")

    def recorded_encode(model: VqAutoencoder, array: Array, mx: Mlx) -> Array:
        """Compare the fixture image passed into upstream encoding."""
        # Normalization is rounded to half at the upstream VQ input boundary.
        # A float32 affine rounding difference may straddle a half midpoint.
        np.testing.assert_allclose(
            cast(NumericArray, np.array(array)),
            cast(NumericArray, fixture["normalized"]),
            atol=2**-11,
            rtol=0,
        )
        return cast(ArrayConstructor, mx.array)(fixture["encoded"])

    class RecordedQuantizer(VqAutoencoder):
        def quantize(self, z: Array, *, chunk_size: int = 4096) -> Array:
            """Compare the fixture latent passed to source quantization."""
            np.testing.assert_array_equal(cast(NumericArray, np.array(z)), fixture["final_latent"])
            return cast(ArrayConstructor, mx.array)(fixture["quantized"])

    class FixedNoiseSampler(DdimSampler):
        attempts: ClassVar[int] = 0

        def __call__(
            self,
            denoiser: Callable[[Array, Array, Array | None], Array],
            condition: Array | None,
            conditioning_image: Array,
            *,
            initial_noise: Array | None = None,
            step_noise: tuple[Array, ...] | None = None,
            mask: Array | None = None,
        ) -> Array:
            """Return recorded sampling outputs while checking the supplied mask."""
            if mask is not None:
                assert condition is None  # Upstream LINet skips text conditioning.
                np.testing.assert_array_equal(
                    cast(NumericArray, np.array(conditioning_image)), fixture["latent_crop"]
                )
                np.testing.assert_array_equal(
                    cast(NumericArray, np.array(mask)), fixture["latent_mask"]
                )
                return cast(ArrayConstructor, mx.array)(fixture["sampled_crop"])
            index = FixedNoiseSampler.attempts
            FixedNoiseSampler.attempts += 1
            initial, noise = fixture[f"lm_initial.{index}"], fixture[f"lm_noise.{index}"]
            return super().__call__(
                denoiser,
                condition,
                conditioning_image,
                initial_noise=cast(ArrayConstructor, mx.array)(initial),
                step_noise=tuple(cast(ArrayConstructor, mx.array)(item) for item in noise),
                mask=mask,
            )

    monkeypatch.setattr(medmlx, "DdimSampler", FixedNoiseSampler)
    monkeypatch.setattr(medmlx, "encode_source", recorded_encode)
    monkeypatch.setattr(medmlx, "VqAutoencoder", RecordedQuantizer)


def test_run_matches_upstream_end_to_end(
    scan: tuple[Path, Path],
    tiny_weights: Path,
    tiny_networks: None,
    tmp_path: Path,
    mx: Mlx,
    upstream_inpainting: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare the native runner with the recorded upstream pipeline."""
    fixture = reference("generation")
    output = tmp_path / "out"
    output.mkdir()
    victim = tmp_path / "victim.nii.gz"
    victim.write_bytes(b"original")
    monkeypatch.setattr(medmlx.time, "monotonic_ns", lambda: 0)
    (output / f".lesiondiffusion-{os.getpid()}-0-target_mask.nii.gz").symlink_to(victim)

    result = run(**_arguments(scan, output, tiny_weights))
    assert victim.read_bytes() == b"original"
    images = {role: nib.load(path) for role, path in result["outputs"].items()}
    assert set(images) == {"synthetic_image", "target_mask", "organ_mask"}
    source = nib.load(scan[0])
    for image in images.values():
        assert image.shape == source.shape
        assert np.array_equal(image.affine, source.affine)
    actual = cast(NumericArray, np.asanyarray(images["synthetic_image"].dataobj))
    # Fixed upstream quantized input isolates native VQ decoding and HU writing.
    # The decoder's 0.002 normalized absolute bound maps to at most 8 HU.
    # Native generation is exercised separately below: nearest-code switches make
    # a uniform HU bound invalid for independently computed latents.
    np.testing.assert_allclose(actual, cast(NumericArray, fixture["output"]), atol=8, rtol=0)
    assert np.array_equal(np.asanyarray(images["target_mask"].dataobj), fixture["lesion"])
    assert np.array_equal(np.asanyarray(images["organ_mask"].dataobj), cases.organ_block())


@pytest.mark.parametrize("preexisting", ["none", "files", "symlink"])
def test_output_publication_restores_all_roles_after_move_failure(
    scan: tuple[Path, Path],
    tiny_networks: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting: str,
) -> None:
    """A failed final move leaves neither partial new files nor mixed old and new files."""
    options = medmlx.LesionDiffusionOptions(
        organ="liver",
        lesion_type="tumor",
        bbox_size_mm=BBOX_MM,
        placement="inside_parenchyma",
        diffusion_steps=10,
        seed=4,
        inference_mode="source",
    )
    geometry = medmlx.prepare_geometry(source=scan[0], supplied_mask=scan[1], options=options)
    image = np.zeros(geometry.reference.shape, dtype=np.float32)
    lesion = np.zeros(geometry.reference.shape, dtype=np.uint8)
    lesion[tuple(size // 2 for size in geometry.output_reference.shape)] = 2
    outputs = medmlx.output_paths(tmp_path / "out")
    old = {role: f"old {role}".encode() for role in outputs}
    if preexisting != "none":
        for role, path in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(old[role])
    victim = tmp_path / "victim.nii.gz"
    if preexisting == "symlink":
        victim.write_bytes(b"outside")
        outputs["synthetic_image"].unlink()
        outputs["synthetic_image"].symlink_to(victim)

    replace = Path.replace

    def fail_second_move(source: Path, destination: Path) -> Path:
        if destination == outputs["target_mask"] and source.parent.name.startswith(
            ".lesiondiffusion-"
        ):
            raise OSError("injected final move failure")
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_second_move)
    with pytest.raises(OSError, match="injected final move failure"):
        medmlx.materialize_arrays(
            outputs=outputs, geometry=geometry, image_data=image, lesion_data=lesion
        )

    for role, path in outputs.items():
        if preexisting == "symlink" and role == "synthetic_image":
            assert path.is_symlink() and path.resolve() == victim
            assert victim.read_bytes() == b"outside"
        elif preexisting != "none":
            assert path.read_bytes() == old[role]
        else:
            assert not path.exists()


def test_native_generation_is_finite_and_repeats_its_seed(
    scan: tuple[Path, Path], conditioned_weights: Path, tiny_networks: None, tmp_path: Path, mx: Mlx
) -> None:
    """Run the default conditioned mode natively; the same seed repeats exactly."""

    def generate(name: str) -> dict[str, NumericArray]:
        """Run conditioned generation and read every written role."""
        result = run(
            **_arguments(scan, tmp_path / name, conditioned_weights, inference_mode="conditioned")
        )
        return {
            role: cast(NumericArray, np.asanyarray(nib.load(path).dataobj))
            for role, path in result["outputs"].items()
        }

    first, again = generate("first"), generate("again")
    for role, actual in first.items():
        np.testing.assert_array_equal(actual, again[role])
    image = first["synthetic_image"]
    assert image.dtype == np.float32 and np.isfinite(image).all()
    assert image.min() >= -1000 and image.max() <= 1000
    assert first["target_mask"].dtype == np.uint8
    assert set(np.unique(first["target_mask"])) <= {0, 1}
    assert np.any(first["target_mask"])
