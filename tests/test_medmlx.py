"""MedMLX file contracts and upstream preprocessing/generation reference checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, cast

from lesiondiffusion_mlx._arrays import BoolArray, FloatArray, HostArray, NumericArray

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Mlx

import inspect
import json
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import safetensors.numpy as safe_numpy
from medmlx_core.errors import AssetNotReadyError, InvalidInputError, ModelExecutionError
from medmlx_core.weights import weights_repo

from lesiondiffusion_mlx import medmlx
from lesiondiffusion_mlx._nifti import nib
from lesiondiffusion_mlx._typing import ArrayConstructor
from lesiondiffusion_mlx.medmlx import MODEL, run
from lesiondiffusion_mlx.mlx_model import UnetConfig, VqAutoencoder
from tests import _runner_cases as cases
from tests._fixture_models import text_weights, unet_weights, vq_weights
from tests.test_equivalence import reference


class SamplerFixture(Protocol):
    attempts: ClassVar[int]


ROOT = Path(__file__).resolve().parents[1]

load_safe = cast("Callable[[str], dict[str, HostArray]]", vars(safe_numpy)["load_file"])
save_safe = cast("Callable[[dict[str, HostArray], str], None]", vars(safe_numpy)["save_file"])


def test_automatic_weights_use_published_bundle() -> None:
    """Resolve the public model ID to its published Hub bundle."""
    assert weights_repo(medmlx.MODEL_ID) == "MedMLX/lesiondiffusion-mlx"


def test_entry_point_and_declaration() -> None:
    """Verify the registered runner declaration and entry point."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert import_module("lesiondiffusion_mlx").__version__ == project["version"]
    assert project["entry-points"]["medmlx.models"] == {
        "lesiondiffusion": "lesiondiffusion_mlx.medmlx:MODEL"
    }
    assert {"id", "label", "task", "run"} <= MODEL.keys()
    assert MODEL["id"] == "lesiondiffusion" and MODEL["task"] == "generation"
    module, name = MODEL["run"].split(":")
    assert getattr(import_module(module), name) is run
    parameters = inspect.signature(run).parameters
    assert all(item.kind is inspect.Parameter.KEYWORD_ONLY for item in parameters.values())
    assert parameters["overwrite"].default is False
    assert parameters["diffusion_steps"].default == 200
    assert parameters["seed"].default == 23
    assert parameters["weights_path"].default is None
    assert parameters["inference_mode"].default == "conditioned"
    assert {"image_path", "organ_mask_path", "output_dir", "weights_path"} <= parameters.keys()
    # MedMLX checks optional ``*_path`` inputs; a supplied lesion mask replaces the box size.
    assert parameters["lesion_mask_path"].default is None
    assert parameters["bbox_size_mm"].default is None


def test_upstream_vocabulary_and_controls() -> None:
    """Compare organ vocabulary and controls with the source recording."""
    recorded = json.loads(reference("runner_host")["organ_labels"].item())
    for label, name in medmlx.SOURCE_ORGAN_LABELS.values():
        assert recorded[str(label)] == name
    for overrides in (
        {"lesion_type": "stone"},
        {"organ": "spleen"},
        {"organ": "brain"},
        {"shape": "spiky"},
        {"density_variation": "mixed"},
        {"diffusion_steps": 5},
    ):
        options = cast(cases.OptionsArguments, {**cases.OPTION_BASE, **overrides})
        medmlx.LesionDiffusionOptions(**options)


@pytest.mark.parametrize("name", cases.REFERENCE_CASES)
def test_reference_loading_contract(name: str, tmp_path: Path) -> None:
    """Validate NIfTI image loading and geometry contracts."""
    suffix, data, affine = cases.reference_input(name)
    path = tmp_path / f"reference_{name}{suffix}"
    if suffix == ".img":
        path.write_bytes(b"not a nifti")
    else:
        nib.save(nib.Nifti1Image(data, affine), path)
    if name.startswith("valid"):
        image, loaded = medmlx.load_reference_nifti(path)
        assert np.array_equal(loaded, data)
        assert np.array_equal(image.affine, affine)
    elif name in {"lps", "two_mm"}:
        # Not yet RAS 1 mm, but axis-aligned: resampled rather than refused,
        # following upstream's own ``preprocess.standardize_image``.
        image, loaded = medmlx.load_reference_nifti(path)
        assert tuple(str(code) for code in nib.aff2axcodes(image.affine)) == ("R", "A", "S")
        assert np.allclose(image.header.get_zooms()[:3], (1.0, 1.0, 1.0))
        assert np.allclose(image.affine[:3, :3], np.eye(3), atol=1e-4, rtol=0)
        assert all(size % 2 == 0 for size in image.shape)
        if name == "lps":
            # A pure LPS->RAS reorientation flips the first two axes without
            # resampling (zoom factor 1); voxel count is unchanged.
            assert loaded.shape == data.shape
            assert np.array_equal(loaded, data[::-1, ::-1, :])
        else:
            from scipy.ndimage import zoom

            assert loaded.shape == tuple(2 * size for size in data.shape)
            assert np.allclose(loaded, zoom(data, (2.0, 2.0, 2.0), order=1))
    else:
        with pytest.raises(InvalidInputError):
            medmlx.load_reference_nifti(path)


@pytest.mark.parametrize(
    ("ct_unit", "mask_unit"),
    [("mm", "unknown"), ("meter", "mm"), ("micron", "meter"), ("unknown", "micron")],
    ids=["mm", "meter", "micron", "unknown"],
)
def test_reference_loading_normalizes_spatial_units(
    ct_unit: str, mask_unit: str, tmp_path: Path
) -> None:
    """Convert source spatial units to the expected millimeter grid."""
    data, affine = cases.ct_data(), cases.ras_affine((12.0, -18.0, 24.0))
    organ = cases.organ_block()
    ct_path, mask_path = tmp_path / "ct.nii.gz", tmp_path / "mask.nii.gz"
    units_per_mm = {"mm": 1.0, "meter": 0.001, "micron": 1000.0, "unknown": 1.0}
    for path, values, unit in ((ct_path, data, ct_unit), (mask_path, organ, mask_unit)):
        encoded_affine = affine.copy()
        encoded_affine[:3] *= units_per_mm[unit]
        image = nib.Nifti1Image(values, encoded_affine)
        image.header.set_xyzt_units(unit, "sec")
        image.set_qform(encoded_affine, code=2)
        image.set_sform(encoded_affine, code=4)
        nib.save(image, path)

    reference_image, loaded = medmlx.load_reference_nifti(ct_path)
    np.testing.assert_array_equal(loaded, data)
    normalized_path = tmp_path / "normalized.nii.gz"
    nib.save(reference_image, normalized_path)
    reference_image = nib.load(normalized_path)
    np.testing.assert_allclose(reference_image.affine, affine, rtol=0, atol=2e-6)
    assert reference_image.header.get_xyzt_units() == ("mm", "sec")
    for read_form, expected_code in (
        (reference_image.get_qform, 2),
        (reference_image.get_sform, 4),
    ):
        form, code = read_form(coded=True)
        assert code == expected_code
        assert form is not None
        np.testing.assert_allclose(form, affine, rtol=0, atol=2e-6)
    mask, _provenance = medmlx.prepare_organ_mask(
        reference=reference_image, supplied_mask=mask_path
    )
    np.testing.assert_array_equal(mask, organ > 0)


@pytest.mark.parametrize("name", cases.ORGAN_MASK_CASES)
def test_organ_mask_contract(name: str, tmp_path: Path) -> None:
    """Validate organ mask labels and geometry."""
    reference_image = nib.Nifti1Image(*cases.reference_input("valid")[1:])
    data, affine = cases.organ_mask_input(name)
    path = tmp_path / "organ.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), path)
    if name in {"binary", "binary_float"}:
        mask, provenance = medmlx.prepare_organ_mask(reference=reference_image, supplied_mask=path)
        assert np.array_equal(mask, data > 0)
        assert provenance["producer"] == "supplied_organ_mask"
    else:
        with pytest.raises(InvalidInputError):
            medmlx.prepare_organ_mask(reference=reference_image, supplied_mask=path)


@pytest.mark.parametrize("invalid_label", [-1.0, 0.5, 2.0])
def test_downsampling_does_not_hide_invalid_organ_labels(
    tmp_path: Path, invalid_label: float
) -> None:
    """Reject invalid organ labels before spatial downsampling."""
    affine = np.diag((0.5, 0.5, 0.5, 1.0))
    source, mask = tmp_path / "ct.nii.gz", tmp_path / "organ.nii.gz"
    nib.save(nib.Nifti1Image(cases.ct_data(), affine), source)
    labels = np.ones(cases.CT_SHAPE, dtype=np.float32)
    labels[1, 1, 1] = invalid_label
    nib.save(nib.Nifti1Image(labels, affine), mask)
    reference_image, _data = medmlx.load_reference_nifti(source)
    with pytest.raises(InvalidInputError):
        medmlx.prepare_organ_mask(reference=reference_image, supplied_mask=mask)


@pytest.mark.parametrize("name", cases.LESION_MASK_CASES)
def test_supplied_lesion_mask_contract(name: str, tmp_path: Path, tiny_networks: None) -> None:
    """Validate a supplied lesion mask and its spatial grid."""
    source, organ, lesion = (
        tmp_path / "ct.nii.gz",
        tmp_path / "organ.nii.gz",
        tmp_path / "lesion.nii.gz",
    )
    nib.save(nib.Nifti1Image(cases.ct_data(), cases.ras_affine()), source)
    nib.save(nib.Nifti1Image(cases.organ_block(), cases.ras_affine()), organ)
    nib.save(nib.Nifti1Image(*cases.lesion_mask_input(name)), lesion)
    options: cases.OptionsArguments = {**cases.OPTION_BASE, "bbox_size_mm": None}

    def prepare():
        """Prepare geometry for the supplied-lesion fixture."""
        return medmlx.prepare_geometry(
            source=source,
            supplied_mask=organ,
            supplied_lesion=lesion,
            options=medmlx.LesionDiffusionOptions(**options),
        )

    message = cases.LESION_MASK_CASES[name]
    if message is not None:
        with pytest.raises(InvalidInputError, match=message):
            prepare()
        return
    geometry = prepare()
    assert geometry.lesion_mask is not None
    np.testing.assert_array_equal(geometry.lesion_mask, cases.lesion_block() > 0)
    # Lesion bounds [9:14, 10:13, 7:11] widen to even indices [8:14, 10:14, 6:12].
    assert geometry.center == (11, 12, 9)
    assert geometry.bbox_size_voxel == (6, 4, 6)
    box = np.zeros(cases.CT_SHAPE, dtype=np.uint8)
    box[8:14, 10:14, 6:12] = 2
    np.testing.assert_array_equal(geometry.bbox_mask, box)
    provenance = cast(dict[str, object], geometry.organ_source["lesion_mask"])
    assert provenance["voxel_count"] == provenance["voxels_inside_organ"] == 60


def test_resample_pads_an_odd_extent_to_even(tmp_path: Path) -> None:
    """An anisotropic RAS spacing that resamples to an odd extent is padded.

    1.3 mm spacing on the (24, 24, 20) seeded shape resamples the first two
    axes to round(24 * 1.3) = 31 (odd) and the third to round(20 * 1.3) = 26
    (already even), exercising ``_pad_to_even`` on exactly two axes for both
    the CT (edge padding) and the organ mask (zero padding), and checking
    they still land on one matching, fully even grid.
    """
    data = cases.ct_data()
    affine = np.diag((1.3, 1.3, 1.3, 1.0))
    ct_path = tmp_path / "anisotropic.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), ct_path)
    mask_path = tmp_path / "organ.nii.gz"
    nib.save(nib.Nifti1Image(cases.organ_block(), affine), mask_path)

    image, loaded = medmlx.load_reference_nifti(ct_path)
    assert image.shape == (32, 32, 26)
    assert all(size % 2 == 0 for size in image.shape)

    mask, _provenance = medmlx.prepare_organ_mask(reference=image, supplied_mask=mask_path)
    assert mask.shape == image.shape

    from scipy.ndimage import zoom

    unpadded_ct = zoom(data, (1.3, 1.3, 1.3), order=1)
    assert unpadded_ct.shape == (31, 31, 26)
    # Edge-padded CT repeats the last resampled slice on the padded axes.
    assert np.array_equal(loaded[:31, :31, :], unpadded_ct)
    assert np.array_equal(loaded[31], loaded[30])
    assert np.array_equal(loaded[:, 31], loaded[:, 30])
    # Zero-padded mask gets background, not a repeated label, on the new slice.
    assert not np.any(mask[31])
    assert not np.any(mask[:, 31])


@pytest.mark.parametrize("orientation", ["ras", "lps"])
@pytest.mark.parametrize("inference_mode", ["conditioned", "source"])
def test_resampling_keeps_world_coordinates_in_conditioned_mode(
    tmp_path: Path, tiny_networks: None, orientation: str, inference_mode: str
) -> None:
    """Preserve world coordinates during conditioned-mode resampling."""
    data = 12.0 + 2.0 * np.indices(cases.CT_SHAPE, dtype=np.float32)[0]
    labels = np.zeros(cases.CT_SHAPE, dtype=np.uint8)
    labels[5:15] = 1
    affine = np.diag((2.0, 1.0, 1.0, 1.0))
    affine[0, 3] = 12.0
    if orientation == "lps":
        data, labels = data[::-1, ::-1], labels[::-1, ::-1]
        affine[0, 0], affine[1, 1] = -2.0, -1.0
        affine[:2, 3] = (58.0, 23.0)
    source, mask, anatomy = (
        tmp_path / "ct.nii.gz",
        tmp_path / "organ.nii.gz",
        tmp_path / "anatomy.nii.gz",
    )
    for path, values in ((source, data), (mask, labels), (anatomy, labels * 5)):
        nib.save(nib.Nifti1Image(values, affine), path)
    options: cases.OptionsArguments = {**cases.OPTION_BASE, "inference_mode": inference_mode}
    geometry = medmlx.prepare_geometry(
        source=source,
        supplied_mask=mask,
        supplied_anatomy=anatomy,
        options=medmlx.LesionDiffusionOptions(**options),
    )
    if inference_mode == "conditioned":
        # HU encode world X: interpolation must agree with the returned affine.
        world_x = geometry.reference.affine[0, 3] + np.arange(48)
        np.testing.assert_allclose(geometry.source_data[:, 10, 10], np.minimum(world_x, 58.0))
        np.testing.assert_array_equal(
            geometry.organ_mask[:, 10, 10], (np.arange(48) >= 9) & (np.arange(48) < 29)
        )
    else:
        from scipy.ndimage import zoom

        ras_data = data if orientation == "ras" else data[::-1, ::-1]
        np.testing.assert_array_equal(geometry.source_data, zoom(ras_data, (2, 1, 1), order=1))
    assert geometry.anatomy_mask is not None
    np.testing.assert_array_equal(geometry.anatomy_mask == 5, geometry.organ_mask)


@pytest.mark.parametrize("name", cases.CENTER_CASES)
def test_upstream_box_placement(name: str) -> None:
    """Compare lesion-box placement with the source recording."""
    mode, shape, size, requested, seed = cases.CENTER_CASES[name]
    organ = cases.organ_shape(shape) > 0
    if requested is not None:
        if medmlx.bbox_slices(requested, size, organ.shape) is None:
            with pytest.raises(InvalidInputError):
                medmlx.resolve_bbox_center(
                    organ_mask=organ,
                    bbox_size=size,
                    placement=mode,
                    requested_center=requested,
                    seed=seed,
                )
        else:
            assert (
                medmlx.resolve_bbox_center(
                    organ_mask=organ,
                    bbox_size=size,
                    placement=mode,
                    requested_center=requested,
                    seed=seed,
                )
                == requested
            )
        return
    recorded = reference("runner_host")
    result = medmlx.resolve_bbox_center(
        organ_mask=organ,
        bbox_size=size,
        placement=mode,
        requested_center=None,
        seed=seed,
        organ="colon" if mode in {"within_lumen", "wall_thickening"} else "liver",
    )
    assert list(result) == recorded[f"center.{name}"].tolist()
    if f"box.{name}" in recorded:
        assert np.array_equal(medmlx.bbox_mask(organ.shape, result, size), recorded[f"box.{name}"])


def test_upstream_threshold_selection() -> None:
    """Compare lesion-component threshold selection with the source."""
    fixture = reference("selection")
    assert np.array_equal(
        medmlx.select_lesion_component(
            cast(NumericArray, fixture["samples"]), cast(BoolArray, fixture["box"])
        ),
        fixture["selected"],
    )
    assert np.array_equal(
        medmlx.select_lesion_component(
            cast(NumericArray, fixture["samples"]) * 0.1, cast(BoolArray, fixture["box"])
        ),
        fixture["empty"],
    )


@pytest.mark.parametrize("name", cases.CONVERSION_CASES)
def test_conversion_metadata_contract(name: str, tmp_path: Path) -> None:
    """Validate optional conversion metadata fields."""
    payload = cases.conversion_payload(name, medmlx.SOURCE_REF)
    (tmp_path / "conversion.json").write_text(
        payload if isinstance(payload, str) else json.dumps(payload)
    )
    if name == "valid":
        assert medmlx.load_conversion(tmp_path) == payload
    else:
        with pytest.raises(ValueError):
            medmlx.load_conversion(tmp_path)


TINY_LMNET = UnetConfig(32, 1, (1, 2), (1, 2), 16)
TINY_LINET = UnetConfig(32, 1, (1, 2), (1, 2), 16, skip_context=True)
TINY_LMNET_CROP = (8, 12, 12)
TINY_LINET_CROP = (16, 16, 16)
BBOX_MM = (14.0, 14.0, 10.0)


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

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "liver": 2,
        "lesion": 3,
        "irregular": 4,
        ",": 5,
        "hypodense": 6,
        "hyperdense": 7,
    }
    tokenizer = Tokenizer(WordLevel(vocab, "[UNK]"))
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
    monkeypatch.setattr(medmlx, "LMNET_CROP", TINY_LMNET_CROP)
    monkeypatch.setattr(medmlx, "LINET_CROP", TINY_LINET_CROP)


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


def test_run_admits_automatically_downloaded_weights(
    scan: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request the public model repository when weights are omitted."""
    folder = tmp_path / "downloaded"
    folder.mkdir()

    def download(model_id: str) -> Path:
        """Verify the requested repository and return a fixture folder."""
        assert model_id == "lesiondiffusion"
        return folder

    monkeypatch.setattr(medmlx, "weights_dir", download)
    with pytest.raises(AssetNotReadyError, match=r"model\.safetensors") as error:
        run(
            image_path=scan[0], organ_mask_path=scan[1], output_dir=tmp_path / "out",
            organ="liver", lesion_type="tumor", bbox_size_mm=BBOX_MM,
            placement="inside_parenchyma",
        )
    assert str(folder) in str(error.value)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"organ": "not_an_organ"}, "Unknown upstream organ"),
        ({"placement": "outside"}, "Unknown LesionDiffusion placement"),
        ({"bbox_size_mm": [10, 10]}, "must contain X, Y, Z"),
        ({"bbox_size_mm": "10,10,10"}, "must contain X, Y, Z"),
        ({"bbox_size_mm": [10, True, 10]}, "values must be numbers"),
        ({"diffusion_steps": 27}, "out-of-bounds indices"),
        ({"diffusion_steps": 0}, "positive integer"),
        ({"seed": -3}, "must not be negative"),
        ({"seed": 2**32}, "must be less than 2\\*\\*32"),
        ({"surface": 3}, "surface must be a string"),
        ({"inference_mode": "unknown"}, "must be conditioned or source"),
        ({"output_region": "crop"}, "output_region must be full or box"),
        ({"bbox_center_voxel": [1, 2]}, "must contain I, J, K"),
        ({"overwrite": "yes"}, "overwrite must be a boolean"),
        ({"bbox_size_mm": None}, "bbox_size_mm is required unless lesion_mask_path"),
        ({"lesion_mask_path": "lesion.nii.gz"}, "from lesion_mask_path; omit bbox_size_mm"),
        (
            {
                "lesion_mask_path": "lesion.nii.gz",
                "bbox_size_mm": None,
                "bbox_center_voxel": [12, 12, 10],
            },
            "omit bbox_center_voxel",
        ),
        (
            {
                "lesion_mask_path": "lesion.nii.gz",
                "bbox_size_mm": None,
                "anatomy_mask_path": "anatomy.nii.gz",
            },
            "omit anatomy_mask_path",
        ),
    ],
)
def test_invalid_options_raise_before_asset_access(
    overrides: dict[str, object], message: str, scan: tuple[Path, Path], tmp_path: Path
) -> None:
    """Reject invalid options before accessing model assets."""
    with pytest.raises(InvalidInputError, match=message):
        run(**_arguments(scan, tmp_path / "out", tmp_path / "absent", **overrides))
    assert not (tmp_path / "out").exists()


def test_invalid_inputs_raise_invalid_input(
    scan: tuple[Path, Path], tiny_weights: Path, tmp_path: Path
) -> None:
    """Reject invalid runner inputs with an input error."""
    ct, _organ = scan
    output = tmp_path / "out"
    with pytest.raises(InvalidInputError, match="Missing input CT NIfTI"):
        run(**_arguments(scan, output, tiny_weights, image_path=tmp_path / "absent.nii.gz"))
    with pytest.raises(InvalidInputError, match="Missing organ mask NIfTI"):
        run(**_arguments(scan, output, tiny_weights, organ_mask_path=tmp_path / "none.nii"))
    collision = tmp_path / "synthetic_ct.nii.gz"
    collision.write_bytes(ct.read_bytes())
    with pytest.raises(InvalidInputError, match="must not overwrite the input CT"):
        run(**_arguments(scan, tmp_path, tiny_weights, image_path=collision))
    # A pure orientation/spacing mismatch (e.g. LPS) is resampled to RAS 1 mm
    # rather than refused; see test_reference_loading_contract. A genuinely
    # oblique grid is refused.
    _, oblique_data, oblique_affine = cases.reference_input("oblique")
    oblique = tmp_path / "oblique.nii.gz"
    nib.save(nib.Nifti1Image(oblique_data, oblique_affine), oblique)
    with pytest.raises(InvalidInputError, match="axis-aligned grid"):
        run(**_arguments(scan, output, tiny_weights, image_path=oblique))
    labelled = tmp_path / "labels.nii.gz"
    nib.save(nib.Nifti1Image(cases.organ_block() * 5, nib.load(ct).affine), labelled)
    with pytest.raises(InvalidInputError, match="binary with label 1"):
        run(**_arguments(scan, output, tiny_weights, organ_mask_path=labelled))
    with pytest.raises(InvalidInputError, match="bbox extends outside the image"):
        run(**_arguments(scan, output, tiny_weights, bbox_size_mm=[30, 14, 10]))


@pytest.mark.parametrize("shape", [(128, 160, 128), (160, 160, 80)])
def test_geometry_rejects_inputs_smaller_than_either_network_crop(
    tmp_path: Path, shape: tuple[int, int, int]
) -> None:
    """Reject geometry smaller than either required network crop."""
    source, mask = tmp_path / "ct.nii.gz", tmp_path / "organ.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), source)
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.uint8), np.eye(4)), mask)
    with pytest.raises(InvalidInputError, match="crop"):
        medmlx.prepare_geometry(
            source=source,
            supplied_mask=mask,
            options=medmlx.LesionDiffusionOptions(**cases.OPTION_BASE),
        )
    # A supplied lesion skips LMNet, so only LINet's 128-voxel crop has to fit.
    lesion = tmp_path / "lesion.nii.gz"
    lesion_data = np.zeros(shape, dtype=np.uint8)
    lesion_data[60:64, 60:64, 40:44] = 1
    nib.save(nib.Nifti1Image(lesion_data, np.eye(4)), lesion)
    options: cases.OptionsArguments = {**cases.OPTION_BASE, "bbox_size_mm": None}

    def prepare_with_lesion():
        """Prepare the undersized case with a supplied lesion mask."""
        return medmlx.prepare_geometry(
            source=source,
            supplied_mask=mask,
            supplied_lesion=lesion,
            options=medmlx.LesionDiffusionOptions(**options),
        )

    if shape[2] < 128:
        with pytest.raises(InvalidInputError, match="LINet crop"):
            prepare_with_lesion()
    else:
        assert prepare_with_lesion().lesion_mask is not None


def test_upstream_reconstruction_is_written_without_box_blending(
    scan: tuple[Path, Path], tmp_path: Path, tiny_networks: None
) -> None:
    """Write source reconstruction without blending the lesion box."""
    fixture = reference("generation")
    options = medmlx.LesionDiffusionOptions(
        organ="liver",
        lesion_type="tumor",
        bbox_size_mm=BBOX_MM,
        placement="inside_parenchyma",
        seed=4,
    )
    geometry = medmlx.prepare_geometry(source=scan[0], supplied_mask=scan[1], options=options)
    outputs = medmlx.output_paths(tmp_path / "out")
    validation, compositing = medmlx.materialize_arrays(
        outputs=outputs,
        geometry=geometry,
        image_data=fixture["output"],
        lesion_data=cast(NumericArray, fixture["lesion"]) * 2,
    )
    assert compositing is None
    assert validation["lesion_voxel_count"] == int(np.count_nonzero(fixture["lesion"]))
    # Every output, including the organ mask, is written; no staging files remain.
    assert sorted(path.name for path in outputs["synthetic_image"].parent.iterdir()) == sorted(
        path.name for path in outputs.values()
    )
    image = nib.load(outputs["synthetic_image"])
    assert np.array_equal(np.asanyarray(image.dataobj), fixture["output"])
    outside = geometry.bbox_mask == 0
    assert np.any(fixture["output"][outside] != geometry.source_data[outside])
    assert np.array_equal(image.affine, geometry.reference.affine)
    assert np.array_equal(
        np.asanyarray(nib.load(outputs["target_mask"]).dataobj), fixture["lesion"]
    )


def test_box_output_region_feathers_generated_box_into_float32_source(
    scan: tuple[Path, Path], tmp_path: Path, tiny_networks: None
) -> None:
    """Blend generated box output into the float32 source grid."""
    fixture = reference("generation")
    generated = cast(NumericArray, fixture["output"])
    options = medmlx.LesionDiffusionOptions(
        organ="liver",
        lesion_type="tumor",
        bbox_size_mm=BBOX_MM,
        placement="inside_parenchyma",
        seed=4,
        output_region="box",
    )
    original = nib.load(scan[0])
    precision_ct = np.asarray(original.dataobj, dtype=np.float64) + 1e-7
    precision_path = tmp_path / "precision_ct.nii.gz"
    nib.save(nib.Nifti1Image(precision_ct, original.affine), precision_path)
    geometry = medmlx.prepare_geometry(
        source=precision_path, supplied_mask=scan[1], options=options
    )
    outputs = medmlx.output_paths(tmp_path / "out", "box")
    _, compositing = medmlx.materialize_arrays(
        outputs=outputs,
        geometry=geometry,
        image_data=generated,
        lesion_data=cast(NumericArray, fixture["lesion"]) * 2,
        output_region="box",
    )
    source = np.asarray(geometry.source_data, dtype=np.float32)
    localized = np.asanyarray(nib.load(outputs["synthetic_image"]).dataobj)
    raw = np.asanyarray(nib.load(outputs["raw_synthetic_image"]).dataobj)
    np.testing.assert_array_equal(raw, generated)
    box = np.argwhere(geometry.bbox_mask == 2)
    start, stop = box.min(axis=0), box.max(axis=0) + 1
    inside = tuple(slice(int(a), int(b)) for a, b in zip(start, stop, strict=True))
    np.testing.assert_array_equal(localized[inside], generated[inside])
    # 1 mm grid: an axis offset of 6 or more voxels from the box is beyond the margin.
    grid = np.indices(source.shape)
    offset = np.maximum(start[:, None, None, None] - grid, grid - (stop - 1)[:, None, None, None])
    far = (offset >= 6).any(axis=0)
    assert far.any() and np.any(generated[far] != source[far])
    np.testing.assert_array_equal(localized[far], source[far])
    assert np.any(localized[far].astype(np.float64) != geometry.source_data[far])
    # Two voxels before the box along I: weight 0.5 * (1 + cos(pi * 2 / 6)) = 0.75.
    i, j, k = int(start[0]) - 2, int(start[1]), int(start[2])
    assert i >= 0
    expected = np.float32(0.75) * generated[i, j, k] + np.float32(0.25) * source[i, j, k]
    np.testing.assert_allclose(localized[i, j, k], expected, rtol=0, atol=1e-3)
    assert compositing is not None
    assert compositing["kept_region"] == "requested_box_union_lesion_bounding_box"
    assert compositing["outside_affected_region"] == "original_ct_cast_float32"
    assert compositing["margin_mm"] == 6.0

    # A lesion voxel beyond the box and its margin widens the kept region, so the
    # written mask never labels unmodified input intensities.
    lesion = cast(NumericArray, fixture["lesion"]) * 2
    stray = tuple(int(value) for value in np.argwhere(far & (generated != source))[0])
    lesion[stray] = 2
    wider = medmlx.output_paths(tmp_path / "wider", "box")
    _, compositing = medmlx.materialize_arrays(
        outputs=wider,
        geometry=geometry,
        image_data=generated,
        lesion_data=lesion,
        output_region="box",
    )
    localized = np.asanyarray(nib.load(wider["synthetic_image"]).dataobj)
    written = np.asanyarray(nib.load(wider["target_mask"]).dataobj) == 1
    np.testing.assert_array_equal(localized[written], generated[written])
    assert compositing is not None
    assert compositing["requested_box"] == {
        "start_voxel": start.tolist(),
        "stop_voxel_exclusive": stop.tolist(),
    }


@pytest.mark.parametrize("orientation", ["ras", "lps"])
def test_full_anatomy_reaches_mask_generator_on_the_source_grid(
    orientation: str,
    scan: tuple[Path, Path],
    tiny_weights: Path,
    tiny_networks: None,
    tmp_path: Path,
    mx: Mlx,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass full anatomy context to the mask generator on the source grid."""
    from lesiondiffusion_mlx.mlx_sampling import DdimSampler

    image = nib.load(scan[0])
    organ = cast(NumericArray, np.asanyarray(nib.load(scan[1]).dataobj))
    anatomy = np.where(organ > 0, 5, 104).astype(np.uint8)
    anatomy[0] = 0
    path = tmp_path / "anatomy.nii.gz"
    anatomy_affine = image.affine.copy()
    anatomy_data = anatomy
    if orientation == "lps":
        anatomy_data = anatomy[::-1, ::-1, :]
        anatomy_affine[0, 0] = -1
        anatomy_affine[1, 1] = -1
        anatomy_affine[:2, 3] += np.array(anatomy.shape[:2]) - 1
    nib.save(nib.Nifti1Image(anatomy_data, anatomy_affine), path)
    observed: list[NumericArray] = []

    class ObserveSampler(DdimSampler):
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
            """Record the anatomy-conditioned mask sampling inputs."""
            if mask is None:
                observed.append(cast(NumericArray, np.array(conditioning_image)))
            return super().__call__(
                denoiser,
                condition,
                conditioning_image,
                initial_noise=initial_noise,
                step_noise=step_noise,
                mask=mask,
            )

    monkeypatch.setattr(medmlx, "DdimSampler", ObserveSampler)
    result = run(
        **_arguments(
            scan,
            tmp_path / "out",
            tiny_weights,
            anatomy_mask_path=path,
            bbox_center_voxel=[12, 12, 10],
            bbox_size_mm=[6, 6, 4],
        )
    )
    # Source NCDHW is LAS, with Z/Y/X axes. The even box has centre 9/11/11,
    # so the released 8/12/12 mask crop starts at 5/5/5.
    segmentation = anatomy[::-1, :, :].transpose(2, 1, 0)[5:13, 5:17, 5:17] / 104.0
    target = organ[::-1, :, :].transpose(2, 1, 0)[5:13, 5:17, 5:17]
    box = np.zeros((8, 12, 12), dtype=np.float32)
    box[3:7, 4:10, 4:10] = 1
    expected = np.stack((segmentation, target, box))[None].astype(np.float32)
    assert observed
    for actual in observed:
        np.testing.assert_array_equal(actual, expected)
    assert result["details"]["generation_spec"]["native_runtime"]["lmnet_anatomy_context"] == (
        "supplied_source_labels"
    )


@pytest.mark.parametrize("invalid", ["unknown_label", "target_disagrees", "different_grid"])
def test_anatomy_context_refuses_ambiguous_labels_or_geometry(
    invalid: str,
    scan: tuple[Path, Path],
    tiny_weights: Path,
    tmp_path: Path,
) -> None:
    """Reject ambiguous anatomy labels or incompatible geometry."""
    image = nib.load(scan[0])
    anatomy = np.asanyarray(nib.load(scan[1]).dataobj).astype(np.uint8) * 5
    affine = image.affine.copy()
    if invalid == "unknown_label":
        anatomy[0, 0, 0] = 105
        message = "upstream TotalSegmentator v1"
    elif invalid == "target_disagrees":
        anatomy[0, 0, 0] = 5
        message = "selected-organ label must match"
    else:
        affine[0, 3] += 1
        message = "must share the input CT grid"
    path = tmp_path / "anatomy.nii.gz"
    nib.save(nib.Nifti1Image(anatomy, affine), path)
    with pytest.raises(InvalidInputError, match=message):
        run(**_arguments(scan, tmp_path / "out", tiny_weights, anatomy_mask_path=path))


def test_supplied_lesion_mask_skips_lmnet_and_is_inpainted_with_phenotype_text(
    scan: tuple[Path, Path],
    conditioned_weights: Path,
    tiny_networks: None,
    tmp_path: Path,
    mx: Mlx,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip mask generation for a supplied lesion and retain phenotype conditioning."""
    from lesiondiffusion_mlx.mlx_sampling import DdimSampler

    lesion = cases.lesion_block()
    path = tmp_path / "lesion.nii.gz"
    nib.save(nib.Nifti1Image(lesion, nib.load(scan[0]).affine), path)
    observed: list[tuple[bool, NumericArray | None]] = []

    class ObserveSampler(DdimSampler):
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
            """Record the supplied-mask image-sampling inputs."""
            latent_mask = None if mask is None else cast(NumericArray, np.array(mask))
            observed.append((condition is not None, latent_mask))
            return super().__call__(
                denoiser,
                condition,
                conditioning_image,
                initial_noise=initial_noise,
                step_noise=step_noise,
                mask=mask,
            )

    monkeypatch.setattr(medmlx, "DdimSampler", ObserveSampler)
    result = run(
        **_arguments(
            scan,
            tmp_path / "out",
            conditioned_weights,
            lesion_mask_path=path,
            bbox_size_mm=None,
            inference_mode="conditioned",
            output_region="box",
            density="hyperdense lesion",
        )
    )
    # One sampler call, from LINet: LMNet's unmasked sampling never ran.
    assert len(observed) == 1
    has_text, latent_mask = observed[0]
    assert has_text and latent_mask is not None
    # Source Z/Y/X with RAS I reversed (X = 23 - I). Lesion voxels at even source
    # indices, Z 8-10, Y 10-12, X 10-14, sit at latent Z 4-5, Y 5-6, X 5-7. The 16-voxel
    # crop around the lesion starts at source 0/3/4, so the latent window starts at 0/1/2.
    expected = np.zeros((1, 1, 8, 8, 8), dtype=np.float32)
    expected[..., 4:6, 4:6, 3:6] = 1
    np.testing.assert_array_equal(latent_mask, expected)

    written = np.asanyarray(nib.load(result["outputs"]["target_mask"]).dataobj)
    np.testing.assert_array_equal(written, lesion)
    details = result["details"]
    assert details["weights_revision"] is None
    assert details["stages"]["lmnet"] == {"status": "skipped", "reason": "supplied_lesion_mask"}
    spec = details["generation_spec"]
    runtime = spec["native_runtime"]
    assert runtime["lesion_mask_source"] == "supplied"
    assert runtime["lmnet_attempts"] == 0
    assert runtime["linet_conditioning"]["density"] == "hyperdense lesion,"
    assert "lesion_mask" in spec
    assert cast(dict[str, object], spec["lesion_mask"])["mask"] == path.resolve().as_posix()
    # The derived box [8:14, 10:14, 6:12] is the region kept generated.
    assert spec["bbox_center_voxel"] == [11, 12, 9]
    assert spec["bbox_size_voxel"] == [6, 4, 6]
    assert "compositing" in spec
    assert spec["compositing"]["kept_region"] == "lesion_mask_box"
    assert spec["compositing"]["bounding_box"] == {
        "start_voxel": [8, 10, 6],
        "stop_voxel_exclusive": [14, 14, 12],
    }


@pytest.mark.parametrize("automatic", [True, False])
def test_native_folder_runs_without_conversion_receipts(
    automatic: bool,
    scan: tuple[Path, Path],
    tiny_weights: Path,
    tiny_networks: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run with automatic or explicit native weights without a conversion receipt."""
    folder = tmp_path / "bundle"
    folder.mkdir()
    shutil.copyfile(tiny_weights / "model.safetensors", folder / "checkpoint-blob")
    (folder / "model.safetensors").symlink_to("checkpoint-blob")
    shutil.copyfile(tiny_weights / "tokenizer.json", folder / "tokenizer.json")
    (folder / "conversion.json").write_text("malformed receipt is not a runtime input")
    revision = "d" * 40
    (folder / ".medmlx-revision").write_text(revision)
    requested: list[str] = []

    def download(model_id: str) -> Path:
        """Record the requested public model repository."""
        requested.append(model_id)
        return folder

    monkeypatch.setattr(medmlx, "weights_dir", download)
    expected_folder = folder
    if automatic:
        arguments = _arguments(scan, tmp_path / "out", folder)
        arguments.pop("weights_path")
    else:
        expected_folder = tmp_path / "linked-folder"
        expected_folder.symlink_to(folder, target_is_directory=True)
        arguments = _arguments(scan, tmp_path / "out", expected_folder)
    result = run(**arguments)
    assert requested == (["lesiondiffusion"] if automatic else [])
    assert result["details"]["weights_revision"] == revision
    assert result["details"]["weights_path"] == str(expected_folder)
    assert Path(result["outputs"]["synthetic_image"]).is_file()


def test_missing_or_mismatched_weights_raise_asset_not_ready(
    scan: tuple[Path, Path], tiny_weights: Path, tmp_path: Path
) -> None:
    """Report missing or mismatched weights as asset errors."""
    output = tmp_path / "out"
    with pytest.raises(AssetNotReadyError, match="weights directory is missing") as error:
        run(**_arguments(scan, output, tmp_path / "absent"))
    assert error.value.hint == medmlx.WEIGHTS_HINT

    def copy(name: str) -> Path:
        """Copy fixture weights into a named temporary directory."""
        directory = tmp_path / name
        directory.mkdir()
        for path in tiny_weights.iterdir():
            directory.joinpath(path.name).write_bytes(path.read_bytes())
        return directory

    weights = copy("missing_safetensors")
    (weights / "model.safetensors").unlink()
    with pytest.raises(AssetNotReadyError, match=r"missing .*model\.safetensors"):
        run(**_arguments(scan, output, weights))
    for stage in ("vq", "linet_text"):
        weights = copy(f"no_{stage}")
        packed = load_safe(str(weights / "model.safetensors"))
        save_safe(
            {key: value for key, value in packed.items() if not key.startswith(stage + ".")},
            str(weights / "model.safetensors"),
        )
        with pytest.raises(AssetNotReadyError, match=f"missing stage {stage}"):
            run(**_arguments(scan, output, weights, inference_mode="conditioned"))
        if stage == "linet_text":
            assert medmlx.require_weights(weights, inference_mode="source") == weights
    weights = copy("corrupt_safetensors")
    (weights / "model.safetensors").write_bytes(b"not a safetensors header")
    with pytest.raises(AssetNotReadyError, match="not readable"):
        run(**_arguments(scan, output, weights))


@pytest.fixture
def upstream_inpainting(mx: Mlx, monkeypatch: pytest.MonkeyPatch) -> type[SamplerFixture]:
    """Fix discontinuous learned stages while exercising the runner's wiring."""
    fixture = reference("generation")
    from lesiondiffusion_mlx.mlx_sampling import DdimSampler

    def recorded_encode(model: VqAutoencoder, array: Array, mx: Mlx) -> Array:
        # Normalization is rounded to half at the upstream VQ input boundary.
        # A float32 affine rounding difference may straddle a half midpoint.
        """Compare the fixture image passed into upstream encoding."""
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
    return FixedNoiseSampler


def test_run_matches_upstream_end_to_end(
    scan: tuple[Path, Path],
    tiny_weights: Path,
    tiny_networks: None,
    tmp_path: Path,
    mx: Mlx,
    upstream_inpainting: type[SamplerFixture],
) -> None:
    """Compare the native runner with the recorded upstream pipeline."""
    fixture = reference("generation")
    result = run(**_arguments(scan, tmp_path / "out", tiny_weights))
    images = {role: nib.load(path) for role, path in result["outputs"].items()}
    assert set(images) == {"synthetic_image", "target_mask", "organ_mask"}
    source = nib.load(scan[0])
    for image in images.values():
        assert image.shape == source.shape
        assert np.array_equal(image.affine, source.affine)
    actual = cast(NumericArray, np.asanyarray(images["synthetic_image"].dataobj))
    # Fixed upstream quantized input isolates native VQ decoding and HU writing.
    # The decoder's 0.002 normalized absolute bound maps to at most 8 HU.
    # Native generation/repeatability is exercised separately below: nearest-code
    # switches make a uniform HU bound invalid for independently computed latents.
    np.testing.assert_allclose(actual, cast(NumericArray, fixture["output"]), atol=8, rtol=0)
    assert np.array_equal(np.asanyarray(images["target_mask"].dataobj), fixture["lesion"])
    assert np.array_equal(np.asanyarray(images["organ_mask"].dataobj), cases.organ_block())
    details = result["details"]
    assert (
        details["generation_spec"]["native_runtime"]["linet_conditioning"]["lesion location"]
        == "liver"
    )
    assert "size" not in details["generation_spec"]["native_runtime"]["linet_conditioning"]
    assert details["generation_spec"]["native_runtime"]["lesion_mask_source"] == "lmnet"
    assert details["generation_spec"]["bbox_center_voxel"] == fixture["center"].tolist()
    assert (
        json.loads(fixture["lm_attributes"].item())
        == details["generation_spec"]["native_runtime"]["lmnet_conditioning"]
    )
    json.dumps(details)
    with pytest.raises(InvalidInputError, match="overwrite=true"):
        run(**_arguments(scan, tmp_path / "out", tiny_weights))
    upstream_inpainting.attempts = 0
    again = run(**_arguments(scan, tmp_path / "out", tiny_weights, overwrite=True))
    for role, path in again["outputs"].items():
        assert np.array_equal(
            np.asanyarray(nib.load(path).dataobj), np.asanyarray(images[role].dataobj)
        )


def test_run_writes_odd_ras_1mm_input_on_its_own_grid(
    tiny_weights: Path, tiny_networks: None, tmp_path: Path, mx: Mlx
) -> None:
    """Even padding for the VQ stages must not leak into the written outputs."""
    shape = (23, 24, 21)
    affine = cases.ras_affine((30.5, -42.0, 117.0))
    ct, organ = tmp_path / "ct.nii.gz", tmp_path / "liver.nii.gz"
    organ_data = cases.organ_block(shape)
    nib.save(nib.Nifti1Image(cases.ct_data(shape), affine), ct)
    nib.save(nib.Nifti1Image(organ_data, affine), organ)
    result = run(**_arguments((ct, organ), tmp_path / "out", tiny_weights))
    for path in result["outputs"].values():
        image = nib.load(path)
        assert image.shape == shape
        assert np.array_equal(image.affine, affine)
    written = np.asanyarray(nib.load(result["outputs"]["organ_mask"]).dataobj)
    assert np.array_equal(written, organ_data)
    assert np.any(np.asanyarray(nib.load(result["outputs"]["target_mask"]).dataobj))
    assert result["details"]["output_shape_by_role"]["synthetic_image"] == list(shape)


def test_full_run_has_no_forbidden_runtime_imports(
    scan: tuple[Path, Path], tiny_weights: Path, tmp_path: Path
) -> None:
    """Runs the real ``run()`` call path end to end (tiny architectures, seeded
    weights, as ``tiny_networks``/``tiny_weights`` do for the in-process tests
    above) inside a fresh process that blocks torch/monai/transformers from the
    first import, then checks every module the call actually pulled in.
    """
    output = tmp_path / "out"
    arguments = _arguments(scan, output, tiny_weights)
    kwargs_json = json.dumps(
        {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in arguments.items()
        }
    )
    program = f"""
import importlib.abc
import json
import sys

class BlockForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {{"torch", "monai", "transformers"}}:
            raise AssertionError(f"Forbidden runtime import: {{fullname}}")

sys.meta_path.insert(0, BlockForbidden())

from lesiondiffusion_mlx import medmlx
from lesiondiffusion_mlx.mlx_model import UnetConfig

# Tiny architectures/crops, matching the tests/test_medmlx.py tiny_networks
# fixture's monkeypatch, so this seeded run completes quickly.
medmlx.LMNET_CONFIG = UnetConfig(32, 1, (1, 2), (1, 2), 16)
medmlx.LINET_CONFIG = UnetConfig(32, 1, (1, 2), (1, 2), 16, skip_context=True)
medmlx.LMNET_CROP = (8, 12, 12)
medmlx.LINET_CROP = (16, 16, 16)

result = medmlx.run(**json.loads({kwargs_json!r}))
_BLOCKED = {{"torch", "monai", "transformers"}}
forbidden = sorted(name for name in sys.modules if name.split(".")[0] in _BLOCKED)
print(json.dumps({{"forbidden": forbidden, "image": result["outputs"]["synthetic_image"]}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["forbidden"] == []
    assert Path(payload["image"]).is_file()


def test_native_seed_is_repeatable(
    scan: tuple[Path, Path], conditioned_weights: Path, tiny_networks: None, tmp_path: Path, mx: Mlx
) -> None:
    """Verify reproducibility for the same native generation seed."""
    first = run(
        **_arguments(scan, tmp_path / "first", conditioned_weights, inference_mode="conditioned")
    )
    again = run(
        **_arguments(scan, tmp_path / "again", conditioned_weights, inference_mode="conditioned")
    )
    for role, path in first["outputs"].items():
        actual = cast(NumericArray, np.asanyarray(nib.load(path).dataobj))
        assert np.isfinite(actual).all()
        if role == "synthetic_image":
            assert actual.dtype == np.float32
            assert actual.min() >= -1000 and actual.max() <= 1000
        else:
            assert actual.dtype == np.uint8
            assert set(np.unique(actual)) <= {0, 1}
        assert np.array_equal(
            actual,
            np.asanyarray(nib.load(again["outputs"][role]).dataobj),
        )


def test_density_text_changes_native_image_on_the_same_mask(
    scan: tuple[Path, Path],
    conditioned_weights: Path,
    tiny_networks: None,
    tmp_path: Path,
    mx: Mlx,
) -> None:
    """Check that phenotype density text affects image generation on a fixed mask."""
    results = [
        run(
            **_arguments(
                scan,
                tmp_path / density,
                conditioned_weights,
                inference_mode="conditioned",
                density=f"{density} lesion",
            )
        )
        for density in ("hypodense", "hyperdense")
    ]
    first, second = results
    np.testing.assert_array_equal(
        np.asanyarray(nib.load(first["outputs"]["target_mask"]).dataobj),
        np.asanyarray(nib.load(second["outputs"]["target_mask"]).dataobj),
    )
    assert not np.array_equal(
        np.asanyarray(nib.load(first["outputs"]["synthetic_image"]).dataobj),
        np.asanyarray(nib.load(second["outputs"]["synthetic_image"]).dataobj),
    )
    for result in results:
        runtime = result["details"]["generation_spec"]["native_runtime"]
        assert runtime["linet_skip_context"] is False
        assert runtime["convolution_arithmetic"] == "mlx_native"
        assert "CT value" not in runtime["linet_conditioning"]


def test_lmnet_qc_rejection_is_a_model_failure(
    scan: tuple[Path, Path],
    tiny_weights: Path,
    tiny_networks: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mx: Mlx,
) -> None:
    """Report rejection of an empty generated lesion as model failure."""
    def empty_lesion(samples: NumericArray, reference: BoolArray) -> BoolArray:
        """Return an empty lesion to exercise quality rejection."""
        return np.zeros_like(reference)

    output = tmp_path / "out"
    with monkeypatch.context() as patch:
        patch.setattr(medmlx, "select_lesion_component", empty_lesion)
        with pytest.raises(ModelExecutionError, match="LMNet QC rejected this seed"):
            run(**_arguments(scan, output, tiny_weights))
    # A rejected seed writes nothing, so another seed can use the same directory.
    assert not output.exists() or not any(output.iterdir())
    result = run(**_arguments(scan, output, tiny_weights, seed=5))
    assert all(Path(path).is_file() for path in result["outputs"].values())
