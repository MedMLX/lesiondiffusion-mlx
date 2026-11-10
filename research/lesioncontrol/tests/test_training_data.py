"""Protect fractional CT attenuation at the real-tumor preparation boundary."""

import json
from pathlib import Path
from typing import Protocol, cast

import nibabel as nib
import numpy as np
import pytest
from numpy.typing import NDArray

from lesioncontrol.training_data import prepare_patient


class _Writable(Protocol):
    def to_filename(self, filename: str) -> None: ...


def _save(path: Path, values: NDArray[np.int16] | NDArray[np.uint8]) -> None:
    cast(_Writable, nib.Nifti1Image(values, np.eye(4))).to_filename(str(path))


def test_real_tumor_preparation_preserves_fractional_interpolated_hu(tmp_path: Path) -> None:
    """An analytic linear HU field stays linear after a half-voxel grid shift."""
    patient = "HCC_TEST"
    source = tmp_path / "native" / patient
    source.mkdir(parents=True)
    shape = (128, 128, 128)
    ct = np.broadcast_to(80 + np.arange(128, dtype=np.int16)[:, None, None], shape)
    tumor = np.zeros(shape, dtype=np.uint8)
    tumor[50:67, 50:67, 50:67] = 1
    _save(source / "source_contrast.nii.gz", ct)
    _save(source / "tumor_mask.nii.gz", tumor)
    _save(source / "liver_mask.nii.gz", np.ones(shape, dtype=np.uint8))
    catalog: dict[str, object] = {
        "source_contrast": {"series_uid": "test-source", "image_count": 128},
        "segmentation": {"series_uid": "test-seg"},
        "tumor_voxels": 17**3,
    }
    outline = {
        "lesion_id": "HCC_TEST_L01",
        "rank": "1",
        "voxel_count": str(17**3),
        "centroid_r_mm": "58",
        "centroid_a_mm": "58",
        "centroid_s_mm": "58",
    }
    cases, failures = prepare_patient(
        patient, "train", catalog, [outline], tmp_path / "native", tmp_path / "prepared", []
    )
    assert not failures
    image = nib.Nifti1Image.from_filename(cast(str, cases[0]["real_ct"]))
    mask = nib.Nifti1Image.from_filename(cast(str, cases[0]["real_mask"]))
    selected = np.asarray(mask.dataobj) > 0
    coordinates = np.argwhere(selected)
    assert image.affine is not None
    world_x = coordinates[:, 0] + image.affine[0, 3]
    expected = 80 + world_x
    np.testing.assert_allclose(np.asarray(image.dataobj)[selected], expected, rtol=0, atol=1e-5)


def test_named_mass_acquisition_proof_admits_real_preparation_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    """A validated selected acquisition may use the dataset's explicit Mass name."""
    patient = "HCC_TEST"
    source = tmp_path / "native" / patient
    source.mkdir(parents=True)
    shape = (128, 128, 128)
    tumor = np.zeros(shape, dtype=np.uint8)
    tumor[50:67, 50:67, 50:67] = 1
    _save(source / "source_contrast.nii.gz", np.full(shape, 80, dtype=np.int16))
    _save(source / "tumor_mask.nii.gz", tumor)
    _save(source / "liver_mask.nii.gz", np.ones(shape, dtype=np.uint8))
    catalog: dict[str, object] = {
        "source_contrast": {"series_uid": "test-source", "image_count": 256},
        "segmentation": {"series_uid": "test-seg"},
        "tumor_voxels": 17**3,
    }
    outline = {
        "lesion_id": "HCC_TEST_L01",
        "rank": "1",
        "voxel_count": str(17**3),
        "centroid_r_mm": "58",
        "centroid_a_mm": "58",
        "centroid_s_mm": "58",
    }
    proof: dict[str, object] = {
        "case": patient,
        "source_series_uid": "test-source",
        "selected_ct_shape": list(shape),
        "cached_ct_array_equal": True,
        "cached_ct_affine_equal": True,
        "cached_mass_array_equal": True,
    }
    path = tmp_path / "source_acquisition_qc.json"
    path.write_text(json.dumps(proof))
    cases, failures = prepare_patient(
        patient,
        "heldout",
        catalog,
        [outline],
        tmp_path / "native",
        tmp_path / "prepared",
        [path],
    )
    assert len(cases) == 1 and not failures
    for mass, tumor_match in ((True, False), (False, True)):
        proof.update(cached_mass_array_equal=mass, cached_tumor_array_equal=tumor_match)
        path.write_text(json.dumps(proof))
        with pytest.raises(ValueError):
            prepare_patient(
                patient,
                "heldout",
                catalog,
                [outline],
                tmp_path / "native",
                tmp_path / "must-not-prepare",
                [path],
            )
