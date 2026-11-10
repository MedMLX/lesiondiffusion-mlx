"""Scientific intensity-space and clean-context contracts for reference control."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Protocol, cast

import nibabel as nib
import numpy as np
import pytest
from numpy.typing import NDArray

from lesioncontrol.profile import (
    ReferenceProfile,
    build_profile,
    compare_features,
    default_settings,
    load_host,
    load_profile,
    save_profile,
)


class _WritableImage(Protocol):
    def to_filename(self, filename: str) -> None: ...


def _save(path: Path, data: NDArray[np.float32]) -> Path:
    image = nib.Nifti1Image(data, np.eye(4))
    cast(_WritableImage, image).to_filename(str(path))
    return path


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Use integer HU so a uniform scan offset has no rounding ambiguity."""
    shape = (128, 128, 128)
    image = np.full(shape, 80, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.float32)
    mask[56:72, 56:72, 56:72] = 1
    x, y, z = np.indices((16, 16, 16))
    image[56:72, 56:72, 56:72] = 40 + 2 * ((3 * x + 5 * y + 7 * z) % 31)
    return (
        _save(tmp_path / "image.nii.gz", image),
        _save(tmp_path / "shifted.nii.gz", image + np.float32(32)),
        _save(tmp_path / "mask.nii.gz", mask),
        _save(tmp_path / "liver.nii.gz", np.ones(shape, dtype=np.float32)),
    )


def test_contrast_profile_removes_scan_offset_before_discretization(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """A scan-wide HU shift affects absolute targets, not liver-relative targets."""
    image, shifted, mask, liver = prepared
    contrast = build_profile(image, mask, liver, target_space="liver_contrast")
    shifted_contrast = build_profile(shifted, mask, liver, target_space="liver_contrast")
    absolute = build_profile(image, mask, liver, target_space="absolute_hu")
    shifted_absolute = build_profile(shifted, mask, liver, target_space="absolute_hu")
    assert contrast.features == shifted_contrast.features
    assert absolute.target_mean_hu - contrast.target_mean_hu == pytest.approx(80, abs=1e-5)
    assert shifted_absolute.target_mean_hu - absolute.target_mean_hu == pytest.approx(32, abs=1e-5)


def test_known_tumor_anywhere_in_encoded_host_is_rejected(
    prepared: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    """VQ sees the full crop, including tumors far outside the placement mask."""
    image, _, mask, liver = prepared
    known = np.zeros((128, 128, 128), dtype=np.float32)
    known_path = _save(tmp_path / "known.nii.gz", known)
    host = load_host(image, liver, mask, known_tumor_mask=known_path)
    assert host.ring_mean_hu == 80
    known[4, 4, 4] = 1
    _save(known_path, known)
    with pytest.raises(ValueError, match="full encoded host"):
        load_host(image, liver, mask, known_tumor_mask=known_path)
    with pytest.raises(ValueError, match="declare this host clean"):
        load_host(image, liver, mask)
    with pytest.raises(ValueError, match="declared_clean must be a boolean"):
        load_host(image, liver, mask, declared_clean=cast(bool, "false"))


def test_profile_edits_and_family_balanced_distance(tmp_path: Path) -> None:
    """Persisted edits must control the numeric target and preserve equal family weight."""
    settings = default_settings()
    features = {f"original_{name}": 1.0 for name in settings.features}
    profile = ReferenceProfile(features, "absolute_hu", settings, {})
    generated = dict(features)
    generated["original_firstorder_Mean"] = 3
    generated["original_glcm_Contrast"] = 3
    result = compare_features(profile, generated)
    # Each changed feature contributes |3-1|/(|3|+|1|)=1/2.
    assert result["score"] == pytest.approx((0.5 / 21 + 0.5 / 26) / 6, abs=1e-12)
    edited = replace(profile, features=generated)
    path = tmp_path / "profile.json"
    save_profile(edited, path)
    restored = load_profile(path)
    assert restored.target_mean_hu == 3
    assert restored.features == generated


@pytest.mark.parametrize(
    "metadata",
    [
        {"reference_qc_status": "excluded_appearance_reference"},
        {"qc_record": {"reference_qc_status": "excluded_appearance_reference"}},
    ],
)
def test_profile_quarantine_survives_saved_profile_edits(
    tmp_path: Path, metadata: dict[str, object]
) -> None:
    """Loading edited target features must retain an explicit reference exclusion."""
    settings = default_settings()
    features = {f"original_{name}": 1.0 for name in settings.features}
    profile = ReferenceProfile(features, "absolute_hu", settings, {})
    path = tmp_path / "profile.json"
    save_profile(profile, path)
    payload = cast(dict[str, object], json.loads(path.read_text()))
    payload["metadata"] = metadata
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="quarantined"):
        load_profile(path)
