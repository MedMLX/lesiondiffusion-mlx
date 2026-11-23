"""Scientific intensity-space and clean-context contracts for reference control."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Protocol, cast

import nibabel as nib
import numpy as np
import pytest
from hcc_fixture import ct_cube
from numpy.typing import NDArray
from scipy import ndimage

from lesioncontrol.placement import OVERHANG, rescaled, translated_placement
from lesioncontrol.profile import (
    ReferenceProfile,
    build_profile,
    compare_features,
    default_settings,
    load_host,
    load_profile,
    save_profile,
)
from lesioncontrol.spatial import (
    SpatialTarget,
    calibrate_appearance,
    liver_surface_fraction,
    solidity,
)


class _WritableImage(Protocol):
    def to_filename(self, filename: str) -> None: ...


def _save(path: Path, data: NDArray[np.float32]) -> Path:
    image = nib.Nifti1Image(data, np.eye(4))
    cast(_WritableImage, image).to_filename(str(path))
    return path


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[Path, Path, Path, Path, float]:
    """Use integer HU so a uniform scan offset has no rounding ambiguity."""
    shape = (128, 128, 128)
    image = ct_cube()
    mask = np.zeros(shape, dtype=np.float32)
    mask[56:72, 56:72, 56:72] = 1
    distance = ndimage.distance_transform_edt(mask == 0)
    ring = (distance >= 3) & (distance <= 10)
    ring_mean = float(np.mean(image[ring], dtype=np.float64))
    return (
        _save(tmp_path / "image.nii.gz", image),
        _save(tmp_path / "shifted.nii.gz", image + np.float32(32)),
        _save(tmp_path / "mask.nii.gz", mask),
        _save(tmp_path / "liver.nii.gz", np.ones(shape, dtype=np.float32)),
        ring_mean,
    )


def test_contrast_profile_removes_scan_offset_before_discretization(
    prepared: tuple[Path, Path, Path, Path, float],
) -> None:
    """A scan-wide HU shift affects absolute targets, not liver-relative targets."""
    image, shifted, mask, liver, ring_mean = prepared
    contrast = build_profile(image, mask, liver, target_space="liver_contrast")
    shifted_contrast = build_profile(shifted, mask, liver, target_space="liver_contrast")
    absolute = build_profile(image, mask, liver, target_space="absolute_hu")
    shifted_absolute = build_profile(shifted, mask, liver, target_space="absolute_hu")
    assert contrast.features == shifted_contrast.features
    assert absolute.target_mean_hu - contrast.target_mean_hu == pytest.approx(ring_mean, abs=1e-5)
    assert shifted_absolute.target_mean_hu - absolute.target_mean_hu == pytest.approx(32, abs=1e-5)


def test_spatial_profile_restores_hu_and_preserves_host_ring(
    prepared: tuple[Path, Path, Path, Path, float], tmp_path: Path
) -> None:
    """Recover a known depth/scale distortion from scalar targets on a shifted host."""
    image, shifted, mask, liver, _ = prepared
    profile = build_profile(image, mask, liver, target_space="liver_contrast", spatial=True)
    archive = tmp_path / "profile.json"
    save_profile(profile, archive)
    target = SpatialTarget.from_record(load_profile(archive).metadata["spatial_v1"])
    host = load_host(shifted, liver, mask, declared_clean=True)
    source = np.asarray(nib.Nifti1Image.from_filename(str(image)).dataobj, dtype=np.float32)
    depth = ndimage.distance_transform_edt(host.placement_mask)
    draw = (source * 1.5 - 18.0 + 2.0 * depth).astype(np.float32)
    calibrated = calibrate_appearance(
        draw,
        host.image,
        host.placement_mask,
        host.liver_mask,
        target,
        profile.target_mean_hu + host.ring_mean_hu,
    )
    inside = host.placement_mask
    np.testing.assert_allclose(calibrated[inside], source[inside] + 32, rtol=0, atol=2e-4)
    far = ndimage.distance_transform_edt(~inside) >= 3
    np.testing.assert_array_equal(calibrated[far], host.image[far])


def test_air_at_the_liver_surface_does_not_set_reference_texture() -> None:
    """Contour voxels within 2 mm of the liver surface are partial volume, not tumor."""
    image = ct_cube()
    mask = np.zeros(image.shape, dtype=np.bool_)
    mask[56:72, 56:72, 56:72] = True
    liver = np.ones_like(mask)
    liver[:56] = False
    air = image.copy()
    air[56:58][mask[56:58]] = -700
    clean, dirty = (SpatialTarget.measure(x, mask, liver) for x in (image, air))
    assert (dirty.slope, dirty.coarse_sd, dirty.fine_sd, dirty.fine_to_liver) == (
        clean.slope,
        clean.coarse_sd,
        clean.fine_sd,
        clean.fine_to_liver,
    )
    tumor = mask.copy()
    tumor[56:58] = False
    expected = float(air[tumor].mean(dtype=np.float64) - air[mask].mean(dtype=np.float64))
    assert dirty.mean_shift_hu == pytest.approx(expected, abs=1e-9)
    assert expected > 50


def test_host_noise_gives_the_lesion_the_reference_ratio_to_liver() -> None:
    """A lesion placed in a noisier scan is as grainy against its liver as the reference."""
    reference = ct_cube()
    mask = np.zeros(reference.shape, dtype=np.bool_)
    mask[56:72, 56:72, 56:72] = True
    liver = np.ones_like(mask)
    target = SpatialTarget.measure(reference, mask, liver)
    assert target.fine_to_liver is not None
    level = float(reference.mean(dtype=np.float64))
    # Doubling every deviation doubles the scan noise without changing anatomy.
    noisy = (level + 2 * (reference - level)).astype(np.float32)
    draw = reference[::-1, ::-1, ::-1].copy()
    for host in (reference, noisy):
        output = calibrate_appearance(draw, host, mask, liver, target, 60.0, host_noise=True)
        measured = SpatialTarget.measure(output, mask, liver)
        # Splitting the composed lesion again moves about 5% of amplitude between scales.
        assert measured.fine_to_liver == pytest.approx(target.fine_to_liver, rel=0.1)
        assert float(output[mask].mean(dtype=np.float64)) == pytest.approx(60.0, abs=1e-3)
    amplitude = calibrate_appearance(draw, noisy, mask, liver, target, 60.0)
    smooth = SpatialTarget.measure(amplitude, mask, liver).fine_to_liver
    assert smooth is not None and smooth < 0.65 * target.fine_to_liver


def test_intrahepatic_translation_preserves_the_generated_shape() -> None:
    """The nearest valid site retains the entire asymmetric outline and its volume."""
    liver = np.zeros((16, 15, 14), dtype=np.bool_)
    liver[5:14, 3:12, 2:10] = True
    contour = np.zeros_like(liver)
    contour[4:8, 5:8, 5:8] = True
    contour[7:10, 6, 6] = True
    result = translated_placement(contour, liver)
    assert result is not None
    moved, shift = result
    assert shift == (1, 0, 0)
    np.testing.assert_array_equal(np.argwhere(moved), np.argwhere(contour) + shift)
    assert np.all(liver[moved])
    assert translated_placement(contour, np.zeros_like(liver)) is None


def test_surface_contact_rigidly_repositions_a_deep_contour() -> None:
    """A peripheral reference selects the closest site with the same measured contact."""
    liver = np.zeros((35, 35, 35), dtype=np.bool_)
    liver[5:30, 5:30, 5:30] = True
    reference = np.zeros_like(liver)
    reference[5:10, 14:19, 14:19] = True
    deep = np.roll(reference, 9, axis=0)
    target = liver_surface_fraction(reference, liver)
    assert target is not None and target > 0.5
    assert liver_surface_fraction(deep, liver) == 0
    placement = translated_placement(deep, liver, surface_fraction=target)
    assert placement is not None
    moved, shift = placement
    assert liver_surface_fraction(moved, liver) == pytest.approx(target)
    np.testing.assert_array_equal(np.argwhere(moved), np.argwhere(deep) + shift)
    limited = translated_placement(deep, liver, surface_fraction=target, max_distance_mm=2)
    assert limited is not None
    np.testing.assert_array_equal(limited[0], deep)


def _ellipsoid(shape: tuple[int, int, int], center: tuple[float, ...], radii: tuple[float, ...]):
    grid = np.indices(shape, dtype=np.float64)
    offsets = [(grid[i] - center[i]) / radii[i] for i in range(3)]
    return np.sum(np.square(offsets), axis=0) <= 1.0


@pytest.mark.parametrize("radii", [(11.5, 7.6, 5.1), (16.2, 10.8, 7.2), (12.0, 10.5, 8.5)])
def test_rescaling_matches_the_ellipsoid_drawn_with_the_requested_axes(
    radii: tuple[float, float, float],
) -> None:
    """A rescaled contour overlaps the ellipsoid drawn at the requested size and axis ratios."""
    shape, center = (64, 52, 44), (31.3, 25.6, 21.2)
    drawn = _ellipsoid(shape, center, radii)
    scaled = rescaled(
        _ellipsoid(shape, center, (13.5, 9.0, 6.0)),
        int(drawn.sum()),
        radii[1] / radii[0],
        radii[2] / radii[0],
    )
    assert scaled.sum() == pytest.approx(drawn.sum(), rel=0.005)
    # Each voxelization moves the surface by up to half a voxel; unscaled overlap is under 0.86.
    assert 2 * (scaled & drawn).sum() / (scaled.sum() + drawn.sum()) > 0.96


def test_solidity_is_the_share_of_the_convex_hull_a_contour_fills() -> None:
    """An L of two 10 x 4 x 4 arms fills 256 of the 328 voxels in its hull."""
    contour = np.zeros((16, 16, 8), dtype=np.bool_)
    contour[2:12, 2:6, 2:6] = True
    assert solidity(contour) == pytest.approx(1.0)
    contour[2:6, 2:12, 2:6] = True
    # The hull is a 10 x 10 square without a 6 x 6 right triangle, 4 voxels thick.
    assert solidity(contour) == pytest.approx(256 / (4 * (100 - 18)))


def test_capsule_clipping_trims_overhang_and_reaches_surface_contact() -> None:
    """A contour crossing a flat liver surface is cut flush with it, within the allowed share."""
    liver = np.zeros((44, 40, 40), dtype=np.bool_)
    liver[9:] = True
    sphere = _ellipsoid(liver.shape, (16.5, 20.0, 20.0), (8.6, 8.6, 8.6))
    assert 0 < (sphere & ~liver).sum() < OVERHANG * sphere.sum()
    whole = translated_placement(sphere, liver)
    assert whole is not None and whole[1] == (1, 0, 0)
    trimmed = translated_placement(sphere, liver, max_overhang=OVERHANG)
    assert trimmed is not None and trimmed[1] == (0, 0, 0)
    np.testing.assert_array_equal(trimmed[0], sphere & liver)

    # A whole sphere meets a plane tangentially; the reference is cut two voxels deep.
    reference = np.roll(sphere, -2, axis=0) & liver
    target = liver_surface_fraction(reference, liver)
    deep = np.roll(sphere, 10, axis=0)
    tangent = translated_placement(deep, liver, surface_fraction=target)
    assert target is not None and tangent is not None
    rigid = liver_surface_fraction(tangent[0], liver)
    assert rigid is not None and rigid < target - 0.1
    cut = translated_placement(deep, liver, surface_fraction=target, max_overhang=0.1)
    assert cut is not None and cut[1] == (-12, 0, 0)
    np.testing.assert_array_equal(cut[0], reference)


def test_margin_correction_preserves_the_interior_and_the_contrast_ring() -> None:
    """A known HU step has an exact face-gradient solution on the adjacent liver."""
    host = np.zeros((19, 19, 19), dtype=np.float32)
    mask = np.zeros_like(host, dtype=np.bool_)
    mask[6:13, 6:13, 6:13] = True
    liver = np.ones_like(mask)
    draw = host.copy()
    draw[mask] = 100
    target = SpatialTarget((7, 7, 7), (343, 1, 1, 1), 0, 0, 0, crossing_hu=15)
    output = calibrate_appearance(draw, host, mask, liver, target, 100)
    exterior = ndimage.binary_dilation(mask) & ~mask
    np.testing.assert_allclose(100 - output[exterior], 15, atol=1e-5)
    np.testing.assert_array_equal(output[mask], draw[mask])
    ring = ndimage.distance_transform_edt(~mask) >= 3
    np.testing.assert_array_equal(output[ring], host[ring])


def test_known_tumor_anywhere_in_encoded_host_is_rejected(
    prepared: tuple[Path, Path, Path, Path, float], tmp_path: Path
) -> None:
    """VQ sees the full crop, including tumors far outside the placement mask."""
    image, _, mask, liver, ring_mean = prepared
    known = np.zeros((128, 128, 128), dtype=np.float32)
    known_path = _save(tmp_path / "known.nii.gz", known)
    host = load_host(image, liver, mask, known_tumor_mask=known_path)
    assert host.ring_mean_hu == ring_mean
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
