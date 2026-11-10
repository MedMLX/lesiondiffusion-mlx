"""Reference appearance profiles and prepared, independently clean host inputs.

V0 accepts aligned 128³ NIfTI-1 arrays on an axis-aligned 1 mm RAS grid. It never
resamples, clips, erodes, joins lesions, or fabricates normal tissue. Contrast
profiles subtract the reference liver-ring mean before feature extraction;
generated features subtract the fixed clean-host ring mean. The common -1000
bin anchor therefore applies to intensities in the declared profile space.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, NoReturn, TypedDict, cast

import nibabel as nib
import numpy as np
from mlxradiomics import Settings, list_features
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial.distance import canberra

TargetSpace = Literal["absolute_hu", "liver_contrast"]
FloatImage = NDArray[np.float32]
Mask = NDArray[np.bool_]
Affine = NDArray[np.float64]
FAMILIES = ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
MEAN_FEATURE = "original_firstorder_Mean"


class FeatureDeviation(TypedDict):
    """One raw feature comparison, retained even when its optimization weight is zero."""

    target: float
    generated: float
    difference: float
    canberra: float


class FeatureComparison(TypedDict):
    """Family-balanced objective and every unaggregated feature deviation."""

    score: float
    family_scores: dict[str, float]
    deviations: dict[str, FeatureDeviation]


def default_settings() -> Settings:
    """Return the fixed six-family appearance contract; shape is not an objective."""
    return Settings(
        label=1,
        features=list_features(*FAMILIES, mode="roi", ndim=3),
        bin_width=25.0,
        bin_anchor=-1000.0,
        normalize=False,
        resegment_range=None,
        distances=(1,),
        symmetrical_glcm=True,
        glcm_aggregation="3d_average",
        glrlm_aggregation="3d_average",
        glszm_aggregation="3d",
        gldm_aggregation="3d",
        ngtdm_aggregation="3d",
    )


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object with string keys")
    mapping = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in mapping):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, object], value)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _space(value: object) -> TargetSpace:
    if value not in ("absolute_hu", "liver_contrast"):
        raise ValueError("target_space must be absolute_hu or liver_contrast")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _settings_payload(settings: Settings) -> dict[str, object]:
    return _object(json.loads(json.dumps(asdict(settings), allow_nan=False)), "settings")


def _feature_values(values: Mapping[str, object], settings: Settings) -> dict[str, float]:
    expected = {f"original_{name}" for name in settings.features}
    if set(values) != expected:
        raise ValueError("Feature names must match the profile's appearance settings")
    return {name: _number(value, name) for name, value in values.items()}


def _check_reference_quarantine(record: Mapping[str, object] | None) -> None:
    if record is not None and record.get("reference_qc_status") == "excluded_appearance_reference":
        raise ValueError("Reference is explicitly quarantined as an appearance reference")


@dataclass(frozen=True)
class ReferenceProfile:
    """Target radiomics in one explicit intensity space, with reusable provenance.

    Mean is derived from the Mean feature, including after dataclasses.replace;
    editing that feature cannot leave the numerical conditioning target stale.
    """

    features: dict[str, float]
    target_space: TargetSpace
    settings: Settings
    metadata: dict[str, object]
    target_mean_hu: float = field(init=False)

    def __post_init__(self) -> None:
        _space(self.target_space)
        _check_reference_quarantine(self.metadata)
        qc_record = self.metadata.get("qc_record")
        if qc_record is not None:
            _check_reference_quarantine(_object(qc_record, "metadata.qc_record"))
        if self.settings != default_settings():
            raise ValueError("V0 profiles use the fixed six-family 25 HU appearance settings")
        values = _feature_values(self.features, self.settings)
        object.__setattr__(self, "features", values)
        object.__setattr__(self, "target_mean_hu", values[MEAN_FEATURE])


@dataclass(frozen=True)
class PreparedHost:
    """Clean model input and an unchanged supplied outline on the prepared grid."""

    image_path: Path
    liver_mask_path: Path
    placement_mask_path: Path
    image: FloatImage
    liver_mask: Mask
    placement_mask: Mask
    affine: Affine
    ring_mask: Mask
    ring_mean_hu: float
    declared_clean: bool
    known_tumor_mask_path: Path | None


def _read(path: Path, affine: Affine | None = None) -> tuple[FloatImage, Affine]:
    image = nib.Nifti1Image.from_filename(str(path))
    if image.header.get_xyzt_units()[0] not in ("mm", "unknown"):
        raise ValueError(f"{path}: prepared spatial units must be mm or unspecified")
    matrix = np.asarray(image.affine, dtype=np.float64)
    if image.shape != (128, 128, 128) or not np.isfinite(matrix).all():
        raise ValueError(f"{path}: V0 requires a finite-affine 128³ image")
    if not np.allclose(matrix[:3, :3], np.eye(3), rtol=0, atol=1e-5):
        raise ValueError(f"{path}: V0 requires an axis-aligned 1 mm RAS grid")
    if affine is not None and not np.allclose(matrix, affine, rtol=0, atol=1e-5):
        raise ValueError(f"{path}: physical geometry differs from its CT")
    values = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: contains non-finite voxels")
    return values, matrix


def _mask(path: Path, affine: Affine, *, connected: bool = False, empty: bool = False) -> Mask:
    values, _ = _read(path, affine)
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"{path}: requires binary 0/1 labels")
    mask = values.astype(np.bool_)
    if not empty and not mask.any():
        raise ValueError(f"{path}: required mask is empty")
    if connected and ndimage.label(mask)[1] != 1:
        raise ValueError(f"{path}: select exactly one face-connected lesion")
    return mask


def _ring(mask: Mask, liver: Mask) -> Mask:
    distance = ndimage.distance_transform_edt(~mask)
    ring = liver & ~mask & (distance >= 3) & (distance <= 10)
    if not ring.any():
        raise ValueError("No liver voxels in the 3–10 mm lesion ring")
    return ring


def extract_features(image: FloatImage, mask: Mask, settings: Settings) -> dict[str, float]:
    """Extract finite ROI features with public MLXRadiomics on Metal at 1 mm spacing."""
    import mlx.core as mx
    from mlxradiomics import extract

    if not mx.metal.is_available():
        raise RuntimeError("LesionControl feature extraction requires MLX Metal")
    if image.dtype != np.float32 or mask.dtype != np.bool_ or image.shape != mask.shape:
        raise ValueError("Feature arrays must be aligned float32 image and boolean mask")
    if not np.isfinite(image).all() or not mask.any():
        raise ValueError("Feature extraction requires finite CT and a nonempty mask")
    mx.set_default_device(mx.gpu)
    result = extract(image, mask, settings=settings, spacing=(1.0, 1.0, 1.0), mode="roi")
    return _feature_values(result.features, settings)


def build_profile(
    reference_image: Path,
    reference_mask: Path,
    reference_liver_mask: Path | None = None,
    target_space: TargetSpace = "absolute_hu",
    qc_record: Mapping[str, object] | None = None,
) -> ReferenceProfile:
    """Profile one qualified reference; contrast centering precedes discretization."""
    target_space = _space(target_space)
    _check_reference_quarantine(qc_record)
    image, affine = _read(Path(reference_image))
    mask = _mask(Path(reference_mask), affine, connected=True)
    ring_mean: float | None = None
    ring_voxels = 0
    if reference_liver_mask is not None:
        liver = _mask(Path(reference_liver_mask), affine)
        ring = _ring(mask, liver)
        ring_mean = float(np.mean(image[ring], dtype=np.float64))
        ring_voxels = int(ring.sum())
    if target_space == "liver_contrast" and ring_mean is None:
        raise ValueError("liver_contrast requires a reference liver mask and ring")
    centered = image if target_space == "absolute_hu" else image - np.float32(ring_mean)
    settings = default_settings()
    metadata: dict[str, object] = {
        "reference_image": str(Path(reference_image).resolve()),
        "reference_mask": str(Path(reference_mask).resolve()),
        "reference_liver_mask": str(Path(reference_liver_mask).resolve())
        if reference_liver_mask
        else None,
        "reference_ring_mean_hu": ring_mean,
        "reference_ring_voxels": ring_voxels,
        "spacing_mm": [1.0, 1.0, 1.0],
        "qc_record": dict(qc_record) if qc_record is not None else {},
        "conditioning_inputs": "Reference appearance features are supplied targets",
    }
    return ReferenceProfile(
        extract_features(centered, mask, settings), target_space, settings, metadata
    )


def load_host(
    image: Path,
    liver_mask: Path,
    placement_mask: Path,
    known_tumor_mask: Path | None = None,
    declared_clean: bool = False,
) -> PreparedHost:
    """Load clean context; known tumors must be absent from the entire encoded input.

    Partial liver overlap is allowed for an explicit exophytic placement. The
    liver mask and supplied outline are never clipped or unioned together.
    """
    declared_clean = _boolean(declared_clean, "declared_clean")
    if known_tumor_mask is None and not declared_clean:
        raise ValueError("Supply known-tumor exclusion or explicitly declare this host clean")
    values, affine = _read(Path(image))
    liver = _mask(Path(liver_mask), affine)
    lesion = _mask(Path(placement_mask), affine, connected=True)
    if not np.any(liver & lesion):
        raise ValueError("Placement must overlap the selected liver")
    if not lesion[::-1].transpose(2, 1, 0)[::2, ::2, ::2].any():
        raise ValueError("Placement has no voxel on LINet's half-resolution latent grid")
    if known_tumor_mask is not None and _mask(Path(known_tumor_mask), affine, empty=True).any():
        raise ValueError("Known original tumor remains in the full encoded host input")
    ring = _ring(lesion, liver)
    return PreparedHost(
        Path(image).resolve(),
        Path(liver_mask).resolve(),
        Path(placement_mask).resolve(),
        values,
        liver,
        lesion,
        affine,
        ring,
        float(np.mean(values[ring], dtype=np.float64)),
        declared_clean,
        Path(known_tumor_mask).resolve() if known_tumor_mask is not None else None,
    )


def extract_generated_features(
    profile: ReferenceProfile, host: PreparedHost, generated_image_path: Path
) -> dict[str, float]:
    """Use the fixed clean-host ring for contrast targets, never the generated ring."""
    image, _ = _read(Path(generated_image_path), host.affine)
    if profile.target_space == "liver_contrast":
        image = image - np.float32(host.ring_mean_hu)
    return extract_features(image, host.placement_mask, profile.settings)


def compare_features(
    profile: ReferenceProfile,
    generated_features: Mapping[str, float],
    family_weights: Mapping[str, float] | None = None,
    feature_weights: Mapping[str, float] | None = None,
) -> FeatureComparison:
    """Average standard Canberra distances within families, then across families.

    Default family weights are equal. Feature weights act within each family;
    zero weights exclude a control term while raw deviations remain reported.
    SciPy defines a feature with target=generated=0 to contribute zero.
    """
    generated = _feature_values(generated_features, profile.settings)
    fw, weights = dict(family_weights or {}), dict(feature_weights or {})
    if set(fw) - set(FAMILIES) or set(weights) - set(profile.features):
        raise ValueError("Unknown family or feature weight")
    for name, value in {**fw, **weights}.items():
        if _number(value, name) < 0:
            raise ValueError("Comparison weights must be nonnegative")
    deviations: dict[str, FeatureDeviation] = {}
    family_scores: dict[str, float] = {}
    active: list[tuple[float, float]] = []
    for family in FAMILIES:
        names = [name for name in profile.features if name.startswith(f"original_{family}_")]
        target = np.asarray([profile.features[name] for name in names], dtype=np.float64)
        observed = np.asarray([generated[name] for name in names], dtype=np.float64)
        importance = np.asarray([weights.get(name, 1.0) for name in names], dtype=np.float64)
        for name, wanted, actual in zip(names, target, observed, strict=True):
            deviations[name] = {
                "target": float(wanted),
                "generated": float(actual),
                "difference": float(actual - wanted),
                "canberra": float(canberra([wanted], [actual])),
            }
        total = float(importance.sum())
        if total == 0:
            continue
        score = float(canberra(target, observed, w=importance) / total)
        family_scores[family] = score
        if fw.get(family, 1.0) > 0:
            active.append((score, fw.get(family, 1.0)))
    if not active:
        raise ValueError("At least one appearance feature must have positive weight")
    score = sum(value * weight for value, weight in active) / sum(weight for _, weight in active)
    return {"score": score, "family_scores": family_scores, "deviations": deviations}


def save_profile(profile: ReferenceProfile, path: Path) -> None:
    """Write a finite, editable profile with its exact V0 extraction contract."""
    payload = {
        "schema_version": 1,
        "target_space": profile.target_space,
        "target_mean_hu": profile.target_mean_hu,
        "features": profile.features,
        "settings": _settings_payload(profile.settings),
        "metadata": profile.metadata,
    }
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _bad_constant(value: str) -> NoReturn:
    raise ValueError(f"Non-finite JSON constant is not accepted: {value}")


def load_profile(path: Path) -> ReferenceProfile:
    """Parse a strict profile object and reject inconsistent or unsupported settings."""
    payload = _object(json.loads(Path(path).read_text(), parse_constant=_bad_constant), "profile")
    expected = {
        "schema_version",
        "target_space",
        "target_mean_hu",
        "features",
        "settings",
        "metadata",
    }
    if (
        set(payload) != expected
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
    ):
        raise ValueError("Unsupported profile schema or fields")
    settings = default_settings()
    if _object(payload["settings"], "settings") != _settings_payload(settings):
        raise ValueError("Profile extraction settings differ from the V0 contract")
    profile = ReferenceProfile(
        _feature_values(_object(payload["features"], "features"), settings),
        _space(payload["target_space"]),
        settings,
        _object(payload["metadata"], "metadata"),
    )
    if _number(payload["target_mean_hu"], "target_mean_hu") != profile.target_mean_hu:
        raise ValueError("Stored target_mean_hu disagrees with the Mean feature")
    return profile
