"""Seeded CT, organ and lesion cases shared by runner tests and fixture recording."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import numpy as np

from lesiondiffusion_mlx._arrays import NumericArray
from lesiondiffusion_mlx._nifti import nib


class RunArguments(TypedDict):
    image_path: Path
    organ_mask_path: Path
    output_dir: Path
    weights_path: NotRequired[Path]
    organ: str
    lesion_type: str
    bbox_size_mm: list[float] | None
    placement: str
    seed: int
    diffusion_steps: int
    overwrite: NotRequired[bool]
    inference_mode: NotRequired[str]


CT_SHAPE = (24, 24, 20)

# (placement, organ mask, bbox size, requested center, seed)
CENTER_CASES = {
    "inside_seed0": ("inside_parenchyma", "interior", (6, 6, 4), None, 0),
    "inside_seed9": ("inside_parenchyma", "interior", (6, 6, 4), None, 9),
    "inside_odd": ("inside_parenchyma", "interior", (7, 5, 3), None, 9),
    "inside_requested": ("inside_parenchyma", "interior", (6, 6, 4), (12, 12, 10), 0),
    "inside_requested_outside": ("inside_parenchyma", "interior", (6, 6, 4), (2, 2, 2), 0),
    "inside_too_large": ("inside_parenchyma", "interior", (30, 6, 4), None, 0),
    "protruding": ("protruding_from_parenchyma", "boundary", (8, 8, 6), None, 3),
    "protruding_requested": ("protruding_from_parenchyma", "boundary", (8, 8, 6), (12, 12, 10), 3),
    "protruding_requested_bad": (
        "protruding_from_parenchyma",
        "boundary",
        (8, 8, 6),
        (1, 1, 1),
        3,
    ),
    "lumen": ("within_lumen", "boundary", (4, 4, 4), None, 5),
    "lumen_thin": ("within_lumen", "sparse", (2, 2, 2), None, 5),
    "wall": ("wall_thickening", "boundary", (4, 4, 4), None, 7),
    "wall_edge": ("wall_thickening", "edge", (6, 6, 6), None, 7),
}


def ras_affine(origin: tuple[float, float, float] = (-12.0, -12.0, -10.0)) -> NumericArray:
    """Construct a controlled RAS affine for runner crops."""
    affine = np.eye(4)
    affine[:3, 3] = origin
    return affine


@lru_cache(maxsize=1)
def liver_source() -> NumericArray:
    """Read the HCC-TACE liver annotation after checking its CT source grid."""
    root = Path.home() / ".cache/medmlx/test-data"
    ct = nib.load(os.environ.get("MEDMLX_TEST_CT", str(root / "hcc_063_ct.nii.gz")))
    mask = nib.load(os.environ.get("MEDMLX_TEST_MASK", str(root / "hcc_063_liver.nii.gz")))
    if ct.shape != mask.shape or not np.allclose(ct.affine, mask.affine, rtol=0, atol=1e-5):
        raise ValueError("HCC-TACE CT and liver mask grids differ")
    values = np.asarray(mask.dataobj, dtype=np.uint8)
    if not np.any(values) or not np.isin(values, (0, 1)).all():
        raise ValueError("Expected a nonempty binary liver mask")
    return cast(NumericArray, values)


def organ_shape(name: str) -> NumericArray:
    """Select a liver annotation crop with the requested placement edge case."""
    starts = {
        "interior": (110, 205, 50),
        "boundary": (100, 240, 55),
        "sparse": (94, 180, 50),
        "edge": (120, 210, 55),
    }
    start = starts[name]
    crop = liver_source()[tuple(slice(i, i + n) for i, n in zip(start, CT_SHAPE, strict=True))]
    if crop.shape != CT_SHAPE or not np.any(crop):
        raise ValueError(f"Invalid HCC-TACE liver crop {name}: {crop.shape}")
    return crop


def real_scan_data() -> tuple[NumericArray, NumericArray, NumericArray]:
    """Return aligned CT, inclusive liver and tumor crops from HCC-TACE HCC_063."""
    root = Path.home() / ".cache/medmlx/test-data"
    image = nib.load(os.environ.get("MEDMLX_TEST_CT", str(root / "hcc_063_ct.nii.gz")))
    tumor = nib.load(os.environ.get("MEDMLX_TEST_TUMOR", str(root / "hcc_063_tumor.nii.gz")))
    if image.shape != tumor.shape or not np.allclose(image.affine, tumor.affine, rtol=0, atol=1e-5):
        raise ValueError("HCC-TACE CT and tumor mask grids differ")
    region = np.s_[85:109, 250:274, 55:75]
    ct = cast(NumericArray, np.asarray(image.dataobj, dtype=np.float32)[region])
    lesion = cast(NumericArray, np.asarray(tumor.dataobj, dtype=np.uint8)[region])
    if ct.shape != CT_SHAPE or not np.any(lesion) or not np.isin(lesion, (0, 1)).all():
        raise ValueError("Invalid HCC-TACE CT or tumor crop")
    # HCC-TACE stores tumor and liver as separate labels. Include tumor voxels
    # in the organ extent for this runner's organ-overlap contract.
    organ = cast(NumericArray, np.maximum(liver_source()[region], lesion))
    return ct, organ, lesion
