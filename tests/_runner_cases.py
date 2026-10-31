"""Seeded CT, organ, box and malformed-input cases for the package runner."""

from __future__ import annotations

from pathlib import Path
from typing import NotRequired, TypedDict, cast

import numpy as np

from lesiondiffusion_mlx._arrays import NumericArray


class OptionsArguments(TypedDict):
    organ: str
    lesion_type: str
    bbox_size_mm: tuple[float, float, float] | None
    placement: str
    diffusion_steps: NotRequired[int]
    shape: NotRequired[str]
    density_variation: NotRequired[str]
    inference_mode: NotRequired[str]


class RunArguments(TypedDict):
    image_path: Path
    organ_mask_path: Path
    anatomy_mask_path: NotRequired[Path]
    lesion_mask_path: NotRequired[Path]
    output_dir: Path
    weights_path: NotRequired[Path]
    organ: str
    lesion_type: str
    bbox_size_mm: list[float] | None
    placement: str
    seed: int
    diffusion_steps: int
    shape: NotRequired[str]
    density: NotRequired[str]
    density_variation: NotRequired[str]
    enhancement: NotRequired[str]
    surface: NotRequired[str]
    adjacent_relationship: NotRequired[str]
    specific_features: NotRequired[str]
    bbox_center_voxel: NotRequired[list[int]]
    overwrite: NotRequired[bool]
    inference_mode: NotRequired[str]


CT_SHAPE = (24, 24, 20)

REFERENCE_CASES = (
    "valid",
    "valid_origin",
    "bad_suffix",
    "four_d",
    "non_finite",
    "lps",
    "two_mm",
    "oblique",
)
ORGAN_MASK_CASES = (
    "binary",
    "binary_float",
    "grid_shape",
    "grid_affine",
    "non_finite",
    "negative",
    "fractional",
    "multi_label",
    "label_two",
    "empty",
)
# Supplied lesion mask case -> expected InvalidInputError text, or None if accepted.
# The geometry is organ_block() on the default RAS grid with the tiny 16-voxel LINet crop.
LESION_MASK_CASES: dict[str, str | None] = {
    "single": None,
    "single_lps": None,
    "empty": "no lesion voxels",
    "label_two": "binary with label 1 for the lesion",
    "fractional": "non-negative integers",
    "grid_affine": "must share the input CT grid",
    "two_lesions": "one connected lesion; found 2 components",
    "outside_organ": "does not overlap the organ mask",
    "one_voxel_thick": "half-resolution latent grid",
    "wider_than_crop": "must fit LINet's inpainting crop",
}
# (placement, organ mask, bbox size, requested center, seed)
CENTER_CASES = {
    "inside_seed0": ("inside_parenchyma", "block", (6, 6, 4), None, 0),
    "inside_seed9": ("inside_parenchyma", "block", (6, 6, 4), None, 9),
    "inside_odd": ("inside_parenchyma", "block", (7, 5, 3), None, 9),
    "inside_requested": ("inside_parenchyma", "block", (6, 6, 4), (12, 12, 10), 0),
    "inside_requested_outside": ("inside_parenchyma", "block", (6, 6, 4), (2, 2, 2), 0),
    "inside_too_large": ("inside_parenchyma", "block", (30, 6, 4), None, 0),
    "protruding": ("protruding_from_parenchyma", "ellipsoid", (8, 8, 6), None, 3),
    "protruding_requested": ("protruding_from_parenchyma", "ellipsoid", (8, 8, 6), (12, 12, 10), 3),
    "protruding_requested_bad": (
        "protruding_from_parenchyma",
        "ellipsoid",
        (8, 8, 6),
        (1, 1, 1),
        3,
    ),
    "lumen": ("within_lumen", "ellipsoid", (4, 4, 4), None, 5),
    "lumen_thin": ("within_lumen", "thin", (2, 2, 2), None, 5),
    "wall": ("wall_thickening", "ellipsoid", (4, 4, 4), None, 7),
    "wall_edge": ("wall_thickening", "edge", (6, 6, 6), None, 7),
}
OPTION_BASE: OptionsArguments = {
    "organ": "liver",
    "lesion_type": "tumor",
    "bbox_size_mm": (10.0, 10.0, 8.0),
    "placement": "inside_parenchyma",
}
CONVERSION_CASES = (
    "valid",
    "not_object",
    "bad_json",
    "schema",
    "source_ref",
    "converter_version",
    "ema",
    "layout",
    "precision",
)


def ct_data(shape: tuple[int, int, int] = CT_SHAPE) -> NumericArray:
    """A smooth, varied HU volume without randomness."""

    grid = np.indices(shape, dtype=np.float32)
    values = (
        np.sin(grid[0] * np.float32(0.41)) * np.float32(300.0)
        + np.cos(grid[1] * np.float32(0.29)) * np.float32(250.0)
        + grid[2] * np.float32(12.0)
        - np.float32(150.0)
    )
    return values.astype(np.float32)


def ras_affine(origin: tuple[float, float, float] = (-12.0, -12.0, -10.0)) -> NumericArray:
    """Construct a RAS affine for synthetic runner cases."""
    affine = np.eye(4)
    affine[:3, 3] = origin
    return affine


def reference_input(name: str) -> tuple[str, NumericArray, NumericArray]:
    """Return (file suffix, data, affine) for one input-CT case."""

    data, affine = ct_data(), ras_affine()
    if name == "valid":
        return ".nii.gz", data, affine
    if name == "valid_origin":
        return ".nii", data, ras_affine((105.5, -40.0, 210.25))
    if name == "bad_suffix":
        return ".img", data, affine
    if name == "four_d":
        return ".nii.gz", data[..., None], affine
    if name == "non_finite":
        bad = data.copy()
        bad[3, 4, 5] = np.inf
        return ".nii.gz", bad, affine
    if name == "lps":
        return ".nii.gz", data, np.diag((-1.0, -1.0, 1.0, 1.0))
    if name == "two_mm":
        return ".nii.gz", data, np.diag((2.0, 2.0, 2.0, 1.0))
    if name == "oblique":
        angle = 0.05
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0, 0.0],
                [np.sin(angle), np.cos(angle), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return ".nii.gz", data, rotation
    raise ValueError(name)


def organ_block(shape: tuple[int, int, int] = CT_SHAPE) -> NumericArray:
    """Construct a block-shaped synthetic organ mask."""
    mask = np.zeros(shape, np.uint8)
    mask[3:21, 3:21, 2:18] = 1
    return mask


def organ_ellipsoid(shape: tuple[int, int, int] = CT_SHAPE) -> NumericArray:
    """Construct an ellipsoidal synthetic organ mask."""
    grid = np.indices(shape, dtype=np.float64)
    center = [(size - 1) / 2 for size in shape]
    radius = (9.0, 8.0, 7.0)
    distance = cast(
        NumericArray, sum(((grid[axis] - center[axis]) / radius[axis]) ** 2 for axis in range(3))
    )
    return (distance <= 1.0).astype(np.uint8)


def organ_shape(name: str) -> NumericArray:
    """Select a named synthetic organ geometry."""
    if name == "block":
        return organ_block()
    if name == "ellipsoid":
        return organ_ellipsoid()
    if name == "thin":
        mask = np.zeros(CT_SHAPE, np.uint8)
        mask[6:18, 10:13, 8:11] = 1
        return mask
    if name == "edge":
        mask = np.zeros(CT_SHAPE, np.uint8)
        mask[0:6, 0:24, 0:20] = 1
        return mask
    raise ValueError(name)


def organ_mask_input(name: str) -> tuple[NumericArray, NumericArray]:
    """Return (mask data, affine) for one supplied organ-mask case."""

    mask, affine = organ_block(), ras_affine()
    if name == "binary":
        return mask, affine
    if name == "binary_float":
        return mask.astype(np.float32), affine
    if name == "grid_shape":
        return mask[:, :, :18], affine
    if name == "grid_affine":
        return mask, ras_affine((-11.0, -12.0, -10.0))
    if name == "non_finite":
        data = mask.astype(np.float32)
        data[4, 4, 4] = np.nan
        return data, affine
    if name == "negative":
        data = mask.astype(np.int16)
        data[0, 0, 0] = -1
        return data, affine
    if name == "fractional":
        data = mask.astype(np.float32)
        data[4, 4, 4] = 0.5
        return data, affine
    if name == "multi_label":
        data = mask.copy()
        data[3:6, 3:6, 2:5] = 2
        return data, affine
    if name == "label_two":
        return mask * 2, affine
    if name == "empty":
        return np.zeros(CT_SHAPE, np.uint8), affine
    raise ValueError(name)


def lesion_block(shape: tuple[int, int, int] = CT_SHAPE) -> NumericArray:
    """One lesion at voxels [9:14, 10:13, 7:11], inside organ_block."""

    mask = np.zeros(shape, np.uint8)
    mask[9:14, 10:13, 7:11] = 1
    return mask


def lesion_mask_input(name: str) -> tuple[NumericArray, NumericArray]:
    """Return (mask data, affine) for one supplied lesion-mask case."""

    mask, affine = lesion_block(), ras_affine()
    if name == "single":
        return mask, affine
    if name == "single_lps":
        # The same voxels stored LPS: flip I and J and move the origin to their far end.
        flipped = np.diag((-1.0, -1.0, 1.0, 1.0))
        flipped[:3, 3] = affine[:3, 3] + (CT_SHAPE[0] - 1, CT_SHAPE[1] - 1, 0)
        return mask[::-1, ::-1, :], flipped
    if name == "empty":
        return np.zeros(CT_SHAPE, np.uint8), affine
    if name == "label_two":
        return mask * 2, affine
    if name == "fractional":
        data = mask.astype(np.float32)
        data[10, 11, 8] = 0.5
        return data, affine
    if name == "grid_affine":
        return mask, ras_affine((-11.0, -12.0, -10.0))
    data = np.zeros(CT_SHAPE, np.uint8)
    if name == "two_lesions":
        data[:] = mask
        data[4:6, 4:6, 4:6] = 1
    elif name == "outside_organ":
        data[0:3, 0:3, 18:20] = 1
    elif name == "one_voxel_thick":
        # RAS I = 10 maps to source X = 13, which half-size nearest sampling skips.
        data[10, 8:14, 6:12] = 1
    elif name == "wider_than_crop":
        data[3:21, 10:13, 7:11] = 1
    else:
        raise ValueError(name)
    return data, affine


def lesion_samples() -> tuple[NumericArray, NumericArray]:
    """Smooth LMNet-like samples with two blobs and a reference box."""

    shape = (8, 12, 12)
    grid = np.indices(shape, dtype=np.float32)

    def blob(
        center: tuple[float, float, float], radius: tuple[float, float, float]
    ) -> NumericArray:
        """Construct one synthetic lesion probability blob."""
        return np.exp(-sum(((grid[axis] - center[axis]) / radius[axis]) ** 2 for axis in range(3)))

    first = blob((3, 4, 5), (2.5, 3, 3))
    second = 0.7 * blob((5, 9, 8), (1.5, 1.5, 2))
    samples = (first + second).astype(np.float32)
    reference = np.zeros(shape, bool)
    reference[1:6, 1:8, 2:9] = True
    return samples, reference


def conversion_payload(name: str, source_ref: str) -> object:
    """A conversion.json payload, valid unless ``name`` names one defect."""

    payload: dict[str, object] = {
        "schema_version": "medmlx.lesiondiffusion.weights.v1",
        "converter_version": 1,
        "source_ref": source_ref,
        "layout": "source_torch_ncdhw",
        "stored_dtype": "float32",
        "inference_dtype": {"lmnet": "float32", "linet": "float16", "vq": "float16"},
        "ema": True,
    }
    if name == "valid":
        return payload
    if name == "not_object":
        return [payload]
    if name == "bad_json":
        return "{not json"
    if name == "schema":
        payload["schema_version"] = "medmlx.lesiondiffusion.weights.v0"
    elif name == "source_ref":
        payload["source_ref"] = "0" * 40
    elif name == "converter_version":
        payload["converter_version"] = 2
    elif name == "ema":
        payload["ema"] = 1
    elif name == "layout":
        payload["layout"] = "mlx_ndhwc"
    elif name == "precision":
        payload["inference_dtype"] = {"lmnet": "float16", "linet": "float16", "vq": "float16"}
    else:
        raise ValueError(name)
    return payload
