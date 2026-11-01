"""Seeded CT, organ and lesion cases shared by runner tests and fixture recording."""

from __future__ import annotations

from pathlib import Path
from typing import NotRequired, TypedDict, cast

import numpy as np

from lesiondiffusion_mlx._arrays import NumericArray


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


def organ_block(shape: tuple[int, int, int] = CT_SHAPE) -> NumericArray:
    """Construct a block-shaped synthetic organ mask."""
    mask = np.zeros(shape, np.uint8)
    mask[3:21, 3:21, 2:18] = 1
    return mask


def organ_shape(name: str) -> NumericArray:
    """Select a named synthetic organ geometry."""
    if name == "block":
        return organ_block()
    mask = np.zeros(CT_SHAPE, np.uint8)
    if name == "ellipsoid":
        grid = np.indices(CT_SHAPE, dtype=np.float64)
        center = [(size - 1) / 2 for size in CT_SHAPE]
        radius = (9.0, 8.0, 7.0)
        distance = cast(
            NumericArray,
            sum(((grid[axis] - center[axis]) / radius[axis]) ** 2 for axis in range(3)),
        )
        return (distance <= 1.0).astype(np.uint8)
    if name == "thin":
        mask[6:18, 10:13, 8:11] = 1
    elif name == "edge":
        mask[0:6, 0:24, 0:20] = 1
    else:
        raise ValueError(name)
    return mask


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
