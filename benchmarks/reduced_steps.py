"""Explore sampling step counts against one completed 200-step pipeline run.

This is a one-case morphology and intensity comparison, not a quality-equivalence
qualification. Identical seeds do not pair random draws between schedules or
between LMNet retries. The sampler's requested and actual counts are both saved;
for example the upstream uniform schedule uses 77 steps when 75 are requested.
Each model run delegates to pipeline.py; this file owns only experiment selection
and CPU comparisons of the resulting NIfTI images.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from scipy import ndimage

from lesiondiffusion_mlx._arrays import BoolArray, FloatArray
from lesiondiffusion_mlx._nifti import SpatialAffine, nib
from lesiondiffusion_mlx._schedules import dpm_solver_timesteps, lesion_diffusion_timesteps
from lesiondiffusion_mlx.medmlx import DECODER_REACH_VOXELS, RunResult


class PipelineSample(TypedDict):
    wall_s: float
    peak_mlx_bytes: int
    result: RunResult


class PipelineReport(TypedDict):
    revision: str
    seed: int
    image: str
    organ_mask: str
    weights: str
    samples: list[PipelineSample]


def _report(path: Path) -> PipelineReport:
    return cast(PipelineReport, json.loads(path.read_text()))


def hu_distribution(values: FloatArray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values, dtype=np.float64)),
        "sd": float(np.std(values, dtype=np.float64)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def lesion_neighbor_rms(image: FloatArray, mask: BoolArray) -> float:
    """Return the RMS HU difference of face-neighbor pairs inside the lesion."""
    squared_sum = 0.0
    count = 0
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        pairs = mask[tuple(left)] & mask[tuple(right)]
        differences = (image[tuple(left)] - image[tuple(right)])[pairs].astype(np.float64)
        squared_sum += float(np.sum(differences * differences))
        count += differences.size
    return float(np.sqrt(squared_sum / count)) if count else 0.0


def compare_masks(
    expected_mask: BoolArray,
    actual_mask: BoolArray,
    affine: SpatialAffine,
    spacing: tuple[float, ...],
) -> dict[str, object]:
    """Compare non-empty lesions on one grid using 6-connected voxel boundaries."""
    if expected_mask.shape != actual_mask.shape:
        raise ValueError("Mask comparison requires matching shapes")
    if not np.any(expected_mask) or not np.any(actual_mask):
        raise ValueError("Mask comparison requires non-empty lesions")
    voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
    ref_count, count = int(expected_mask.sum()), int(actual_mask.sum())
    intersection = int(np.count_nonzero(expected_mask & actual_mask))
    ref_center = np.mean(np.argwhere(expected_mask), axis=0)
    center = np.mean(np.argwhere(actual_mask), axis=0)
    ref_boundary = expected_mask & ~ndimage.binary_erosion(expected_mask)
    boundary = actual_mask & ~ndimage.binary_erosion(actual_mask)
    to_reference = ndimage.distance_transform_edt(~ref_boundary, sampling=spacing)[boundary]
    to_candidate = ndimage.distance_transform_edt(~boundary, sampling=spacing)[ref_boundary]
    surface_distances = np.concatenate((to_reference, to_candidate))
    _, components = ndimage.label(actual_mask)
    return {
        "mask_dice": 2.0 * intersection / (ref_count + count),
        "lesion_voxels": count,
        "reference_lesion_voxels": ref_count,
        "lesion_volume_mm3": count * voxel_volume,
        "volume_change_percent": 100.0 * (count - ref_count) / ref_count,
        "centroid_displacement_mm": float(np.linalg.norm(affine[:3, :3] @ (center - ref_center))),
        "connected_components_6": int(components),
        "changed_mask_voxels": int(np.count_nonzero(expected_mask ^ actual_mask)),
        "symmetric_surface_mean_mm": float(surface_distances.mean()),
        "symmetric_surface_p95_mm": float(np.percentile(surface_distances, 95)),
        "symmetric_surface_max_mm": float(surface_distances.max()),
    }


def compare_outputs(reference: Path, candidate: Path, original: Path) -> dict[str, object]:
    """Compare aligned RAS outputs with lesion masks and CT values in HU.

    Surface distances use 6-connected voxel boundaries and physical voxel spacing.
    Distant anatomy is tested outside the decoder's cubic lesion neighborhood,
    where lesion output promises bitwise equality to the input cast to float32.
    """
    source_image = nib.load(original)
    before = np.asarray(source_image.dataobj, dtype=np.float32)
    expected_image = nib.load(reference / "synthetic_ct.nii.gz")
    actual_image = nib.load(candidate / "synthetic_ct.nii.gz")
    expected = np.asarray(expected_image.dataobj, dtype=np.float32)
    actual = np.asarray(actual_image.dataobj, dtype=np.float32)
    expected_mask_image = nib.load(reference / "lesion_mask.nii.gz")
    actual_mask_image = nib.load(candidate / "lesion_mask.nii.gz")
    expected_mask = np.asarray(expected_mask_image.dataobj, dtype=np.uint8) > 0
    actual_mask = np.asarray(actual_mask_image.dataobj, dtype=np.uint8) > 0
    for volume in (expected_image, actual_image, expected_mask_image, actual_mask_image):
        if volume.shape != source_image.shape or not np.allclose(
            volume.affine, source_image.affine, rtol=0.0, atol=1e-6
        ):
            raise ValueError("Step comparisons require output and input on the same RAS grid")
    if not all(np.isfinite(array).all() for array in (before, expected, actual)):
        raise ValueError("Step comparisons require finite CT volumes")
    if not np.any(expected_mask) or not np.any(actual_mask):
        raise ValueError("Step comparisons require non-empty lesions")
    spacing = tuple(float(v) for v in source_image.header.get_zooms()[:3])
    affine = source_image.affine
    reached = ndimage.maximum_filter(
        actual_mask, size=2 * DECODER_REACH_VOXELS + 1, mode="constant"
    )
    distant_delta = actual[~reached] - before[~reached]
    outer_shell = ndimage.binary_dilation(actual_mask) & ~actual_mask
    surroundings = reached & ~actual_mask
    error = (actual - expected).astype(np.float64)
    boundary = actual_mask & ~ndimage.binary_erosion(actual_mask)
    return {
        "shape": list(actual.shape),
        "affine_matches_input": True,
        "finite": True,
        **compare_masks(expected_mask, actual_mask, affine, spacing),
        "lesion_hu": hu_distribution(actual[actual_mask]),
        "reference_lesion_hu": hu_distribution(expected[expected_mask]),
        "candidate_hu_on_reference_lesion": hu_distribution(actual[expected_mask]),
        "lesion_change_from_input_hu": hu_distribution((actual - before)[actual_mask]),
        "boundary_inner_minus_outer_mean_hu": float(
            actual[boundary].mean() - actual[outer_shell].mean()
        ),
        "lesion_neighbor_difference_rms_hu": lesion_neighbor_rms(actual, actual_mask),
        "reference_neighbor_difference_rms_hu": lesion_neighbor_rms(expected, expected_mask),
        "surrounding_change_mean_abs_hu": float(
            np.abs(actual[surroundings] - before[surroundings]).mean()
        ),
        "distant_voxels": int(distant_delta.size),
        "distant_changed_voxels": int(np.count_nonzero(distant_delta)),
        "distant_max_abs_change_hu": float(np.max(np.abs(distant_delta), initial=0.0)),
        "ct_mae_hu": float(np.abs(error).mean()),
        "ct_rmse_hu": float(np.sqrt(np.mean(error * error))),
        "ct_max_abs_error_hu": float(np.abs(error).max()),
        "lesion_union_mae_hu": float(np.abs(error[expected_mask | actual_mask]).mean()),
        "lesion_union_rmse_hu": float(np.sqrt(np.mean(error[expected_mask | actual_mask] ** 2))),
    }


class Arguments(argparse.Namespace):
    reference: Path
    output: Path
    steps: list[int]
    lmnet_steps: int | None
    linet_steps: int | None
    sampler: str
    decoder_mode: str
    center: list[int]
    evaluate_only: bool


def main() -> None:
    """Run requested schedules serially, then report their observed output drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="200-step measurements.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, nargs="+", default=[25, 50, 75, 100, 150])
    parser.add_argument("--lmnet-steps", type=int, help="Hold LMNet at this requested count")
    parser.add_argument("--linet-steps", type=int, help="Hold LINet at this requested count")
    parser.add_argument("--sampler", choices=("ddim", "dpmpp_2m"), default="ddim")
    parser.add_argument("--decoder-mode", choices=("reference", "scaled_fp16"), default="reference")
    parser.add_argument("--center", type=int, nargs=3, default=[80, 80, 64])
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args(namespace=Arguments())
    baseline = _report(args.reference)
    reference_outputs = args.reference.parent / "run-0"
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "reference": str(args.reference.resolve()),
        "experimental_unit": "one real CT crop, one seed, one lesion per schedule",
        "noise_comparison": "same seed; schedules and LMNet retry counts change random draws",
        "sampler": args.sampler,
        "decoder_mode": args.decoder_mode,
        "eta": 1 if args.sampler == "ddim" else None,
        "rows": rows,
    }
    for steps in args.steps:
        schedule = lesion_diffusion_timesteps if args.sampler == "ddim" else dpm_solver_timesteps
        actual_steps = schedule(steps)
        lmnet_steps = steps if args.lmnet_steps is None else args.lmnet_steps
        linet_steps = steps if args.linet_steps is None else args.linet_steps
        run_name = (
            f"steps-{steps}"
            if args.lmnet_steps is None and args.linet_steps is None
            else f"lm-{lmnet_steps}-lin-{linet_steps}"
        )
        directory = args.output / (
            run_name if args.sampler == "ddim" else f"{args.sampler}-{run_name}"
        )
        row: dict[str, object] = {
            "requested_steps": steps,
            "actual_steps": len(actual_steps),
            "timesteps_ascending": tuple(sorted(actual_steps)),
            "stage_steps_requested": {"lmnet": lmnet_steps, "linet": linet_steps},
            "stage_steps_actual": {
                "lmnet": len(schedule(lmnet_steps)),
                "linet": len(schedule(linet_steps)),
            },
        }
        if not args.evaluate_only:
            command = [
                sys.executable,
                str(Path(__file__).with_name("pipeline.py")),
                "--image",
                baseline["image"],
                "--organ-mask",
                baseline["organ_mask"],
                "--weights",
                baseline["weights"],
                "--output",
                str(directory),
                "--seed",
                str(baseline["seed"]),
                "--steps",
                str(steps),
                "--sampler",
                args.sampler,
                "--decoder-mode",
                args.decoder_mode,
                "--output-region",
                "lesion",
                "--center",
                *(str(v) for v in args.center),
            ]
            if args.lmnet_steps is not None:
                command.extend(["--lmnet-steps", str(args.lmnet_steps)])
            if args.linet_steps is not None:
                command.extend(["--linet-steps", str(args.linet_steps)])
            completed = subprocess.run(command, check=False)
            row["returncode"] = completed.returncode
            if completed.returncode:
                row["status"] = "failed"
                rows.append(row)
                (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
                continue
        result = _report(directory / "measurements.json")
        sample = result["samples"][0]
        row.update(
            status="completed",
            revision=result["revision"],
            wall_s=sample["wall_s"],
            peak_mlx_bytes=sample["peak_mlx_bytes"],
            stages=sample["result"]["details"]["stages"],
            comparison=compare_outputs(
                reference_outputs, directory / "run-0", Path(baseline["image"])
            ),
        )
        rows.append(row)
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
