"""Qualify acquired-FOV crop banks without coupling crop and lesion centers.

Reference membership and exact outlines remain frozen. Geometry alone selects a
small bank of real tumor-free crops, then the existing conditional-host search
translates each exact 1 mm outline inside those crops. Native annotations are
never edited. CT appearance is read only after a final geometry passes.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import nibabel as nib
import nibabel.affines as affine_operations
import numpy as np
from numpy.typing import NDArray
from prepare_heldout import (
    Affine,
    Mask,
    NativeHost,
    Points,
    admit,
    load,
    native_host,
    read_mask,
    resample,
    save,
    write,
)
from scipy import ndimage


class _AffineOperations(Protocol):
    def apply_affine(self, affine: Affine, points: Points | Affine) -> Affine: ...


def transform(affine: Affine, points: Points | Affine) -> Affine:
    return cast(_AffineOperations, affine_operations).apply_affine(affine, points)


def coordinates(mask: Mask) -> tuple[Points, Points]:
    values = np.argwhere(mask)
    center = np.floor((values.min(0) + values.max(0)) / 2).astype(np.int64)
    return values - center, center


def source_bounds(host: NativeHost, grid: Affine) -> NDArray[np.float64]:
    corners = np.asarray(list(itertools.product((0, 127), repeat=3)))
    native = transform(np.linalg.inv(host.affine) @ grid, corners)
    return np.asarray([native.min(0), native.max(0)])


def clamp(host: NativeHost, grid: Affine) -> tuple[Affine, Points] | None:
    """Shift a RAS 1 mm grid by integer millimeters into acquired source support."""
    linear = host.affine[:3, :3]
    if not np.allclose(linear, np.diag(np.diag(linear)), rtol=0, atol=1e-6):
        raise ValueError("Source-bound clamp requires the acquired axis-aligned CT affine")
    native_corners = np.asarray(list(itertools.product(*[(0, n - 1) for n in host.tumor.shape])))
    world = transform(host.affine, native_corners)
    lower, upper = world.min(0), world.max(0) - 127
    if np.any(lower > upper):
        return None
    origin = grid[:3, 3]
    minimum = np.ceil(lower - origin - 1e-6).astype(np.int64)
    maximum = np.floor(upper - origin + 1e-6).astype(np.int64)
    if np.any(minimum > maximum):
        return None
    shift = np.maximum(minimum, np.minimum(maximum, 0)).astype(np.int64)
    result = grid.copy()
    result[:3, 3] += shift
    bounds = source_bounds(host, result)
    if np.any(bounds[0] < -1e-5) or np.any(bounds[1] > np.asarray(host.tumor.shape) - 1 + 1e-5):
        raise ValueError("Source-bound clamp did not fit acquired CT support")
    return result, shift


def conservative_mass_absent(host: NativeHost, grid: Affine) -> bool:
    """Inspect every native Mass voxel in the physical 132 mm crop-center box."""
    center = grid[:3, 3] + 63.5
    world = np.asarray(list(itertools.product(*[(x - 66, x + 66) for x in center])))
    native = transform(np.asarray(np.linalg.inv(host.affine), dtype=np.float64), world)
    lower = np.maximum(np.ceil(native.min(0)).astype(int), 0)
    upper = np.minimum(np.floor(native.max(0)).astype(int) + 1, host.tumor.shape)
    region = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper, strict=True))
    return not bool(host.tumor[region].any())


def organ_on_grid(host: NativeHost, grid: Affine) -> Mask:
    image = nib.Nifti1Image((host.liver | host.tumor).astype(np.uint8), host.affine)
    return resample(image, grid, 0) > 0


def shape_checks(mask: Mask, liver: Mask) -> tuple[dict[str, object], str]:
    points = np.argwhere(mask)
    if np.any(points.min(0) < 3) or np.any(points.max(0) > 124):
        return {}, "outline_margin_outside_crop"
    distance = ndimage.distance_transform_edt(~mask)
    margin = (distance <= 3) & ~mask
    ring = liver & ~mask & (distance >= 3) & (distance <= 10)
    count = int(ring.sum())
    if not liver[mask].all():
        return {"liver_ring_voxels": count}, "outline_outside_liver"
    if not liver[margin].all():
        return {"liver_ring_voxels": count}, "final_3mm_margin_outside_liver"
    if count < 1000:
        return {"liver_ring_voxels": count}, "fewer_than_1000_ring_voxels"
    if ndimage.label(mask)[1] != 1:
        return {}, "not_face_connected"
    if not mask[::-1].transpose(2, 1, 0)[::2, ::2, ::2].any():
        return {}, "empty_half_resolution_mask"
    return {
        "liver_ring_voxels": count,
        "single_face_connected_supplied_mask": True,
        "supplied_mask_and_3mm_margin_inside_liver": True,
        "3to10mm_liver_ring_at_least_1000_voxels": True,
        "half_resolution_mask_nonempty": True,
    }, "qualified"


def save_case(
    destination: Path,
    reference: dict[str, object],
    host: NativeHost,
    grid: Affine,
    placed: Mask,
    liver: Mask,
    checks: dict[str, object],
) -> dict[str, object]:
    """Write float32 CT only after all admission and geometric checks pass."""
    destination.mkdir(parents=True, exist_ok=False)
    clean = resample(nib.Nifti1Image(host.ct.get_fdata(dtype=np.float32), host.affine), grid, 1)
    known = resample(host.tumor_image, grid, 0) > 0
    if known.any() or not np.isfinite(clean).all():
        raise ValueError("Qualified crop changed its native Mass or finite CT contract")
    for role, values in {
        "clean_ct": clean,
        "placed_mask": placed.astype(np.float32),
        "liver_mask": liver.astype(np.float32),
        "known_tumors_mask": known.astype(np.float32),
    }.items():
        save(destination / f"{role}.nii.gz", values, grid)
    for role in ("real_ct", "real_mask", "real_liver_mask", "all_tumors_mask"):
        (destination / f"{role}.nii.gz").symlink_to(Path(cast(str, reference[role])))
    return {
        "id": reference["id"],
        "patient": reference["patient"],
        "partition": "heldout",
        "host_patient": host.patient,
        "input_directory": str(destination),
        "reference_record": reference,
        "checks": {
            **checks,
            "original_tumor_absent_from_full_host_input": True,
            "all_native_mass_excluded_by_physical132mm_filter": True,
            "entire_crop_inside_source_fov": True,
            "source_coordinate_corner_bounds": source_bounds(host, grid).tolist(),
            "rigid_shape_preserved": True,
        },
        **{
            role: str(destination / f"{role}.nii.gz")
            for role in (
                "real_ct",
                "real_mask",
                "real_liver_mask",
                "clean_ct",
                "liver_mask",
                "placed_mask",
                "known_tumors_mask",
            )
        },
        "clean_provenance": {
            "fully_unseen_host": True,
            "shared_training_patient_host": False,
            "source_provenance": host.provenance,
            "whole_input_original_tumor_absence_verified_again": True,
        },
    }


def clamp_initial(
    original: dict[str, object],
    hosts: dict[str, NativeHost],
    output: Path,
) -> list[dict[str, object]]:
    accepted: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for case in cast(list[dict[str, object]], original["cases"]):
        host = hosts[cast(str, case["host_patient"])]
        old = nib.Nifti1Image.from_filename(cast(str, case["placed_mask"]))
        mask = read_mask(Path(cast(str, case["placed_mask"])))
        result = clamp(host, np.asarray(old.affine))
        record: dict[str, object] = {"id": case["id"], "initial_proposal": case}
        if result is None:
            record["status"] = "source_fov_cannot_fit_128mm"
        else:
            grid, shift = result
            points = np.argwhere(mask) - shift
            record["crop_grid_shift_mm"] = shift.tolist()
            record["opposite_mask_translation_voxels"] = (-shift).tolist()
            record["initial_affine"] = np.asarray(old.affine).tolist()
            record["final_affine"] = grid.tolist()
            if np.any(points < 0) or np.any(points >= 128):
                record["status"] = "translated_mask_leaves_crop"
            elif not conservative_mass_absent(host, grid):
                record["status"] = "clamp_introduced_native_mass_box"
            elif (resample(host.tumor_image, grid, 0) > 0).any():
                record["status"] = "clamp_introduced_native_mass_on_grid"
            else:
                placed = np.zeros((128, 128, 128), dtype=bool)
                placed[tuple(points.T)] = True
                liver = organ_on_grid(host, grid)
                checks, status = shape_checks(placed, liver)
                record["status"] = status
                if status == "qualified":
                    # Integer opposite shifts preserve every supplied voxel in world space.
                    if not np.allclose(
                        transform(np.asarray(old.affine), np.argwhere(mask)),
                        transform(grid, points),
                        rtol=0,
                        atol=1e-6,
                    ):
                        raise ValueError("Source-bound clamp moved the physical lesion")
                    reference = cast(dict[str, object], case["reference_record"])
                    final = save_case(
                        output / "inputs" / cast(str, case["id"]),
                        reference,
                        host,
                        grid,
                        placed,
                        liver,
                        {**checks, "source_fov_clamp": record.copy()},
                    )
                    accepted.append(final)
        records.append(record)
        write(output / "source_fov_clamp.json", {"attempts": records})
        print(
            {"stage": "source_fov_clamp", "id": case["id"], "status": record["status"]}, flush=True
        )
    return accepted


def native_diagnostic(mask: Mask, host: NativeHost) -> dict[str, object]:
    """Replay the initial proposal and expose the stage that rejected a donor."""
    offsets, center = coordinates(mask)
    offsets = np.unique(
        np.rint(offsets @ np.linalg.inv(host.affine[:3, :3]).T).astype(np.int64), axis=0
    )
    in_bounds, fit = 0, 0
    best: tuple[Points, float] | None = None
    for position in host.candidates:
        moved = offsets + position
        if np.any(moved < 0) or np.any(moved >= np.asarray(host.tumor.shape)):
            continue
        in_bounds += 1
        minimum = float(host.clearance[tuple(moved.T)].min())
        if minimum >= 3:
            fit += 1
            if best is None or minimum > best[1]:
                best = position, minimum
    record: dict[str, object] = {
        "patient": host.patient,
        "native_candidates": len(host.candidates),
        "native_in_bounds": in_bounds,
        "native_outline_fit": fit,
    }
    if best is None:
        record["status"] = "no_native_outline_fit"
        return record
    position, minimum = best
    world = transform(host.affine, position)
    grid = np.eye(4, dtype=np.float64)
    grid[:3, 3] = world - center
    record["best_minimum_outline_clearance_mm"] = minimum
    bounds = source_bounds(host, grid)
    record["entire_crop_inside_source_fov"] = bool(
        np.all(bounds[0] >= 0) and np.all(bounds[1] <= np.asarray(host.tumor.shape) - 1)
    )
    if (resample(host.tumor_image, grid, 0) > 0).any():
        record["status"] = "final_grid_has_native_mass"
    else:
        checks, status = shape_checks(mask, organ_on_grid(host, grid))
        record.update(checks)
        record["status"] = status
    return record


@dataclass
class Crop:
    host: NativeHost
    grid: Affine
    liver: Mask
    clearance: NDArray[np.float64]
    candidates: Points
    record: dict[str, object]


def crop_bank(host: NativeHost) -> tuple[list[Crop], list[dict[str, object]]]:
    """Retain at most eight clearance-ranked crops at least16 mm apart."""
    bank: list[Crop] = []
    attempts: list[dict[str, object]] = []
    centers: list[NDArray[np.float64]] = []
    for position in host.candidates:
        world = np.asarray(transform(host.affine, position))
        grid = np.eye(4, dtype=np.float64)
        grid[:3, 3] = world - 63.5
        result = clamp(host, grid)
        record: dict[str, object] = {"native_center": position.tolist()}
        if result is None:
            record["status"] = "source_fov_cannot_fit_128mm"
            attempts.append(record)
            continue
        grid, shift = result
        center = grid[:3, 3] + 63.5
        if any(float(np.linalg.norm(center - previous)) < 16 for previous in centers):
            continue
        record.update(clamp_shift_mm=shift.tolist(), center_ras_mm=center.tolist())
        if not conservative_mass_absent(host, grid):
            record["status"] = "native_mass_in_132mm_box"
        elif (resample(host.tumor_image, grid, 0) > 0).any():
            record["status"] = "native_mass_on_final_grid"
        else:
            liver = organ_on_grid(host, grid)
            clearance = ndimage.distance_transform_edt(liver)
            candidates = np.argwhere(liver & (clearance >= 3))
            ordered = np.argsort(clearance[tuple(candidates.T)])[::-1]
            stride = max(1, len(ordered) // 1500)
            candidates = candidates[ordered[::stride][:1500]]
            record.update(status="qualified_crop", bank_index=len(bank), candidates=len(candidates))
            bank.append(Crop(host, grid, liver, clearance, candidates, record))
            centers.append(center)
        attempts.append(record)
        if len(bank) == 8:
            break
    return bank, attempts


def outline_in_crop(mask: Mask, crop: Crop) -> tuple[Mask | None, dict[str, object]]:
    """Use the established bounded 1 mm conditional-host translation search."""
    offsets, center = coordinates(mask)
    best: tuple[Points, float, Points] | None = None
    fit = 0
    for position in crop.candidates:
        shifted = offsets + position
        if np.any(shifted < 3) or np.any(shifted > 124):
            continue
        minimum = float(crop.clearance[tuple(shifted.T)].min())
        if minimum >= 3:
            fit += 1
            if best is None or minimum > best[1]:
                best = shifted, minimum, position
    record: dict[str, object] = {**crop.record, "outline_fit_candidates": fit}
    if best is None:
        record["status"] = "no_1mm_outline_fit"
        return None, record
    points, minimum, position = best
    placed = np.zeros(mask.shape, dtype=bool)
    placed[tuple(points.T)] = True
    checks, status = shape_checks(placed, crop.liver)
    record.update(checks)
    record.update(
        status=status,
        minimum_outline_clearance_mm=minimum,
        placement_translation_voxels=(position - center).tolist(),
    )
    if status != "qualified":
        return None, record
    if not np.array_equal(np.argwhere(placed) - position, offsets):
        raise ValueError("Conditional placement changed the supplied outline")
    return placed, record


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    output = root / "geometry-v2"
    output.mkdir(parents=True, exist_ok=True)
    if not (root / "crop-bank-extension.json").exists():
        raise ValueError("Freeze bounded crop-bank plan before any new search")
    plan = load(root / "registration.json")
    checkpoint = load(args.checkpoint_metadata)
    allowed = set(cast(list[str], plan["candidate_patients"]))
    development = set(cast(list[str], checkpoint["train_patients"])) | set(
        cast(list[str], checkpoint["selection_patients"])
    )
    references = cast(list[dict[str, object]], load(root / "real_manifest.json")["cases"])
    hosts: dict[str, NativeHost] = {}
    for reference in references:
        patient = cast(str, reference["patient"])
        admit(patient, allowed, development)
        hosts[patient] = native_host(
            patient, args.source_data, cast(dict[str, object], reference["source_provenance"])
        )
    original = load(root / "manifest.json")
    for initial_case in cast(list[dict[str, object]], original["cases"]):
        admit(cast(str, initial_case["patient"]), allowed, development)
        admit(cast(str, initial_case["host_patient"]), allowed, development)
    cases = clamp_initial(original, hosts, output)
    manifest: dict[str, object] = {
        **{
            key: original[key]
            for key in ("schema", "scope", "phase", "host_order", "frozen_host_pool")
        },
        "registration": str(root / "registration.json"),
        "geometry_extension": str(root / "crop-bank-extension.json"),
        "initial_manifest_preserved": str(root / "manifest.json"),
        "checkpoint_metadata": str(args.checkpoint_metadata),
        "admission": {
            "reference_and_host_disjoint_from_adapter_train_and_selection": True,
            "only_registered_seven_patient_pool_allowed": True,
            "previously_evaluated_HCC_078_excluded": True,
        },
        "cases": cases,
        "attempts": [],
        "preparation_complete": False,
    }
    write(output / "manifest.json", manifest)
    diagnostics: list[dict[str, object]] = []
    for reference in references:
        mask = read_mask(Path(cast(str, reference["real_mask"])))
        record = {
            "id": reference["id"],
            "hosts": [native_diagnostic(mask, host) for host in hosts.values()],
        }
        diagnostics.append(record)
        write(output / "initial_search_diagnostics.json", {"references": diagnostics})
        print({"stage": "native_diagnostics", "id": reference["id"]}, flush=True)
    banks: dict[str, list[Crop]] = {}
    bank_records: list[dict[str, object]] = []
    for patient, host in hosts.items():
        banks[patient], attempts = crop_bank(host)
        bank_records.append(
            {"patient": patient, "qualified_crops": len(banks[patient]), "attempts": attempts}
        )
        write(output / "crop_bank.json", {"hosts": bank_records})
        print({"stage": "crop_bank", "patient": patient, "crops": len(banks[patient])}, flush=True)
    attempts = cast(list[dict[str, object]], manifest["attempts"])
    order = cast(list[str], plan["candidate_patients"])
    for reference in sorted(
        references,
        key=lambda row: float(
            cast(int, cast(dict[str, object], row["source_provenance"])["native_component_voxels"])
        ),
    ):
        patient = cast(str, reference["patient"])
        attempt: dict[str, object] = {"patient": patient, "id": reference["id"], "hosts": []}
        if any(row["patient"] == patient for row in cases):
            attempt["status"] = "prepared_source_fov_clamp"
        else:
            mask = read_mask(Path(cast(str, reference["real_mask"])))
            attempt["status"] = "no_qualified_unseen_crop_bank_placement"
            for donor in [patient, *[name for name in order if name != patient]]:
                admit(donor, allowed, development)
                donor_record: dict[str, object] = {"patient": donor, "crops": []}
                cast(list[dict[str, object]], attempt["hosts"]).append(donor_record)
                for crop in banks.get(donor, []):
                    placed, record = outline_in_crop(mask, crop)
                    cast(list[dict[str, object]], donor_record["crops"]).append(record)
                    if placed is not None:
                        case = save_case(
                            output / "inputs" / cast(str, reference["id"]),
                            reference,
                            crop.host,
                            crop.grid,
                            placed,
                            crop.liver,
                            record,
                        )
                        cases.append(case)
                        attempt["status"] = "prepared_crop_bank"
                        break
                if attempt["status"] == "prepared_crop_bank":
                    break
        attempts.append(attempt)
        manifest.update(
            cases=cases,
            attempts=attempts,
            exclusions=[
                {"patient": row["patient"], "id": row["id"], "reason": row["status"]}
                for row in attempts
                if not str(row["status"]).startswith("prepared")
            ],
            counts={
                "registered_candidates": len(allowed),
                "attempted_candidates": len(attempts),
                "prepared_reference_patients": len(references),
                "prepared_conditional_cases": len(cases),
                "failed_candidates": sum(
                    not str(row["status"]).startswith("prepared") for row in attempts
                ),
                "previously_evaluated_excluded": 1,
            },
            preparation_complete=len(attempts) == len(allowed),
        )
        write(output / "manifest.json", manifest)
        print(
            {
                "stage": "crop_placement",
                "id": reference["id"],
                "status": attempt["status"],
                "prepared": len(cases),
            },
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint-metadata", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
