"""Prepare frozen HCC references in host anatomy outside adapter development.

Only metadata selects reference components and host order. The existing float32
real-crop preparer owns source and lesion identity. Native 1500-center clearance
search proposes a host; the final 1 mm mask is copied rigidly without resizing.
All annotated native tumors must be absent from the full encoded crop. This CPU
entrypoint performs no radiomics extraction or model execution.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import nibabel as nib
import nibabel.processing as processing
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from lesioncontrol.training_data import prepare_patient

type Mask = NDArray[np.bool_]
type Affine = NDArray[np.float64]
type Points = NDArray[np.int64]


class _Writable(Protocol):
    def set_qform(self, affine: Affine, code: int) -> None: ...
    def set_sform(self, affine: Affine, code: int) -> None: ...
    def to_filename(self, filename: str) -> None: ...


class _Header(Protocol):
    def set_xyzt_units(self, xyz: str) -> None: ...


class _Processing(Protocol):
    def resample_from_to(
        self,
        image: nib.Nifti1Image,
        target: tuple[tuple[int, int, int], Affine],
        order: int,
        cval: float = 0,
    ) -> nib.Nifti1Image: ...


def load(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text()))


def write(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save(path: Path, values: NDArray[np.float32] | NDArray[np.uint8], affine: Affine) -> None:
    image = nib.Nifti1Image(values, affine)
    writer = cast(_Writable, image)
    writer.set_qform(affine, 1)
    writer.set_sform(affine, 1)
    cast(_Header, image.header).set_xyzt_units("mm")
    writer.to_filename(str(path))


def read_mask(path: Path) -> Mask:
    values = np.asarray(nib.Nifti1Image.from_filename(str(path)).dataobj)
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"Expected binary mask: {path}")
    return values > 0


def resample(
    image: nib.Nifti1Image, affine: Affine, order: int, cval: float = 0
) -> NDArray[np.float32]:
    return (
        cast(_Processing, processing)
        .resample_from_to(image, ((128, 128, 128), affine), order=order, cval=cval)
        .get_fdata(dtype=np.float32)
    )


def admit(patient: str, allowed: set[str], development: set[str]) -> None:
    """Require both reference and donor to be outside all adapter development."""
    if patient not in allowed or patient in development or patient == "HCC_078":
        raise ValueError(f"Patient cannot enter the fully held-out assay: {patient}")


@dataclass
class NativeHost:
    """Native annotations and bounded clean centers; no CT appearance criterion."""

    patient: str
    ct: nib.Nifti1Image
    tumor_image: nib.Nifti1Image
    liver: Mask
    tumor: Mask
    affine: Affine
    clearance: NDArray[np.float64]
    candidates: Points
    provenance: dict[str, object]


def native_host(patient: str, source: Path, provenance: dict[str, object]) -> NativeHost:
    ct = nib.Nifti1Image.from_filename(str(source / patient / "source_contrast.nii.gz"))
    tumor_image = nib.Nifti1Image.from_filename(str(source / patient / "tumor_mask.nii.gz"))
    liver_image = nib.Nifti1Image.from_filename(str(source / patient / "liver_mask.nii.gz"))
    if ct.affine is None:
        raise ValueError("Host CT has no physical affine")
    for image in (tumor_image, liver_image):
        if (
            image.affine is None
            or image.shape != ct.shape
            or not np.allclose(image.affine, ct.affine, rtol=0, atol=1e-4)
        ):
            raise ValueError("Host CT and annotation geometry differs")
    tumor = read_mask(source / patient / "tumor_mask.nii.gz")
    liver = read_mask(source / patient / "liver_mask.nii.gz")
    spacing = np.asarray(ct.header.get_zooms())
    clearance = ndimage.distance_transform_edt(liver, sampling=spacing)
    # Same conservative physical132mm all-Mass exclusion as the original assay.
    near = tumor
    for axis, size in enumerate(2 * np.ceil(66 / spacing).astype(int) + 1):
        near = ndimage.maximum_filter1d(near, int(size), axis=axis, mode="constant")
    candidates = np.argwhere(liver & ~near & (clearance >= 3))
    ordered = np.argsort(clearance[tuple(candidates.T)])[::-1]
    stride = max(1, len(ordered) // 1500)
    candidates = candidates[ordered[::stride][:1500]]
    return NativeHost(
        patient,
        ct,
        tumor_image,
        liver,
        tumor,
        np.asarray(ct.affine),
        clearance,
        candidates,
        provenance,
    )


def place(
    reference_mask: Mask, host: NativeHost
) -> tuple[dict[str, NDArray[np.float32]], Affine, dict[str, object]] | None:
    """Propose in native spacing, then enforce the exact final RAS outline."""
    coordinates = np.argwhere(reference_mask)
    center = np.floor((coordinates.min(0) + coordinates.max(0)) / 2).astype(np.int64)
    offsets = coordinates - center
    # Only the proposal is nearest-sampled. Delivered shape is never resampled.
    native_offsets = np.unique(
        np.rint(offsets @ np.linalg.inv(host.affine[:3, :3]).T).astype(np.int64), axis=0
    )
    best: tuple[Points, float] | None = None
    for position in host.candidates:
        moved = native_offsets + position
        if np.any(moved < 0) or np.any(moved >= np.asarray(host.tumor.shape)):
            continue
        minimum = float(host.clearance[tuple(moved.T)].min())
        if minimum >= 3 and (best is None or minimum > best[1]):
            best = (position, minimum)
    if best is None:
        return None
    position, minimum = best
    world = host.affine[:3, :3] @ position + host.affine[:3, 3]
    grid = np.eye(4, dtype=np.float64)
    grid[:3, 3] = world - center
    known = resample(host.tumor_image, grid, 0) > 0
    if known.any():
        return None
    organ = nib.Nifti1Image((host.liver | host.tumor).astype(np.uint8), host.affine)
    liver = resample(organ, grid, 0) > 0
    distance = ndimage.distance_transform_edt(~reference_mask)
    margin = (distance <= 3) & ~reference_mask
    ring = liver & ~reference_mask & (distance >= 3) & (distance <= 10)
    if not liver[reference_mask].all() or not liver[margin].all() or int(ring.sum()) < 1000:
        return None
    if (
        ndimage.label(reference_mask)[1] != 1
        or not reference_mask[::-1].transpose(2, 1, 0)[::2, ::2, ::2].any()
    ):
        return None
    ct_float = nib.Nifti1Image(host.ct.get_fdata(dtype=np.float32), host.affine)
    clean = resample(ct_float, grid, 1, -1000)
    if not np.isfinite(clean).all():
        raise ValueError("Qualified host CT contains non-finite values")
    arrays = {
        "clean_ct": clean,
        "liver_mask": liver.astype(np.float32),
        "known_tumors_mask": known.astype(np.float32),
        "placed_mask": reference_mask.astype(np.float32),
    }
    return (
        arrays,
        grid,
        {
            "original_tumor_absent_from_full_host_input": True,
            "all_native_mass_excluded_by_physical132mm_filter": True,
            "rigid_shape_preserved": True,
            "single_face_connected_supplied_mask": True,
            "supplied_mask_and_3mm_margin_inside_liver": True,
            "3to10mm_liver_ring_at_least_1000_voxels": True,
            "half_resolution_mask_nonempty": True,
            "liver_ring_voxels": int(ring.sum()),
            "native_candidate_count": len(host.candidates),
            "native_minimum_outline_clearance_mm": minimum,
            "host_center_ras_mm": world.tolist(),
        },
    )


def prepare(args: argparse.Namespace) -> None:
    plan = load(args.registration)
    if plan["schema"] != "lesioncontrol.heldout_reference_registration.v1":
        raise ValueError("Unsupported held-out registration")
    checkpoint = load(args.checkpoint_metadata)
    development = set(cast(list[str], checkpoint["train_patients"])) | set(
        cast(list[str], checkpoint["selection_patients"])
    )
    allowed = set(cast(list[str], plan["candidate_patients"]))
    candidates = cast(list[dict[str, object]], plan["candidates"])
    for candidate in candidates:
        admit(cast(str, candidate["patient"]), allowed, development)
    source = Path(args.source_data) if args.source_data else Path(cast(str, plan["source_data"]))
    catalog_source = load(args.catalog)
    catalog = {
        cast(str, row["id"]): row for row in cast(list[dict[str, object]], catalog_source["cases"])
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    real_path = output / "real_manifest.json"
    real: dict[str, object] = (
        load(real_path) if real_path.exists() else {"cases": [], "attempts": [], "exclusions": []}
    )
    records = cast(list[dict[str, object]], real["cases"])
    attempts = cast(list[dict[str, object]], real["attempts"])
    failures = cast(list[dict[str, object]], real["exclusions"])
    done = {cast(str, row["patient"]) for row in attempts}
    for candidate in candidates:
        patient = cast(str, candidate["patient"])
        if patient in done:
            continue
        try:
            prepared, errors = prepare_patient(
                patient,
                "heldout",
                catalog[patient],
                [cast(dict[str, str], candidate["outline"])],
                source,
                output,
                [args.proof],
            )
            records.extend(prepared)
            failures.extend(errors)
            attempts.append(
                {"patient": patient, "status": "prepared" if prepared else "excluded_real_qc"}
            )
        except ValueError as error:
            attempts.append({"patient": patient, "status": "excluded_source_qc"})
            failures.append({"patient": patient, "reason": str(error)})
        real.update(cases=records, attempts=attempts, exclusions=failures)
        write(real_path, real)
        print(
            json.dumps({"stage": "real", "patient": patient, "status": attempts[-1]["status"]}),
            flush=True,
        )
    references = {cast(str, row["patient"]): row for row in records}
    bank: dict[str, NativeHost] = {}
    for patient in cast(list[str], plan["candidate_patients"]):
        admit(patient, allowed, development)
        if patient in references:
            bank[patient] = native_host(
                patient, source, cast(dict[str, object], references[patient]["source_provenance"])
            )
            print(
                json.dumps(
                    {
                        "stage": "host",
                        "patient": patient,
                        "bounded_centers": len(bank[patient].candidates),
                    }
                ),
                flush=True,
            )
    index_path = output / "manifest.json"
    index: dict[str, object] = (
        load(index_path)
        if index_path.exists()
        else {
            "schema": "lesioncontrol.conditional_reference_holdout.v1",
            "scope": (
                "Reference AND host patients held out from adapter development; "
                "reference features are conditioning inputs; base-model pretraining overlap unknown"
            ),
            "registration": str(Path(args.registration).resolve()),
            "checkpoint_metadata": str(Path(args.checkpoint_metadata).resolve()),
            "frozen_host_pool": cast(list[str], plan["candidate_patients"]),
            "host_order": "Own patient first, then fixed reserved order excluding own patient",
            "phase": plan["phase"],
            "cases": [],
            "attempts": [],
            "exclusions": failures,
            "counts": {},
            "preparation_complete": False,
        }
    )
    cases = cast(list[dict[str, object]], index["cases"])
    host_attempts = cast(list[dict[str, object]], index["attempts"])
    finished = {cast(str, row["patient"]) for row in host_attempts}
    execution_order = sorted(
        candidates, key=lambda row: float(cast(dict[str, str], row["outline"])["max_diameter_mm"])
    )
    for candidate in execution_order:
        patient = cast(str, candidate["patient"])
        if patient in finished:
            continue
        identifier = cast(str, candidate["id"])
        attempt: dict[str, object] = {
            "patient": patient,
            "id": identifier,
            "hosts": [],
            "status": "excluded_real_qc",
        }
        if patient in references:
            reference = references[patient]
            mask = read_mask(Path(cast(str, reference["real_mask"])))
            real_liver = read_mask(Path(cast(str, reference["real_liver_mask"])))
            all_tumors = read_mask(Path(cast(str, reference["all_tumors_mask"])))
            if np.any(real_liver & all_tumors & ~mask):
                raise ValueError("Reference liver contains another annotated Mass")
            order = [
                patient,
                *[name for name in cast(list[str], plan["candidate_patients"]) if name != patient],
            ]
            attempt["status"] = "excluded_no_qualified_unseen_host"
            for donor in order:
                admit(donor, allowed, development)
                if donor not in bank:
                    continue
                result = place(mask, bank[donor])
                cast(list[dict[str, object]], attempt["hosts"]).append(
                    {"patient": donor, "accepted": result is not None}
                )
                if result is None:
                    continue
                arrays, affine, checks = result
                destination = output / "inputs" / identifier
                destination.mkdir(parents=True, exist_ok=False)
                for role, array in arrays.items():
                    save(destination / f"{role}.nii.gz", array, affine)
                for role in ("real_ct", "real_mask", "real_liver_mask", "all_tumors_mask"):
                    (destination / f"{role}.nii.gz").symlink_to(Path(cast(str, reference[role])))
                cases.append(
                    {
                        "id": identifier,
                        "patient": patient,
                        "partition": "heldout",
                        "host_patient": donor,
                        "input_directory": str(destination),
                        "reference_record": reference,
                        "checks": checks,
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
                            "source_provenance": bank[donor].provenance,
                            "whole_input_original_tumor_absence_verified_again": True,
                        },
                    }
                )
                attempt["status"] = "prepared"
                break
        host_attempts.append(attempt)
        finished.add(patient)
        index.update(
            cases=cases,
            attempts=host_attempts,
            preparation_complete=allowed <= finished,
            counts={
                "registered_candidates": len(allowed),
                "attempted_candidates": len(finished),
                "prepared_reference_patients": len(references),
                "prepared_conditional_cases": len(cases),
                "failed_candidates": sum(row["status"] != "prepared" for row in host_attempts),
                "previously_evaluated_excluded": 1,
            },
        )
        write(index_path, index)
        print(
            json.dumps(
                {
                    "stage": "placement",
                    "patient": patient,
                    "status": attempt["status"],
                    "counts": index["counts"],
                }
            ),
            flush=True,
        )
    write(
        output / "execution.json",
        {
            "host": platform.node(),
            "python": platform.python_version(),
            "versions": {
                name: importlib.metadata.version(name) for name in ("numpy", "scipy", "nibabel")
            },
            "source_data": str(source),
            "source_catalog": str(args.catalog),
            "proof": str(args.proof),
            "registration": str(args.registration),
            "output": str(output),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--checkpoint-metadata", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-data", type=Path)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
