"""Try frozen held-out outlines in a registered donor-only extension.

This reuses the acquired-FOV crop-bank and integer-outline placement functions.
Donor source QC uses the existing patient preparer with no requested reference
outlines, so no donor lesion is selected or measured. The already qualified
HCC_017 case is excluded from this additional-cases manifest.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
from pathlib import Path
from typing import cast

from prepare_heldout import load, native_host, read_mask, write
from qualify_heldout_geometry import crop_bank, outline_in_crop, save_case

from lesioncontrol.training_data import prepare_patient


def prepare(args: argparse.Namespace) -> None:
    plan = load(args.registration)
    if plan["schema"] != "lesioncontrol.unseen_donor_extension.v1":
        raise ValueError("Unsupported donor-extension registration")
    checkpoint = load(args.checkpoint_metadata)
    development = set(cast(list[str], checkpoint["train_patients"])) | set(
        cast(list[str], checkpoint["selection_patients"])
    )
    donor_order = cast(list[str], plan["fixed_donor_order"])
    donor_rows = cast(list[dict[str, object]], plan["donor_metadata"])
    registered_donors = {cast(str, row["id"]): row for row in donor_rows}
    blocked = set(cast(list[str], plan["excluded_before_selection"])) | development
    reference_ids = set(cast(list[str], plan["attempt_only_prior_failed_ids"]))
    if len(donor_order) != 12 or donor_order != sorted(registered_donors):
        raise ValueError("Donor membership or ascending order differs from registration")
    for donor in donor_order:
        row = registered_donors[donor]
        if (
            donor in blocked
            or row["partition"] != "excluded_for_initial_shape_contract"
            or row["previous_intensity_or_model_result_inspection"] is not False
        ):
            raise ValueError(f"Unregistered or previously inspected donor: {donor}")
    real = load(args.reference_manifest)
    references = [
        row for row in cast(list[dict[str, object]], real["cases"]) if row["id"] in reference_ids
    ]
    if {row["id"] for row in references} != reference_ids or len(references) != 6:
        raise ValueError("Exactly the six previously excluded reference outlines are required")
    for reference in references:
        patient = cast(str, reference["patient"])
        if (
            patient in development
            or patient == "HCC_017"
            or cast(str, reference["id"]).split("_L")[0] != patient
        ):
            raise ValueError("Reference overlaps adapter development or the already qualified case")
    catalog_source = load(args.catalog)
    catalog = {
        cast(str, row["id"]): row for row in cast(list[dict[str, object]], catalog_source["cases"])
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema": "lesioncontrol.conditional_reference_holdout.v1",
        "scope": (
            "Additional cases only: fixed reference patients and donor patients outside "
            "adapter development; base-model pretraining overlap unknown"
        ),
        "registration": str(Path(args.registration).resolve()),
        "fixed_donor_order": donor_order,
        "fixed_reference_ids": sorted(reference_ids),
        "already_qualified_HCC_017_unchanged": True,
        "checkpoint_metadata": str(Path(args.checkpoint_metadata).resolve()),
        "phase": "Contrast-enhanced CT; arterial/portal timing unresolved",
        "admission": {
            "reference_and_host_disjoint_from_adapter_train_and_selection": True,
            "registered_donor_only_extension": True,
            "source_qc_contract_unchanged": True,
        },
        "cases": [],
        "attempts": [],
        "donor_attempts": [],
        "counts": {},
        "preparation_complete": False,
    }
    cases = cast(list[dict[str, object]], manifest["cases"])
    attempts = cast(list[dict[str, object]], manifest["attempts"])
    donors = cast(list[dict[str, object]], manifest["donor_attempts"])
    active = list(references)
    for donor in donor_order:
        if not active:
            break
        donor_record: dict[str, object] = {"patient": donor}
        donors.append(donor_record)
        try:
            # Empty outline list executes the unchanged source admission checks,
            # without preparing, choosing or measuring a donor reference tumor.
            prepare_patient(
                donor, "donor_only", catalog[donor], [], args.source_data, output, args.proof
            )
        except ValueError as error:
            donor_record.update(status="excluded_source_qc", reason=str(error))
            write(output / "manifest.json", manifest)
            print(
                {"stage": "donor_source", "patient": donor, "status": "excluded_source_qc"},
                flush=True,
            )
            continue
        provenance: dict[str, object] = {
            "source_paths": {
                role: str(Path(args.source_data) / donor / f"{role}.nii.gz")
                for role in ("source_contrast", "tumor_mask", "liver_mask")
            },
            "source_series": catalog[donor]["source_contrast"],
            "segmentation_series": catalog[donor]["segmentation"],
            "segment": "Mass",
            "existing_source_preparer_admission_passed": True,
            "no_donor_reference_outline_prepared_or_measured": True,
        }
        host = native_host(donor, args.source_data, provenance)
        provenance.update(
            native_shape=list(host.ct.shape),
            native_affine=host.affine.tolist(),
            native_z_equals_catalog_series_instances=(
                host.ct.shape[2]
                == cast(dict[str, object], catalog[donor]["source_contrast"])["image_count"]
            ),
            native_qform_code=int(host.ct.header["qform_code"]),
            native_sform_code=int(host.ct.header["sform_code"]),
            native_mass_voxels=int(host.tumor.sum()),
            source_geometry_and_catalog_mass_count_verified=True,
        )
        bank, bank_attempts = crop_bank(host)
        donor_record.update(
            status="source_qualified", qualified_crops=len(bank), crop_attempts=bank_attempts
        )
        print({"stage": "donor_crops", "patient": donor, "crops": len(bank)}, flush=True)
        remaining: list[dict[str, object]] = []
        for reference in active:
            patient = cast(str, reference["patient"])
            identifier = cast(str, reference["id"])
            attempt: dict[str, object] = {
                "patient": patient,
                "id": identifier,
                "host_patient": donor,
                "status": "no_qualified_crop_bank_placement",
                "crops": [],
            }
            mask = read_mask(Path(cast(str, reference["real_mask"])))
            for crop in bank:
                placed, record = outline_in_crop(mask, crop)
                cast(list[dict[str, object]], attempt["crops"]).append(record)
                if placed is None:
                    continue
                if patient in development or donor in development or donor not in registered_donors:
                    raise ValueError("Reference or host failed final independent-patient admission")
                case = save_case(
                    output / "inputs" / identifier,
                    reference,
                    host,
                    crop.grid,
                    placed,
                    crop.liver,
                    record,
                )
                cases.append(case)
                attempt["status"] = "prepared"
                print({"stage": "placed", "id": identifier, "host": donor}, flush=True)
                break
            if attempt["status"] != "prepared":
                remaining.append(reference)
            attempts.append(attempt)
        active = remaining
        manifest.update(
            counts={
                "registered_reference_candidates": 7,
                "already_qualified_unchanged": 1,
                "additional_reference_candidates": 6,
                "registered_donors": 12,
                "attempted_donors": len(donors),
                "prepared_additional_cases": len(cases),
                "remaining_reference_exclusions": len(active),
            }
        )
        write(output / "manifest.json", manifest)
    manifest.update(
        preparation_complete=True,
        exclusions=[
            {
                "id": row["id"],
                "patient": row["patient"],
                "reason": "No valid placement in registered donor-only extension",
            }
            for row in active
        ],
        counts={
            "registered_reference_candidates": 7,
            "already_qualified_unchanged": 1,
            "additional_reference_candidates": 6,
            "registered_donors": 12,
            "attempted_donors": len(donors),
            "source_qualified_donors": sum(row["status"] == "source_qualified" for row in donors),
            "prepared_additional_cases": len(cases),
            "remaining_reference_exclusions": len(active),
        },
    )
    write(output / "manifest.json", manifest)
    write(
        output / "execution.json",
        {
            "host": platform.node(),
            "python": platform.python_version(),
            "versions": {
                name: importlib.metadata.version(name) for name in ("numpy", "scipy", "nibabel")
            },
            "source_data": str(args.source_data),
            "source_catalog": str(args.catalog),
            "proofs": [str(path) for path in args.proof],
            "registration": str(args.registration),
            "output": str(output),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-metadata", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--proof", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
