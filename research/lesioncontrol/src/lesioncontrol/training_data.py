"""Prepare source-provenanced real HCC crops within a frozen patient subdivision.

The existing TALON catalog owns CT/SEG pairing and the Radium outline table owns
26-connected component identity. NiBabel performs the same 128³, 1 mm RAS linear
CT and nearest-label resampling as the existing appearance experiment. CT is
explicitly float32 before interpolation: integer output would round interpolated
HU and make exact half-integer ties depend on the host. This CPU entrypoint has
no MLX or model dependency and never changes source annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import nibabel as nib
import nibabel.processing as processing
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

type Affine = NDArray[np.float64]
type FloatImage = NDArray[np.float32]
type Mask = NDArray[np.bool_]


class _NiftiIO(Protocol):
    def set_qform(self, affine: Affine, code: int) -> None: ...
    def set_sform(self, affine: Affine, code: int) -> None: ...
    def to_filename(self, filename: str) -> None: ...


class _HeaderIO(Protocol):
    def set_xyzt_units(self, xyz: str) -> None: ...


class _Processing(Protocol):
    def resample_from_to(
        self,
        image: nib.Nifti1Image,
        target: tuple[tuple[int, int, int], Affine],
        order: int,
        cval: float = 0,
    ) -> nib.Nifti1Image: ...


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    fields = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in fields):
        raise ValueError("JSON keys must be strings")
    return cast(dict[str, object], value)


def _objects(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("Expected a JSON list")
    return [_object(item) for item in cast(list[object], value)]


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("Expected a list of patient identifiers")
    items = cast(list[object], value)
    if not all(isinstance(item, str) for item in items):
        raise ValueError("Expected a list of patient identifiers")
    return cast(list[str], value)


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected a string")
    return value


def _load(path: Path) -> dict[str, object]:
    return _object(json.loads(path.read_text()))


def _write(path: Path, payload: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _save(path: Path, values: FloatImage, affine: Affine) -> None:
    image = nib.Nifti1Image(values, affine)
    writer = cast(_NiftiIO, image)
    writer.set_qform(affine, 1)
    writer.set_sform(affine, 1)
    cast(_HeaderIO, image.header).set_xyzt_units("mm")
    writer.to_filename(str(path))


def _resample(image: nib.Nifti1Image, affine: Affine, order: int, cval: float = 0) -> FloatImage:
    resampler = cast(_Processing, processing)
    return resampler.resample_from_to(
        image, ((128, 128, 128), affine), order=order, cval=cval
    ).get_fdata(dtype=np.float32)


def _binary(image: nib.Nifti1Image) -> Mask:
    values = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Source segmentation requires binary 0/1 labels")
    return values.astype(np.bool_)


def _acquisition_proof(
    patient: str, ct: nib.Nifti1Image, source: dict[str, object], proofs: Sequence[Path]
) -> dict[str, object]:
    """Keep the earlier full-series or independently reconstructed acquisition rule."""
    if ct.shape[2] == source["image_count"]:
        return {"native_z_equals_catalog_series_instances": True}
    for path in proofs:
        record = _load(path)
        if record.get("case") != patient:
            continue
        # Existing source reconstructions name the same Mass comparison either
        # "mass" or "tumor". Conflicting assertions cannot establish provenance.
        mask_fields = [
            name
            for name in ("cached_tumor_array_equal", "cached_mass_array_equal")
            if name in record
        ]
        if len(mask_fields) == 2 and record[mask_fields[0]] != record[mask_fields[1]]:
            raise ValueError("Conflicting Mass equality assertions in acquisition proof")
        mask_equal = bool(mask_fields) and all(record[name] is True for name in mask_fields)
        if (
            record.get("source_series_uid") == source["series_uid"]
            and record.get("selected_ct_shape") == list(ct.shape)
            and record.get("cached_ct_array_equal") is True
            and record.get("cached_ct_affine_equal") is True
            and mask_equal
        ):
            return {"source_acquisition_qc": str(path.resolve()), "verified_acquisition": True}
    raise ValueError(
        "Source acquisition unresolved: cached z differs from catalog series instances"
    )


def prepare_patient(
    patient: str,
    partition: str,
    catalog: dict[str, object],
    outlines: list[dict[str, str]],
    source_data: Path,
    output: Path,
    proofs: Sequence[Path],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Prepare every catalog-eligible component without merging or reranking lesions."""
    source = source_data / patient
    ct = nib.Nifti1Image.from_filename(str(source / "source_contrast.nii.gz"))
    tumor_image = nib.Nifti1Image.from_filename(str(source / "tumor_mask.nii.gz"))
    liver_image = nib.Nifti1Image.from_filename(str(source / "liver_mask.nii.gz"))
    if ct.affine is None:
        raise ValueError("Source CT has no physical affine")
    provenance = _acquisition_proof(patient, ct, _object(catalog["source_contrast"]), proofs)
    for image in (tumor_image, liver_image):
        if (
            image.affine is None
            or image.shape != ct.shape
            or not np.allclose(image.affine, ct.affine, rtol=0, atol=1e-4)
        ):
            raise ValueError("Source CT and segmentation geometry differs")
    if ct.header.get_xyzt_units()[0] not in ("mm", "unknown"):
        raise ValueError("Source image spatial units are not millimeters")
    tumor, liver = _binary(tumor_image), _binary(liver_image)
    if int(tumor.sum()) != catalog["tumor_voxels"]:
        raise ValueError("Source Mass voxel count differs from its owning catalog")
    labels, count = ndimage.label(tumor, structure=np.ones((3, 3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    ranking = sorted(range(1, count + 1), key=lambda label: int(sizes[label]), reverse=True)
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    affine = np.asarray(ct.affine, dtype=np.float64)
    interpolation_ct = nib.Nifti1Image(ct.get_fdata(dtype=np.float32), affine)
    for outline in outlines:
        identifier = outline["lesion_id"]
        try:
            selected = labels == ranking[int(outline["rank"]) - 1]
            if int(selected.sum()) != int(outline["voxel_count"]):
                raise ValueError("Native component identity differs from the outline catalog")
            coordinates = np.argwhere(selected)
            centroid = coordinates.mean(axis=0) @ affine[:3, :3].T + affine[:3, 3]
            expected = np.array([float(outline[f"centroid_{axis}_mm"]) for axis in ("r", "a", "s")])
            if not np.allclose(centroid, expected, rtol=0, atol=0.011):
                raise ValueError("Native component centroid differs from the outline catalog")
            center = (coordinates.min(axis=0) + coordinates.max(axis=0)) / 2
            world = center @ affine[:3, :3].T + affine[:3, 3]
            target = np.eye(4, dtype=np.float64)
            target[:3, 3] = world - 63.5
            selected_image = nib.Nifti1Image(selected.astype(np.uint8), affine)
            organ = nib.Nifti1Image((liver | selected).astype(np.uint8), affine)
            real = _resample(interpolation_ct, target, 1, -1000)
            mask = _resample(selected_image, target, 0) > 0
            all_tumors = _resample(tumor_image, target, 0) > 0
            organ_mask = _resample(organ, target, 0) > 0
            if not np.isfinite(real).all() or not mask.any() or ndimage.label(mask)[1] != 1:
                raise ValueError("Prepared real crop is nonfinite, empty or not face connected")
            distance = ndimage.distance_transform_edt(~mask)
            ring = organ_mask & ~all_tumors & (distance >= 3) & (distance <= 10)
            if int(ring.sum()) < 1000:
                raise ValueError("Fewer than 1000 non-tumor liver voxels in the real 3–10 mm ring")
            directory = output / "crops" / identifier
            directory.mkdir(parents=True, exist_ok=False)
            arrays = {
                "real_ct": real,
                "real_mask": mask.astype(np.float32),
                "real_liver_mask": organ_mask.astype(np.float32),
                "all_tumors_mask": all_tumors.astype(np.float32),
            }
            for role, values in arrays.items():
                _save(directory / f"{role}.nii.gz", values, target)
            results.append(
                {
                    "id": identifier,
                    "patient": patient,
                    "partition": partition,
                    **{role: str((directory / f"{role}.nii.gz").resolve()) for role in arrays},
                    "source_provenance": {
                        "source_paths": {
                            role: str(source / f"{role}.nii.gz")
                            for role in ("source_contrast", "tumor_mask", "liver_mask")
                        },
                        "source_series": catalog["source_contrast"],
                        "segmentation_series": catalog["segmentation"],
                        "segment": "Mass",
                        "outline_catalog_row": outline,
                        **provenance,
                        "native_shape": list(ct.shape),
                        "native_ct_dtype": str(ct.get_data_dtype()),
                        "ct_interpolation_dtype": "float32 input and output",
                        "native_affine": affine.tolist(),
                        "native_qform_code": int(ct.header["qform_code"]),
                        "native_sform_code": int(ct.header["sform_code"]),
                        "native_component_voxels": int(selected.sum()),
                        "native_components": count,
                    },
                    "checks": {
                        "source_geometry_equal": True,
                        "catalog_voxel_count_equal": True,
                        "component_identity_verified": True,
                        "prepared_finite": True,
                        "single_face_connected_mask": True,
                        "liver_ring_voxels": int(ring.sum()),
                    },
                    "prepared_mean_hu": float(real[mask].mean(dtype=np.float64)),
                    "prepared_fraction_below_minus100_hu": float(np.mean(real[mask] < -100)),
                }
            )
        except ValueError as error:
            failures.append({"patient": patient, "id": identifier, "reason": str(error)})
    return results, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof", type=Path, action="append", default=[])
    parser.add_argument("--patients", nargs="+")
    args = parser.parse_args()
    split = _load(args.split)
    train, selection = _strings(split["train_patients"]), _strings(split["selection_patients"])
    reserved = _strings(split["never_train_reserved_patients"])
    if set(train) & set(selection) or (set(train) | set(selection)) & set(reserved):
        raise ValueError("Patient partitions overlap")
    patients = cast(list[str], args.patients) if args.patients else train + selection
    if not set(patients) <= set(train + selection):
        raise ValueError(
            "Preparation cannot admit patients outside the frozen train/selection split"
        )
    catalog = {
        _text(row["id"]): row
        for row in _objects(_load(Path(_text(split["source_case_catalog"])))["cases"])
    }
    original = _load(Path(_text(split["source_split"])))
    eligible = {
        name
        for row in _objects(original["patients"])
        for name in _strings(row["eligible_lesion_ids"])
    }
    with Path(_text(split["shape_metadata_source"])).open(newline="") as stream:
        outlines = [row for row in csv.DictReader(stream) if row["lesion_id"] in eligible]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "manifest.json"
    preprocessing = (
        "Existing NiBabel resample_from_to: 128³ 1 mm RAS bounding-box-center grid; "
        "linear CT uses explicit float32 input and output; nearest masks; "
        "liver includes selected Mass and excludes other Mass"
    )
    index: dict[str, object] = (
        _load(index_path)
        if index_path.exists()
        else {
            "schema": "lesioncontrol.real_hcc_training_data.v1",
            "split": str(Path(args.split).resolve()),
            "cases": [],
            "exclusions": [],
            "attempted_patients": [],
            "preprocessing": preprocessing,
        }
    )
    if index["preprocessing"] != preprocessing:
        raise ValueError("Existing crops use a different interpolation contract; use a new output")
    records, exclusions = _objects(index["cases"]), _objects(index["exclusions"])
    attempted = _strings(index["attempted_patients"])
    for patient in patients:
        if patient in attempted:
            continue
        try:
            prepared, failures = prepare_patient(
                patient,
                "train" if patient in train else "selection",
                catalog[patient],
                [row for row in outlines if row["patient"] == patient],
                Path(args.source_data),
                output,
                cast(list[Path], args.proof),
            )
            records.extend(prepared)
            exclusions.extend(failures)
        except ValueError as error:
            exclusions.append({"patient": patient, "reason": str(error)})
        attempted.append(patient)
        index.update(
            cases=records,
            exclusions=exclusions,
            attempted_patients=attempted,
            counts={
                "attempted_patients": len(attempted),
                "prepared_lesions": len(records),
                "prepared_patients": len({_text(row["patient"]) for row in records}),
                "train_lesions": sum(row["partition"] == "train" for row in records),
                "selection_lesions": sum(row["partition"] == "selection" for row in records),
                "exclusions": len(exclusions),
            },
        )
        _write(index_path, index)
        print(json.dumps({"patient": patient, "counts": index["counts"]}), flush=True)


if __name__ == "__main__":
    main()
