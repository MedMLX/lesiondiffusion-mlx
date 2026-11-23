"""Generate HCC candidates and match a reference profile on a clean liver CT."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import Protocol, cast

import nibabel as nib
import numpy as np
from lesiondiffusion_mlx import medmlx
from scipy.spatial.distance import canberra

from lesioncontrol.backend import require_metal
from lesioncontrol.placement import (
    OVERHANG,
    SURFACE_OVERHANGS,
    liver_center,
    rescaled,
    touches_face,
    translated_placement,
)
from lesioncontrol.profile import (
    Affine,
    Mask,
    ReferenceProfile,
    compare_features,
    extract_features,
    liver_ring,
    load_clean_liver,
    read_image,
    read_mask,
)
from lesioncontrol.spatial import (
    DIAMETER_FEATURE,
    SHAPE_FEATURES,
    SpatialTarget,
    calibrate_appearance,
    liver_surface_fraction,
    shape_values,
    solidity,
)

# LMNet steps for a proposal. Across 15 reference and donor pairs, the contour
# selected from 12-step proposals matched its reference as closely as the one
# selected from 25-step proposals, in half the sampling time.
PROPOSAL_STEPS = 12


class _WritableImage(Protocol):
    def to_filename(self, filename: str) -> None: ...


def _write_mask(mask: Mask, affine: Affine, path: Path) -> None:
    cast(_WritableImage, nib.Nifti1Image(mask.astype(np.uint8), affine)).to_filename(str(path))


def _selection_targets(target: SpatialTarget) -> dict[str, float]:
    """Return the reference values that rank a proposal.

    Scaling sets a proposal's volume, elongation and flatness, so proposals
    are ranked by what it leaves free: sphericity and, when the reference was
    profiled with them, maximum diameter and solidity.
    """
    wanted = {"sphericity": target.shape[1]}
    if target.max_diameter is not None:
        wanted["max_diameter"] = target.max_diameter
    if target.solidity is not None:
        wanted["solidity"] = target.solidity
    return wanted


def synthesize(
    profile: ReferenceProfile,
    *,
    host: Path,
    liver_mask: Path,
    known_tumor_mask: Path | None,
    declared_clean: bool,
    weights: Path,
    output_dir: Path,
    seed: int,
    candidates: int | None = None,
    steps: int = 25,
    proposal_steps: int | None = None,
    center: tuple[int, int, int] | None = None,
    placement_mask: Path | None = None,
    max_translation_mm: float | None = None,
    match_liver_surface: bool = False,
    reference_noise: bool = False,
) -> dict[str, object]:
    """Select a generated contour by its shape measurements and calibrate its CT.

    The profile supplies scalar targets only. Without a placement mask, LMNet
    draws 12 contour proposals by default, each with its own seed, and LINet
    does not run on them. LMNet runs 12 diffusion steps for a proposal unless
    ``proposal_steps`` sets another count; LINet runs ``steps``.
    Each contour is scaled along its principal axes to the reference volume,
    elongation and flatness, then scored by the mean Canberra distance over
    sphericity, maximum diameter and solidity. A supplied outline runs LINet
    once and is used as given.

    The closest shape that fits is placed. Automatic placement may translate
    it to the nearest liver site that holds it. A specified center fixes the
    site unless a translation limit is supplied. Up to 1% of a contour's voxels
    may cross the liver surface and are removed. Surface matching positions
    the contour by the reference's fraction of boundary near the liver surface
    and may remove up to 10% at that surface. LINet then runs once, on the
    placed contour.
    A liver-contrast profile keeps the host scan's sub-2 mm noise in the lesion
    unless ``reference_noise`` requests the reference's HU amplitude.
    Every proposal's mask accompanies the final CT and mask.
    """
    target = SpatialTarget.from_record(profile.metadata.get("spatial_v1"))
    if match_liver_surface and (
        center is not None or placement_mask is not None or target.liver_surface_fraction is None
    ):
        raise ValueError(
            "Surface matching requires automatic placement and a liver-surface profile"
        )
    count = candidates if candidates is not None else (12 if placement_mask is None else 1)
    if count < 1 or (placement_mask is not None and (count != 1 or center is not None)):
        raise ValueError("Use positive candidate count, or one supplied mask without a box center")
    if max_translation_mm is not None and (
        placement_mask is not None or not np.isfinite(max_translation_mm) or max_translation_mm < 0
    ):
        raise ValueError("A finite nonnegative translation limit applies only to generated masks")
    if proposal_steps is not None and placement_mask is not None:
        raise ValueError("Proposal steps apply only to generated masks")
    lmnet_steps = PROPOSAL_STEPS if proposal_steps is None else proposal_steps
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir = output_dir.parent.resolve() / output_dir.name
    clean, affine, liver = load_clean_liver(
        host, liver_mask, known_tumor_mask, declared_clean, shape=None
    )
    if placement_mask is not None:
        supplied = read_mask(placement_mask, affine, shape=clean.shape, connected=True)
        if (supplied & ~liver).any() or touches_face(supplied):
            raise ValueError("Spatial synthesis requires a complete intrahepatic placement")
    automatic = placement_mask is None and center is None
    require_metal()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".lesioncontrol-", dir=output_dir.parent) as temp:
        stage = Path(temp) / "synthesis"
        stage.mkdir()
        records: list[dict[str, object]] = []
        wanted = _selection_targets(target)
        box: tuple[int, ...] | None = None
        site: tuple[int, int, int] | None = None
        translation = (0, 0, 0)

        def inpaint(folder: Path, draw_seed: int, outline: Path) -> tuple[Path, Path]:
            """Run LINet on one contour and return its CT and mask paths."""
            outputs = medmlx.run(
                image_path=host,
                organ_mask_path=liver_mask,
                weights_path=weights,
                output_dir=folder,
                organ="liver",
                lesion_type="tumor",
                placement="inside_parenchyma",
                lesion_mask_path=outline,
                density="isodense lesion",
                density_variation="heterogeneous",
                enhancement="enhanced CT",
                sampler="dpmpp_2m",
                diffusion_steps=steps,
                seed=draw_seed,
                output_region="lesion",
                inference_mode="conditioned",
                decoder_mode="reference",
            )["outputs"]
            return Path(outputs["image"]), Path(outputs["mask"])

        def measure(index: int, drawn: Mask, contour: Mask, mask_path: Path) -> float:
            """Record one contour's measurements and return its selection score."""
            *shape, diameter = shape_values(contour, (*SHAPE_FEATURES, DIAMETER_FEATURE)).tolist()
            measured = {"sphericity": shape[1], "max_diameter": diameter}
            if "solidity" in wanted:
                measured["solidity"] = solidity(contour)
            actual = {key: measured[key] for key in wanted}
            score = float(canberra(list(wanted.values()), list(actual.values())) / len(wanted))
            if not np.isfinite(score):
                raise ValueError("Non-finite candidate morphology")
            records.append(
                {
                    "seed": seed + index,
                    "mask": str(mask_path.relative_to(stage)),
                    "voxels": int(np.count_nonzero(drawn)),
                    "shape": shape,
                    "selection_values": actual,
                    "shape_distance": score,
                    "voxels_outside_liver": int(np.count_nonzero(drawn & ~liver)),
                    "touches_crop_face": touches_face(contour),
                }
            )
            return score

        if placement_mask is not None:
            selected = 0
            image_path, mask_path = inpaint(stage / "draw-000", seed, placement_mask)
            drawn = read_mask(mask_path, affine, shape=clean.shape, connected=True)
            measure(0, drawn, drawn, mask_path)
        else:
            # The runtime uses center ± size // 2. Round odd extents up to retain their span.
            box = tuple(n + n % 2 for n in target.extent)
            site = liver_center(liver, box, seed) if center is None else center
            # LMNet loads once for all proposals and LINet does not run on them.
            masks = medmlx.draw_lesion_masks(
                image_path=host,
                organ_mask_path=liver_mask,
                weights_path=weights,
                organ="liver",
                lesion_type="tumor",
                bbox_size_mm=box,
                bbox_center_voxel=site,
                seeds=[seed + index for index in range(count)],
                enhancement="enhanced CT",
                diffusion_steps=lmnet_steps,
                sampler="dpmpp_2m",
                inference_mode="conditioned",
            )
            (stage / "proposals").mkdir()
            ranked: list[tuple[float, int, Mask]] = []
            for index, proposal in enumerate(masks):
                mask_path = stage / "proposals" / f"{seed + index}.nii.gz"
                _write_mask(proposal, affine, mask_path)
                drawn = read_mask(mask_path, affine, shape=clean.shape, connected=True)
                # The mask model's contours average 15% under the requested volume
                # and have axis ratios of their own.
                contour = (
                    drawn
                    if touches_face(drawn)
                    else rescaled(drawn, round(target.shape[0]), target.shape[2], target.shape[3])
                )
                score = measure(index, drawn, contour, mask_path)
                if not touches_face(contour):
                    ranked.append((score, index, contour))
            fixed = not automatic and max_translation_mm is None
            contact = target.liver_surface_fraction if match_liver_surface else None
            chosen = None
            # Take the closest shape that fits, at the nearest site that holds it.
            for _, index, contour in sorted(ranked, key=lambda x: x[:2]):
                placement = translated_placement(
                    contour,
                    liver,
                    surface_fraction=contact,
                    max_distance_mm=0.0 if fixed else max_translation_mm,
                    max_overhang=SURFACE_OVERHANGS[-1] if match_liver_surface else OVERHANG,
                )
                if placement is not None:
                    chosen = (index, *placement)
                    break
            if chosen is None:
                leakage = [item["voxels_outside_liver"] for item in records]
                truncated_count = sum(item["touches_crop_face"] is True for item in records)
                raise ValueError(
                    f"No contour fits inside the liver. Outside-liver voxel counts {leakage}. "
                    f"{truncated_count} contours touch the crop boundary."
                )
            selected, placed, translation = chosen
            outline = Path(temp) / "outline.nii.gz"
            _write_mask(placed, affine, outline)
            image_path, mask_path = inpaint(stage / "selected", seed + selected, outline)
            if not np.array_equal(placed, read_mask(mask_path, affine, shape=clean.shape)):
                raise RuntimeError("LINet returned a different contour than it was given")
        generated, _ = read_image(image_path, affine, shape=clean.shape)
        mask = read_mask(mask_path, affine, shape=clean.shape, connected=True)
        # Contrast targets follow the host scan in liver attenuation and in noise.
        contrast = profile.target_space == "liver_contrast"
        offset = float(clean[liver_ring(mask, liver)].mean(dtype=np.float64)) if contrast else 0.0
        calibrated = calibrate_appearance(
            generated,
            clean,
            mask,
            liver,
            target,
            profile.target_mean_hu + offset + target.mean_shift_hu,
            host_noise=contrast and not reference_noise,
        )
        comparison = compare_features(
            profile, extract_features(calibrated - np.float32(offset), mask, profile.settings)
        )
        original = nib.Nifti1Image.from_filename(str(image_path))
        output = nib.Nifti1Image(calibrated, affine, original.header.copy(), dtype=np.float32)
        cast(_WritableImage, output).to_filename(str(stage / "synthetic_ct.nii.gz"))
        copyfile(mask_path, stage / "lesion_mask.nii.gz")
        record: dict[str, object] = {
            "schema": "lesioncontrol.spatial_synthesis.v1",
            "mask_source": "lmnet" if placement_mask is None else "supplied",
            "shape_features": SHAPE_FEATURES,
            "target": asdict(target),
            "host_noise": contrast and not reference_noise,
            "target_space": profile.target_space,
            "target_mean_hu": profile.target_mean_hu,
            "clean_host": str(host.resolve()),
            "liver_mask": str(liver_mask.resolve()),
            "known_tumor_mask": str(known_tumor_mask.resolve()) if known_tumor_mask else None,
            "declared_clean": declared_clean,
            "weights": str(weights.resolve()),
            "steps": steps,
            "proposal_steps": lmnet_steps if placement_mask is None else None,
            "box_center_voxel": site,
            "box_size_mm": box,
            "candidates": records,
            "selected_index": selected,
            "translation_voxels": translation,
            "max_translation_mm": max_translation_mm,
            "match_liver_surface": match_liver_surface,
            "achieved_liver_surface_fraction": liver_surface_fraction(mask, liver),
            "selected_diffusion_image": str(image_path.relative_to(stage)),
            "placement_mask": str(placement_mask.resolve()) if placement_mask else None,
            "comparison": comparison,
            "image": "synthetic_ct.nii.gz",
            "mask": "lesion_mask.nii.gz",
        }
        (stage / "synthesis.json").write_text(json.dumps(record, allow_nan=False, indent=2) + "\n")
        # Only expose the output after both images and their measurements are complete.
        if output_dir.exists():
            raise FileExistsError(output_dir)
        stage.rename(output_dir)
    return record
