"""Three-dimensional characterization of matched synthetic-lesion experiments.

Method agreement describes sampling trajectories. The intended box and organ
provide placement references; a generated DDIM lesion is not an anatomical truth
mask. CNR uses one physical perilesion background ROI per subject and seed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
from reduced_steps import compare_outputs, hu_distribution, lesion_neighbor_rms
from scipy import ndimage

from lesiondiffusion_mlx import medmlx
from lesiondiffusion_mlx._arrays import BoolArray, FloatArray, HostArray
from lesiondiffusion_mlx._nifti import NiftiImage, nib


def _metadata(image: NiftiImage) -> dict[str, object]:
    """Describe the voxel grid and the coded spatial transforms."""
    return {
        "shape": list(image.shape),
        "affine": image.affine.tolist(),
        "spacing_mm": [float(value) for value in image.header.get_zooms()[:3]],
        "orientation": list(nib.aff2axcodes(image.affine)),
        "qform_code": image.get_qform(coded=True)[1],
        "sform_code": image.get_sform(coded=True)[1],
        "units": list(image.header.get_xyzt_units()),
    }


def _load(path: Path, reference: NiftiImage | None = None) -> tuple[NiftiImage, FloatArray]:
    """Load finite 3-D data and require the reference grid when supplied."""
    image = nib.load(path)
    values = cast(FloatArray, np.asarray(image.dataobj, dtype=np.float32))
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f"Finite three-dimensional NIfTI required: {path}")
    if reference is not None and (
        image.shape != reference.shape
        or not np.allclose(image.affine, reference.affine, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"NIfTI grid differs from the source: {path}")
    return image, values


def _binary(path: Path, reference: NiftiImage) -> BoolArray:
    """Load a non-empty mask whose foreground is exactly label 1."""
    _, values = _load(path, reference)
    if not np.all((values == 0) | (values == 1)) or not np.any(values == 1):
        raise ValueError(f"Non-empty binary 0/1 mask required: {path}")
    return cast(BoolArray, values == 1)


def _delta_summary(delta: FloatArray) -> dict[str, object]:
    """Summarize HU changes without treating voxels as independent subjects."""
    values = delta.astype(np.float64)
    if not values.size:
        return {
            "voxels": 0,
            "signed_mean_hu": None,
            "mean_abs_hu": None,
            "p95_abs_hu": None,
            "max_abs_hu": None,
            "fraction_abs_gt_1_hu": None,
            "fraction_abs_gt_5_hu": None,
        }
    absolute = np.abs(values)
    return {
        "voxels": int(values.size),
        "signed_mean_hu": float(values.mean()),
        "mean_abs_hu": float(absolute.mean()),
        "p95_abs_hu": float(np.percentile(absolute, 95)),
        "max_abs_hu": float(absolute.max()),
        "fraction_abs_gt_1_hu": float(np.mean(absolute > 1)),
        "fraction_abs_gt_5_hu": float(np.mean(absolute > 5)),
    }


def halo_profile(
    before: FloatArray, after: FloatArray, mask: BoolArray, spacing: tuple[float, ...]
) -> dict[str, object]:
    """Measure the existing output halo in Euclidean shells and the formal cube.

    Euclidean distance bins are outside the actual lesion mask. They differ
    from the preserved region's per-axis 33-voxel (Chebyshev) reach: a voxel
    farther than 33 mm in Euclidean distance can still be inside that cube.
    """
    delta = after - before
    distance = ndimage.distance_transform_edt(~mask, sampling=spacing)
    reached = ndimage.maximum_filter(
        mask, size=2 * medmlx.DECODER_REACH_VOXELS + 1, mode="constant", cval=0
    )
    bins: dict[str, object] = {}
    for low, high in ((0, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, 33)):
        bins[f"({low},{high}]"] = _delta_summary(delta[(distance > low) & (distance <= high)])
    bins["(33,infinity)"] = _delta_summary(delta[distance > 33])
    return {
        "inside_actual_mask": _delta_summary(delta[mask]),
        "outside_mask_whole_volume": _delta_summary(delta[~mask]),
        "outside_mask_within_chebyshev_33": _delta_summary(delta[~mask & reached]),
        "outside_euclidean_1_to_5_mm": _delta_summary(delta[(distance >= 1) & (distance <= 5)]),
        "outside_euclidean_distance_bins_mm": bins,
        "outside_chebyshev_33_bitwise_changed": int(
            np.count_nonzero(before[~reached].view(np.uint32) != after[~reached].view(np.uint32))
        ),
    }


def characterize_methods(
    image_path: Path,
    organ_path: Path,
    outputs: Mapping[str, Path],
    *,
    bbox_center: tuple[int, int, int] = (80, 80, 64),
    bbox_size: tuple[int, int, int] = (30, 30, 20),
) -> dict[str, object]:
    """Measure geometry, HU, CNR, localization and pairwise trajectory agreement.

    Every output directory contains ``synthetic_ct.nii.gz`` and
    ``lesion_mask.nii.gz`` on the input grid. CNR background is the 3-10 mm
    Euclidean annulus outside the union of all compared lesions, intersected
    with the target-organ mask and input HU in [-50, 150]. The same ROI is
    used for every method; its output SD uses ddof=0. CNR is undefined when
    the ROI is empty or its output SD is zero.

    HD95 pools the two directed nearest-boundary distances, using
    6-connected voxel boundaries and physical spacing. It measures agreement,
    separately from organ/box containment and intensity characterization.
    """
    reference, before = _load(image_path)
    if nib.aff2axcodes(reference.affine) != ("R", "A", "S"):
        raise ValueError("Characterization requires the prepared RAS grid")
    organ = _binary(organ_path, reference)
    spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
    box = np.asarray(medmlx.bbox_mask(reference.shape, bbox_center, bbox_size), dtype=np.uint8) > 0
    box_world = reference.affine[:3, :3] @ np.asarray(bbox_center) + reference.affine[:3, 3]
    arrays: dict[str, FloatArray] = {}
    masks: dict[str, BoolArray] = {}
    union = np.zeros(before.shape, dtype=bool)
    for method, directory in outputs.items():
        _, arrays[method] = _load(directory / "synthetic_ct.nii.gz", reference)
        masks[method] = _binary(directory / "lesion_mask.nii.gz", reference)
        _load(directory / "raw_synthetic_ct.nii.gz", reference)
        if not np.array_equal(_binary(directory / "organ_mask.nii.gz", reference), organ):
            raise ValueError(f"Output organ mask differs from input: {directory}")
        union |= masks[method]
    distances = ndimage.distance_transform_edt(~union, sampling=spacing)
    background = (distances >= 3) & (distances <= 10) & organ & (before >= -50) & (before <= 150)
    background_count = int(np.count_nonzero(background))
    methods: dict[str, dict[str, object]] = {}
    for method, values in arrays.items():
        mask = masks[method]
        points = np.argwhere(mask)
        count = len(points)
        centroid = points.mean(axis=0)
        world = reference.affine[:3, :3] @ centroid + reference.affine[:3, 3]
        _, components = ndimage.label(mask)
        reached = ndimage.maximum_filter(mask, size=2 * medmlx.DECODER_REACH_VOXELS + 1)
        outside = ~reached
        changed = values[outside].view(np.uint32) != before[outside].view(np.uint32)
        lesion = hu_distribution(values[mask])
        bg = hu_distribution(values[background]) if background_count else None
        signed_cnr = (
            (lesion["mean"] - bg["mean"]) / bg["sd"] if bg is not None and bg["sd"] > 0 else None
        )
        methods[method] = {
            "output": str(outputs[method].resolve()),
            "finite": True,
            "lesion_voxels": count,
            "lesion_volume_mm3": count * float(abs(np.linalg.det(reference.affine[:3, :3]))),
            "centroid_voxel": centroid.tolist(),
            "centroid_world_mm": world.tolist(),
            "centroid_to_intended_box_center_mm": float(np.linalg.norm(world - box_world)),
            "bbox_voxel_min": points.min(axis=0).tolist(),
            "bbox_voxel_max_exclusive": (points.max(axis=0) + 1).tolist(),
            "extent_mm": ((points.max(axis=0) - points.min(axis=0) + 1) * spacing).tolist(),
            "components_6": int(components),
            "fraction_in_organ": float(np.count_nonzero(mask & organ) / count),
            "fraction_in_intended_box": float(np.count_nonzero(mask & box) / count),
            "lesion_to_organ_volume_ratio": count / int(np.count_nonzero(organ)),
            "lesion_hu": lesion,
            "input_hu_at_lesion": hu_distribution(before[mask]),
            "lesion_delta_hu": hu_distribution((values - before)[mask]),
            "lesion_neighbor_difference_rms_hu": lesion_neighbor_rms(values, mask),
            "background_voxels": background_count,
            "background_hu": bg,
            "signed_cnr": signed_cnr,
            "absolute_cnr": abs(signed_cnr) if signed_cnr is not None else None,
            "outside_33_voxel_reach_voxels": int(np.count_nonzero(outside)),
            "outside_33_voxel_reach_bitwise_changed": int(np.count_nonzero(changed)),
            "halo": halo_profile(before, values, mask, spacing),
        }
    pairs: dict[str, object] = {}
    names = list(outputs)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            pair = compare_outputs(outputs[first], outputs[second], image_path)
            intersection = int(np.count_nonzero(masks[first] & masks[second]))
            pair["mask_iou"] = intersection / int(np.count_nonzero(masks[first] | masks[second]))
            pair["hd95_pooled_mm"] = pair["symmetric_surface_p95_mm"]
            pairs[first + "__" + second] = pair
    return {
        "image": str(image_path.resolve()),
        "organ_mask": str(organ_path.resolve()),
        "geometry": _metadata(reference),
        "intended_box_center_voxel": list(bbox_center),
        "intended_box_center_world_mm": box_world.tolist(),
        "intended_box_size_mm": list(bbox_size),
        "intended_box_fraction_in_organ": (
            int(np.count_nonzero(box & organ)) / int(np.count_nonzero(box))
        ),
        "background_definition": (
            "3<=Euclidean distance outside union of method lesions<=10 mm; "
            "liver intersect input HU[-50,150]"
        ),
        "background_voxels": background_count,
        "hd95_definition": (
            "95th percentile of pooled bidirectional nearest "
            "6-connected voxel-boundary distances in mm"
        ),
        "methods": methods,
        "pairs": pairs,
    }


def prepare_amos_case(
    image_path: Path, label_path: Path, metadata_path: Path, output: Path
) -> dict[str, object]:
    """Prepare a registered RAS 1 mm 160x160x128 liver-centered AMOS CT crop.

    The archive's dataset.json must identify label 6 as liver and list this
    image/label pair. CT uses the runtime's affine-coordinate linear sampling;
    labels use its matching nearest-neighbor sampling. The existing inside-box
    selector chooses one fixed seed-23 placement, then both arrays are cropped
    identically and the affine origin is translated by the crop start.
    """
    metadata = cast(dict[str, object], json.loads(metadata_path.read_text()))
    labels = cast(dict[str, str], metadata["labels"])
    if labels.get("6") != "liver":
        raise ValueError("AMOS metadata must identify label 6 as liver")
    validation = cast(list[dict[str, str]], metadata["validation"])
    if not any(
        Path(row["image"]).name == image_path.name and Path(row["label"]).name == label_path.name
        for row in validation
    ):
        raise ValueError("Image/label pair is absent from AMOS validation metadata")
    original, _ = _load(image_path)
    label_image, label_values = _load(label_path, original)
    if not np.all(label_values == np.rint(label_values)) or not np.all(
        (label_values >= 0) & (label_values <= 15)
    ):
        raise ValueError("AMOS labels must be finite integers from 0 through 15")
    # Use the public geometry path, including its shared CT/label sampling and
    # even-padding policy. The only temporary input is the selected label mask.
    with TemporaryDirectory(prefix="amos-liver-prep-") as temporary:
        binary_path = Path(temporary) / "liver.nii.gz"
        header = nib.Nifti1Header.from_header(label_image.header)
        header.set_data_dtype(np.uint8)
        nib.save(
            nib.Nifti1Image((label_values == 6).astype(np.uint8), label_image.affine, header),
            binary_path,
        )
        geometry = medmlx.prepare_geometry(
            source=image_path,
            supplied_mask=binary_path,
            options=medmlx.LesionDiffusionOptions(
                organ="liver",
                lesion_type="tumor",
                bbox_size_mm=(30, 30, 20),
                placement="inside_parenchyma",
                seed=23,
            ),
        )
    standardized_image, standardized_data = geometry.reference, geometry.source_data
    organ, center = geometry.organ_mask, geometry.center
    slices = medmlx.bbox_slices(center, (30, 30, 20), organ.shape)
    if slices is None or not np.all(organ[slices]):
        raise ValueError("The selected 30x30x20 mm box must lie inside the liver")
    start = np.asarray(center) - np.asarray((80, 80, 64))
    shape = np.asarray((160, 160, 128))
    if np.any(start < 0) or np.any(start + shape > organ.shape):
        raise ValueError("Liver-centered crop must fit the standardized image")
    selection = tuple(
        slice(int(low), int(low + size)) for low, size in zip(start, shape, strict=True)
    )
    affine = np.array(standardized_image.affine, copy=True)
    affine[:3, 3] = standardized_image.affine[:3, :3] @ start + standardized_image.affine[:3, 3]
    output.mkdir(parents=True, exist_ok=False)
    paths = {"image": output / "image.nii.gz", "organ_mask": output / "organ-mask.nii.gz"}
    for role, values in (
        ("image", standardized_data[selection].astype(np.float32)),
        ("organ_mask", organ[selection].astype(np.uint8)),
    ):
        header = nib.Nifti1Header.from_header(standardized_image.header)
        header.set_data_dtype(values.dtype.type)
        nib.save(nib.Nifti1Image(cast(HostArray, values), affine, header), paths[role])
    report = {
        "image": str(paths["image"].resolve()),
        "organ_mask": str(paths["organ_mask"].resolve()),
        "source_image": str(image_path.resolve()),
        "source_labels": str(label_path.resolve()),
        "dataset_metadata": str(metadata_path.resolve()),
        "dataset_release": metadata.get("release"),
        "dataset_source": "https://zenodo.org/records/7262581",
        "liver_label": 6,
        "source_image_geometry": _metadata(original),
        "source_label_geometry": _metadata(label_image),
        "resampling": "runtime affine coordinates; linear CT, nearest labels",
        "standardized_geometry": _metadata(standardized_image),
        "crop_start": start.tolist(),
        "bbox_center_voxel": [80, 80, 64],
        "bbox_size_mm": [30, 30, 20],
        "crop_geometry": _metadata(nib.load(paths["image"])),
    }
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
