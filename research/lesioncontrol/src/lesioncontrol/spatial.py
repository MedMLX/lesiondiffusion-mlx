"""Reference-conditioned spatial appearance on 1 mm RAS CT crops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from mlxradiomics import Settings, extract
from numpy.typing import NDArray
from scipy import ndimage, optimize, stats
from scipy.spatial import ConvexHull

from lesioncontrol.profile import FloatImage, Mask, liver_ring

type Vector = NDArray[np.float64]
SHAPE_FEATURES = ("shape_VoxelVolume", "shape_Sphericity", "shape_Elongation", "shape_Flatness")
DIAMETER_FEATURE = "shape_Maximum3DDiameter"
SCALE_MM = 2.0
MARGIN_MM = 3.0
SURFACE_DISTANCE_MM = 3
INTERIOR_MM = 2.0


def shape_values(mask: Mask, features: tuple[str, ...] = SHAPE_FEATURES) -> Vector:
    """Measure shape features of the 3D contour on Metal.

    The default features are volume, sphericity and the two axis ratios.
    """
    result = extract(
        mask.astype(np.float32),
        mask,
        spacing=(1.0, 1.0, 1.0),
        settings=Settings(features=features),
        mode="roi",
    )
    return np.array([result.features[f"original_{key}"] for key in features], dtype=np.float64)


def solidity(mask: Mask) -> float:
    """Contour volume divided by the volume of its convex hull on the 1 mm grid.

    The hull encloses whole voxels, so a convex contour scores one and a
    branched or deeply indented contour scores lower.
    """
    surface = np.argwhere(mask & ~ndimage.binary_erosion(mask))
    corners = np.indices((2, 2, 2)).reshape(3, -1).T - 0.5
    hull = ConvexHull((surface[:, None, :] + corners[None]).reshape(-1, 3))
    return float(np.count_nonzero(mask) / hull.volume)


def _region(mask: Mask, margin: int) -> tuple[slice, ...]:
    points = np.nonzero(mask)
    if not points[0].size:
        raise ValueError("A spatial profile requires a nonempty lesion")
    return tuple(
        slice(max(0, int(axis.min()) - margin), min(n, int(axis.max()) + margin + 1))
        for axis, n in zip(points, mask.shape, strict=True)
    )


def near_liver_surface(liver: Mask) -> Mask:
    """Find tissue within 3 mm of observed non-liver voxels on the 1 mm grid."""
    if liver.all():
        return np.zeros_like(liver)
    return liver & (ndimage.distance_transform_edt(liver) <= SURFACE_DISTANCE_MM)


def liver_surface_fraction(mask: Mask, liver: Mask) -> float | None:
    """Fraction of the observed tumor surface within 3 mm of non-liver tissue.

    Include the tumor in the liver envelope. Discard surface voxels whose
    neighborhood extends past the crop, where surrounding anatomy is unknown.
    """
    radius = SURFACE_DISTANCE_MM
    region = _region(mask, radius + 1)
    local = mask[region]
    surface = local & ~ndimage.binary_erosion(local)
    observed = np.zeros_like(mask)
    observed[radius:-radius, radius:-radius, radius:-radius] = True
    surface &= observed[region]
    if not surface.any():
        return None
    return float(near_liver_surface(liver[region] | local)[surface].mean())


def _projection(signal: Vector, direction: Vector) -> float:
    norm = float(direction @ direction)
    return float(signal @ direction) / norm if norm > 0 else 0.0


def _spread(values: Vector) -> float:
    """Normal-consistent median absolute deviation, in the units of the input.

    Vessels crossing a region must not set its noise level.
    """
    return float(stats.median_abs_deviation(values, scale="normal"))


def _split(residual: Vector, support: Mask, trend: Vector | None = None) -> tuple[Vector, Vector]:
    """Split mean-free HU on a support into parts above and below the 2 mm scale.

    The Gaussian is normalized by the support, so voxels outside it do not
    enter the smooth estimate. The coarse part is orthogonal to the fine part
    and to the optional trend, which keeps changes to one from shifting the
    requested mean, depth trend or the other amplitude.
    """
    field = np.zeros(support.shape, dtype=np.float64)
    field[support] = residual
    weight = ndimage.gaussian_filter(support.astype(np.float64), SCALE_MM, mode="constant")
    smooth = ndimage.gaussian_filter(field, SCALE_MM, mode="constant")[support] / weight[support]
    smooth -= smooth.mean()
    if trend is not None:
        smooth -= _projection(smooth, trend) * trend
    coarse = _projection(residual, smooth) * smooth
    return coarse, residual - coarse


def liver_fine_spread(image: FloatImage, mask: Mask, liver: Mask) -> float:
    """Robust spread of sub-2 mm HU variation in the 3–10 mm liver ring around a lesion.

    Acquisition noise dominates this scale in uninvolved liver.
    """
    region = _region(mask, 10)
    ring = liver_ring(mask[region], liver[region])
    values = np.asarray(image[region][ring], dtype=np.float64)
    return _spread(_split(values - values.mean(), ring)[1])


@dataclass(frozen=True)
class _Components:
    depth: Vector
    coarse: Vector
    fine: Vector
    slope: float
    coarse_sd: float
    fine_sd: float


def _components(image: FloatImage, mask: Mask, support: Mask | None = None) -> _Components:
    """Regress HU on relative depth, then split the residual at a 2 mm Gaussian scale.

    Depth is the distance to the contour divided by its maximum. ``support``
    limits the measured voxels to part of the contour; every array in the
    result then covers that part only.
    """
    region = _region(mask, 1)
    local = mask[region]
    used = local if support is None else local & support[region]
    # A zero halo defines distance to the exterior even at an array face.
    distance = ndimage.distance_transform_edt(np.pad(local, 1))[1:-1, 1:-1, 1:-1]
    depth = distance[used] / distance.max()
    depth -= depth.mean()
    values = np.asarray(image[region][used], dtype=np.float64)
    residual = values - values.mean()
    slope = _projection(residual, depth)
    residual -= slope * depth
    coarse, fine = _split(residual, used, depth)
    return _Components(depth, coarse, fine, slope, float(coarse.std()), float(fine.std()))


@dataclass(frozen=True)
class SpatialTarget:
    """Box, shape, intralesional texture and optional liver-relative measurements.

    ``slope`` is HU per unit relative depth. ``coarse_sd`` and ``fine_sd`` are
    HU standard deviations above and below the 2 mm scale. With a reference
    liver mask, these three exclude contour voxels within 2 mm of the liver
    surface, ``mean_shift_hu`` is the mean HU of the retained voxels minus that
    of the whole contour, and ``fine_to_liver`` is the robust sub-2 mm spread
    of the retained voxels divided by that of the 3–10 mm liver ring.
    ``solidity`` is the contour volume over its convex hull volume and
    ``max_diameter`` the largest distance across the contour surface in mm.
    """

    extent: tuple[int, int, int]
    shape: tuple[float, float, float, float]
    slope: float
    coarse_sd: float
    fine_sd: float
    crossing_hu: float | None = None
    liver_surface_fraction: float | None = None
    fine_to_liver: float | None = None
    mean_shift_hu: float = 0.0
    solidity: float | None = None
    max_diameter: float | None = None

    @classmethod
    def measure(cls, image: FloatImage, mask: Mask, liver: Mask | None = None) -> SpatialTarget:
        """Summarize a whole reference lesion without storing reference voxels."""
        region = _region(mask, 0)
        if any(s.start == 0 or s.stop == n for s, n in zip(region, mask.shape, strict=True)):
            raise ValueError(
                "The reference contour reaches the crop edge; include the whole lesion"
            )
        size = cast(tuple[int, int, int], tuple(s.stop - s.start for s in region))
        *measured, diameter = shape_values(mask, (*SHAPE_FEATURES, DIAMETER_FEATURE)).tolist()
        shape = cast(tuple[float, float, float, float], tuple(measured))
        support, shift = None, 0.0
        envelope = mask if liver is None else liver | mask
        # SciPy's distance is undefined without any exterior voxel in the crop.
        if liver is not None and not envelope.all():
            # Contour voxels at the liver surface average tumor with air, fat or
            # muscle. A reference lying mostly in that layer keeps every voxel.
            interior = mask & (ndimage.distance_transform_edt(envelope) > INTERIOR_MM)
            if 2 * np.count_nonzero(interior) >= np.count_nonzero(mask):
                support = interior
                shift = float(
                    image[interior].mean(dtype=np.float64) - image[mask].mean(dtype=np.float64)
                )
        parts = _components(image, mask, support)
        crossing = None
        if liver is not None:
            region = _region(mask, 1)
            outer, neighbors = _crossing_neighbors(image[region], mask[region], liver[region])
            if outer.any():
                crossing = float(np.nanmean(np.abs(neighbors - image[region][outer])))
        contact = liver_surface_fraction(mask, liver) if liver is not None else None
        ratio = None
        if liver is not None:
            noise = liver_fine_spread(image, mask, liver)
            if noise >= 1.0e-8:
                ratio = _spread(parts.fine) / noise
        return cls(
            size,
            shape,
            parts.slope,
            parts.coarse_sd,
            parts.fine_sd,
            crossing,
            contact,
            ratio,
            shift,
            solidity(mask),
            diameter,
        )

    @classmethod
    def from_record(cls, record: object) -> SpatialTarget:
        """Read the finite 1 mm spatial target saved by the profile command."""
        if not isinstance(record, dict):
            raise ValueError("Generate a reference profile with --spatial for spatial synthesis")
        fields = cast(dict[str, object], record)
        extent, shape = fields.get("extent"), fields.get("shape")
        if not isinstance(extent, (tuple, list)) or not isinstance(shape, (tuple, list)):
            raise ValueError("Spatial target requires three extents and four shape values")
        extents = tuple(cast(tuple[object, ...], extent))
        shapes = tuple(cast(tuple[object, ...], shape))
        if len(extents) != 3 or any(type(x) is not int or x < 1 for x in extents):
            raise ValueError("Spatial target extents must be positive integer millimeters")
        values = (*shapes, fields.get("slope"), fields.get("coarse_sd"), fields.get("fine_sd"))
        if len(shapes) != 4 or any(
            isinstance(x, bool) or not isinstance(x, (float, int)) or not np.isfinite(x)
            for x in values
        ):
            raise ValueError("Invalid spatial target measurements")
        volume, sphericity, elongation, flatness, slope, coarse, fine = map(
            float, cast(tuple[float, ...], values)
        )
        if volume <= 0 or min(sphericity, elongation, flatness, coarse, fine) < 0:
            raise ValueError("Invalid shape or texture scale in spatial target")
        crossing = fields.get("crossing_hu")
        if crossing is not None and (
            type(crossing) not in (int, float)
            or not np.isfinite(float(cast(float, crossing)))
            or float(cast(float, crossing)) < 0
        ):
            raise ValueError("The mean lesion-to-liver HU jump must be finite and nonnegative")
        contact = fields.get("liver_surface_fraction")
        if contact is not None and (
            type(contact) not in (int, float) or not 0 <= float(cast(float, contact)) <= 1
        ):
            raise ValueError("The liver-surface fraction must lie between zero and one")
        ratio = fields.get("fine_to_liver")
        if ratio is not None and (
            type(ratio) not in (int, float)
            or not np.isfinite(float(cast(float, ratio)))
            or float(cast(float, ratio)) < 0
        ):
            raise ValueError("The lesion-to-liver fine ratio must be finite and nonnegative")
        shift = fields.get("mean_shift_hu", 0.0)
        if type(shift) not in (int, float) or not np.isfinite(float(cast(float, shift))):
            raise ValueError("The partial-volume mean shift must be finite HU")
        hull = fields.get("solidity")
        if hull is not None and (
            type(hull) not in (int, float) or not 0 < float(cast(float, hull)) <= 1
        ):
            raise ValueError("Solidity must be positive and at most one")
        diameter = fields.get("max_diameter")
        if diameter is not None and (
            type(diameter) not in (int, float)
            or not np.isfinite(float(cast(float, diameter)))
            or float(cast(float, diameter)) <= 0
        ):
            raise ValueError("The maximum diameter must be finite and positive")
        return cls(
            cast(tuple[int, int, int], extents),
            (volume, sphericity, elongation, flatness),
            slope,
            coarse,
            fine,
            None if crossing is None else float(cast(float, crossing)),
            None if contact is None else float(cast(float, contact)),
            None if ratio is None else float(cast(float, ratio)),
            float(cast(float, shift)),
            None if hull is None else float(cast(float, hull)),
            None if diameter is None else float(cast(float, diameter)),
        )


def _rescale(values: Vector, measured: float, target: float) -> Vector:
    if measured < 1.0e-8:
        if target >= 1.0e-8:
            raise ValueError("The generated ROI has no texture at a requested spatial scale")
        return np.zeros_like(values)
    return values * (target / measured)


def _crossing_neighbors(image: FloatImage, mask: Mask, liver: Mask) -> tuple[Mask, Vector]:
    """Collect lesion HU for each of the six faces abutting an exterior liver voxel."""
    outer = liver & ~mask & ndimage.binary_dilation(mask)
    points = np.argwhere(outer) + 1
    padded = np.pad(np.where(mask, image, np.nan), 1, constant_values=np.nan)
    offsets = np.eye(3, dtype=int)
    neighbors = np.stack([padded[tuple((points + d).T)] for d in np.vstack((offsets, -offsets))])
    return outer, neighbors.astype(np.float64)


def _damp_margin(
    image: FloatImage, mask: Mask, liver: Mask, outside: Vector, target: float
) -> None:
    """Reduce excess sharpness across lesion faces by adjusting neighboring liver only."""
    outer, neighbors = _crossing_neighbors(image, mask, liver)
    if not outer.any():
        return
    base = image[outer].astype(np.float64)
    # The median minimizes absolute jump across all lesion faces sharing this voxel.
    shift = cast(Vector, np.nanmedian(neighbors, axis=0)) - base

    def excess(weight: float) -> float:
        return float(np.nanmean(np.abs(neighbors - base - weight * shift))) - target

    if excess(0.0) <= 0:
        return
    weight = 1.0 if excess(1.0) >= 0 else optimize.brentq(excess, 0.0, 1.0)
    delta = np.zeros(mask.shape, dtype=np.float64)
    delta[outer] = shift
    band = ~mask & liver & (outside < MARGIN_MM)
    smooth = ndimage.gaussian_filter(delta, 1.0, mode="constant")
    support = ndimage.gaussian_filter(outer.astype(np.float64), 1.0, mode="constant")
    taper = np.cos(np.pi * (outside[band] - 1.0) / (2 * (MARGIN_MM - 1.0))) ** 2
    delta[band] = taper * np.divide(
        smooth[band], support[band], out=np.zeros_like(taper), where=support[band] > 0
    )
    delta[outer] = shift
    image[band] += weight * delta[band]


def _host_fine(clean: FloatImage, mask: Mask, liver: Mask, ratio: float) -> Vector:
    """Take the sub-2 mm lesion texture from the host scan's own residual in the contour.

    Acquisition noise does not depend on the tissue it overlies, so a lesion
    inserted into a scan keeps that scan's noise, as in hybrid lesion
    insertion. The residual is scaled so that its robust spread, relative to
    the host's liver ring, equals the reference ratio.
    """
    noise = _components(clean, mask).fine
    return _rescale(noise, _spread(noise), ratio * liver_fine_spread(clean, mask, liver))


def calibrate_appearance(
    generated: FloatImage,
    clean: FloatImage,
    mask: Mask,
    liver: Mask,
    target: SpatialTarget,
    mean_hu: float,
    *,
    host_noise: bool = False,
) -> FloatImage:
    """Transfer scalar depth and texture targets onto the generated spatial pattern.

    Arrays share a 1 mm RAS grid. The interior takes the requested absolute HU
    mean. A 3 mm cosine blend into clean liver preserves surrounding anatomy and
    the independent 3–10 mm liver ring used for contrast measurements. With
    ``host_noise`` the sub-2 mm texture follows the host scan and the target's
    lesion-to-liver ratio; otherwise it takes the target's HU amplitude.
    """
    if not (generated.shape == clean.shape == mask.shape == liver.shape):
        raise ValueError("Spatial calibration requires aligned CT and masks")
    parts = _components(generated, mask)
    if not host_noise:
        fine = _rescale(parts.fine, parts.fine_sd, target.fine_sd)
    elif target.fine_to_liver is None:
        raise ValueError("Host noise requires a reference profiled with its liver mask")
    else:
        fine = _host_fine(clean, mask, liver, target.fine_to_liver)
    values = (
        mean_hu
        + target.slope * parts.depth
        + _rescale(parts.coarse, parts.coarse_sd, target.coarse_sd)
        + fine
    )
    region = _region(mask, int(MARGIN_MM))
    local = mask[region]
    image = generated[region]
    output = clean.copy()
    view = output[region]
    view[local] = values.astype(np.float32)
    outside = ndimage.distance_transform_edt(~local)
    band = (~local) & liver[region] & (outside < MARGIN_MM)
    if band.any():
        # Extend the low-frequency HU correction a short distance across the
        # contour. Mask normalization excludes host pixels from the estimate.
        delta = np.zeros(local.shape, dtype=np.float64)
        delta[local] = values - image[local]
        smooth = ndimage.gaussian_filter(delta, SCALE_MM, mode="constant")
        support = ndimage.gaussian_filter(local.astype(np.float64), SCALE_MM, mode="constant")
        alpha = 0.5 * (1.0 + np.cos(np.pi * outside[band] / MARGIN_MM))
        view[band] += alpha * (image[band] - view[band] + smooth[band] / support[band])
        if target.crossing_hu is not None:
            _damp_margin(view, local, liver[region], outside, target.crossing_hu)
    if not np.isfinite(view).all():
        raise ValueError("Spatial calibration produced non-finite HU")
    return output
