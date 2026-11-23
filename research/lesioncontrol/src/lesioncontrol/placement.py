"""Size and place a generated contour inside the acquired liver on its 1 mm grid."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal

from lesioncontrol.profile import Mask
from lesioncontrol.spatial import MARGIN_MM, SURFACE_DISTANCE_MM, near_liver_surface

# Share of a contour's voxels that may cross the liver surface and be removed.
# Contours from the mask model that cross it do so by under 0.2% of their voxels.
OVERHANG = 0.01
# Budgets tried in turn when matching liver-surface contact. A contour held
# wholly inside the liver touches its surface only tangentially.
SURFACE_OVERHANGS = (0.0, 0.005, 0.02, 0.05, 0.1)
_EXACT_CANDIDATES = 64


def liver_center(liver: Mask, box: tuple[int, ...], seed: int) -> tuple[int, int, int]:
    """Sample one site with clearance for the requested ellipsoid and blending margin.

    When the ellipsoid cannot fit, use the deepest grid-fitting site. The actual
    generated contour still has to fit inside the liver.
    """
    half = np.asarray(box) // 2
    grid = tuple(slice(h, n - h) for h, n in zip(half, liver.shape, strict=True))
    distance = ndimage.distance_transform_edt(np.pad(liver, 1), sampling=1.0 / (half + MARGIN_MM))[
        1:-1, 1:-1, 1:-1
    ][grid]
    if not distance.size or distance.max() <= 0:
        raise ValueError("The requested box has no center inside the acquired liver crop")
    sites = np.argwhere(distance >= min(1.0, float(distance.max())))
    center = sites[np.random.default_rng(seed).integers(len(sites))] + half
    return int(center[0]), int(center[1]), int(center[2])


def touches_face(mask: Mask) -> bool:
    """Return whether the field of view may truncate the contour."""
    return any(np.take(mask, (0, -1), axis=i).any() for i in range(3))


def _largest(mask: Mask) -> Mask:
    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask
    return labels == 1 + int(np.argmax(np.bincount(labels.ravel())[1:]))


def filled(mask: Mask) -> Mask:
    """Fill the holes that each axial section of a contour encloses.

    Expert outlines are drawn section by section and include everything they
    enclose, so real lesion masks rarely have a hole in an axial section. The
    mask model's contours can be pierced by tunnels. A cleft that reaches the
    edge of its section stays open.
    """
    result = mask.copy()
    points = np.argwhere(mask)
    if not len(points):
        return result
    low, high = points.min(axis=0), points.max(axis=0) + 1
    for index in range(low[2], high[2]):
        section = (slice(low[0], high[0]), slice(low[1], high[1]), index)
        result[section] = ndimage.binary_fill_holes(mask[section])
    return result


def rescaled(mask: Mask, voxels: int, elongation: float, flatness: float) -> Mask:
    """Scale a contour along its principal axes to a requested volume and axis ratios.

    The mask model returns contours about 15% smaller than the requested
    lesion on average, with axis ratios of their own. The contour is scaled
    about its centroid along the principal axes of its voxel coordinates, so
    that its minor-to-major and least-to-major axis ratios become
    ``elongation`` and ``flatness``. Its signed distance field is resampled at
    the scaled coordinates and thresholded at the level that leaves the
    requested voxel count, which keeps the outline's irregularity. The result
    can differ from the count by voxels tied at that level, and it reaches the
    array face when the scaled contour does not fit.
    """
    count = int(np.count_nonzero(mask))
    if count == 0 or voxels < 1 or min(elongation, flatness) <= 0:
        raise ValueError(
            "Rescaling requires a nonempty contour, a positive volume and positive axis ratios"
        )
    points = np.argwhere(mask)
    center = points.mean(axis=0)
    offsets = points - center
    # Ascending eigenvalues: variances along the least, minor and major axes.
    variances, axes = np.linalg.eigh(offsets.T @ offsets / count)
    if variances[0] <= 0:
        raise ValueError("Rescaling requires a contour that spans three dimensions")
    spread = np.sqrt(variances)
    ratio = np.array([flatness * spread[2] / spread[0], elongation * spread[2] / spread[1], 1.0])
    scale = ratio * (voxels / count / ratio.prod()) ** (1.0 / 3.0)
    # Each output voxel reads the input at center + inverse @ (x - center).
    inverse = axes @ np.diag(1.0 / scale) @ axes.T
    # No voxel ends farther from the centroid than the largest scale carries the farthest one.
    reach = max(float(scale.max()), 1.0) * float(np.linalg.norm(offsets, axis=1).max()) + 2.0
    region = tuple(
        slice(max(0, int(np.floor(c - reach))), min(n, int(np.ceil(c + reach)) + 1))
        for c, n in zip(center, mask.shape, strict=True)
    )
    local = mask[region]
    # Distances to the voxel-face boundary, positive inside the contour.
    signed = np.where(
        local,
        ndimage.distance_transform_edt(local) - 0.5,
        0.5 - ndimage.distance_transform_edt(~local),
    )
    origin = center - np.array([axis.start for axis in region], dtype=np.float64)
    field = ndimage.affine_transform(
        signed, inverse, offset=origin - inverse @ origin, order=1, mode="nearest"
    )
    kept = min(voxels, field.size)
    result = np.zeros_like(mask)
    result[region] = _largest(field >= np.partition(field.ravel(), -kept)[-kept])
    return result


def _count(field: Mask, kernel: Mask) -> NDArray[np.float64]:
    """Count kernel voxels landing on a field at every translation that keeps it in the array."""
    counts = signal.correlate(
        field.astype(np.float64), kernel.astype(np.float64), mode="valid", method="fft"
    )
    # Exact counts are integers; rounding removes FFT error.
    return np.rint(counts)


def translated_placement(
    mask: Mask,
    liver: Mask,
    *,
    surface_fraction: float | None = None,
    max_distance_mm: float | None = None,
    max_overhang: float = 0.0,
) -> tuple[Mask, tuple[int, int, int]] | None:
    """Find the nearest rigid translation that puts a contour inside the liver.

    Up to ``max_overhang`` of the contour's voxels may cross the liver surface;
    they are removed and the largest connected part is kept, which leaves the
    contour flush with the surface there. Without a surface target the nearest
    such translation is returned.

    With a surface target, overhang budgets are tried in increasing order up to
    ``max_overhang``. The first budget with a translation whose contact lies
    within one percentage point of the target returns the nearest of them;
    otherwise the closest contact found is returned. The whole 3 mm surface
    neighborhood must be observed. Contact after trimming is computed on the
    translations ranked closest by a correlation estimate.

    Returns the placed contour and its translation in voxels, or ``None`` when
    no translation fits.
    """
    points = np.argwhere(mask)
    low, high = points.min(axis=0), points.max(axis=0) + 1
    kernel = mask[tuple(slice(a, b) for a, b in zip(low, high, strict=True))]
    volume = int(kernel.sum())
    margin = 1 if surface_fraction is None else SURFACE_DISTANCE_MM
    border = np.ones_like(liver)
    border[(slice(margin, -margin),) * 3] = False
    outside = _count(~liver, kernel)
    starts = np.argwhere(_count(border, kernel) < 0.5)
    distance = np.linalg.norm(starts - low, axis=1)
    if max_distance_mm is not None:
        starts, distance = (
            starts[distance <= max_distance_mm],
            distance[distance <= max_distance_mm],
        )
    overhang = outside[tuple(starts.T)] / volume

    def place(start: NDArray[np.intp]) -> Mask:
        placed = np.zeros_like(mask)
        placed[tuple((points + (start - low)).T)] = True
        placed = _largest(placed & liver)
        if (placed & ~liver).any() or touches_face(placed):
            raise RuntimeError("Contour translation failed its exact anatomical containment check")
        return placed

    def result(start: NDArray[np.intp]) -> tuple[Mask, tuple[int, int, int]]:
        shift = start - low
        return place(start), (int(shift[0]), int(shift[1]), int(shift[2]))

    if surface_fraction is None:
        fits = overhang <= max_overhang
        if not fits.any():
            return None
        return result(starts[fits][np.argmin(distance[fits])])

    near = near_liver_surface(liver)

    def trimmed_contact(start: NDArray[np.intp]) -> float:
        """Measure the liver-surface contact of the contour trimmed at one translation.

        Equals ``liver_surface_fraction(place(start), liver)`` without building the
        full-size mask. Every start keeps the contour's 3 mm neighborhood inside
        the array, so the liver's own surface band decides each boundary voxel.
        """
        region = tuple(slice(a, a + n) for a, n in zip(start, kernel.shape, strict=True))
        local = _largest(kernel & liver[region])
        return float(near[region][local & ~ndimage.binary_erosion(local)].mean())

    boundary = kernel & ~ndimage.binary_erosion(kernel)
    # Boundary voxels near the surface or past it estimate contact after trimming.
    estimate = _count(near | ~liver, boundary)[tuple(starts.T)] / boundary.sum()
    error = np.abs(estimate - surface_fraction)
    best: tuple[float, float, NDArray[np.intp]] | None = None
    tried = np.zeros(len(starts), dtype=np.bool_)
    for budget in (b for b in SURFACE_OVERHANGS if b <= max_overhang):
        fresh = np.flatnonzero((overhang <= budget) & ~tried)
        tried[fresh] = True
        # The estimate is exact for a contour that stays wholly inside the liver.
        whole = fresh[overhang[fresh] == 0]
        matched = whole[error[whole] <= 0.01]
        found = [
            (float(error[i]), int(i))
            for i in (
                *whole[np.lexsort((distance[whole], error[whole]))][:1],
                *matched[np.argsort(distance[matched])][:1],
            )
        ]
        trimmed = fresh[overhang[fresh] > 0]
        for i in trimmed[np.lexsort((distance[trimmed], error[trimmed]))][:_EXACT_CANDIDATES]:
            found.append((abs(trimmed_contact(starts[i]) - surface_fraction), i))
        for miss, i in found:
            # Every contact within one percentage point counts as matched.
            key = (max(miss, 0.01), float(distance[i]))
            if best is None or key < best[:2]:
                best = (*key, starts[i])
        if best is not None and best[0] <= 0.01:
            break
    return None if best is None else result(best[2])
