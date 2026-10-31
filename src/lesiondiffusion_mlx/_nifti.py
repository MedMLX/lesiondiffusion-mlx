"""The NIfTI surface used here, verified against NiBabel's image/header APIs."""

from os import PathLike
from typing import Literal, Protocol, cast

import nibabel
import numpy as np
from nibabel.filebasedimages import ImageFileError as ImageFileError

from lesiondiffusion_mlx._arrays import HostArray

type PathArg = str | PathLike[str]
type SpatialAffine = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


class ArrayBuffer(Protocol):
    def __array__(
        self, dtype: np.dtype[np.generic] | None = None, copy: bool | None = None
    ) -> HostArray:
        """Convert the image data buffer to a host array."""
        ...


class NiftiHeader(Protocol):
    def get_zooms(self) -> tuple[float, ...]:
        """Return voxel spacing from the header."""
        ...
    def get_xyzt_units(self) -> tuple[str, str]:
        """Return spatial and temporal unit names."""
        ...
    def set_xyzt_units(self, xyz: str, t: str) -> None:
        """Set spatial and temporal unit names."""
        ...
    def set_data_dtype(self, dtype: type[np.generic]) -> None:
        """Set the stored image dtype."""
        ...


class HeaderFactory(Protocol):
    def from_header(self, header: NiftiHeader) -> NiftiHeader:
        """Construct a NIfTI header from an existing header."""
        ...


class NiftiImage(Protocol):
    shape: tuple[int, ...]
    affine: SpatialAffine
    dataobj: ArrayBuffer | HostArray
    header: NiftiHeader

    def get_qform(self, *, coded: Literal[True]) -> tuple[SpatialAffine | None, int]:
        """Return the qform affine and its code."""
        ...
    def get_sform(self, *, coded: Literal[True]) -> tuple[SpatialAffine | None, int]:
        """Return the sform affine and its code."""
        ...
    def set_qform(self, affine: HostArray, code: int) -> None:
        """Set the qform affine and code."""
        ...
    def set_sform(self, affine: HostArray, code: int) -> None:
        """Set the sform affine and code."""
        ...


class ImageFactory(Protocol):
    def __call__(
        self, dataobj: HostArray, affine: HostArray, header: NiftiHeader | None = None
    ) -> NiftiImage:
        """Construct a NIfTI image from data, affine, and header."""
        ...


class OrientationFunctions(Protocol):
    """The subset of ``nibabel.orientations`` used to resample to RAS 1 mm."""

    def axcodes2ornt(self, axcodes: tuple[str, str, str]) -> HostArray:
        """Convert axis codes to an orientation transform."""
        ...
    def ornt_transform(self, start_ornt: HostArray, end_ornt: HostArray) -> HostArray:
        """Compute the transform between two orientations."""
        ...
    def apply_orientation(self, data: HostArray, ornt: HostArray) -> HostArray:
        """Apply an orientation transform to image data."""
        ...
    def inv_ornt_aff(self, ornt: HostArray, shape: tuple[int, ...]) -> SpatialAffine:
        """Return the inverse voxel affine for an orientation and shape."""
        ...


class Nibabel(Protocol):
    Nifti1Header: HeaderFactory
    Nifti1Image: ImageFactory
    orientations: OrientationFunctions

    def load(self, filename: PathArg) -> NiftiImage:
        """Load an image from a path."""
        ...
    def save(self, image: NiftiImage, filename: PathArg) -> None:
        """Write a NIfTI image to a path."""
        ...
    def aff2axcodes(self, affine: HostArray) -> tuple[str | None, ...]:
        """Derive axis codes from an affine."""
        ...
    def io_orientation(self, affine: HostArray) -> HostArray:
        """Derive voxel-axis orientation from an affine."""
        ...


# This adapter is used only for .nii/.nii.gz inputs and NIfTI output; the wider
# NiBabel loader supports unrelated image classes outside the runner contract.
nib = cast(Nibabel, nibabel)
