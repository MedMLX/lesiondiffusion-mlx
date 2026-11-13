"""Real HCC-TACE texture on the research tests' controlled 128-cube grid."""

import os
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


@lru_cache(maxsize=1)
def _source_cube() -> NDArray[np.float32]:
    """Resize a CT crop with nearest neighbors, retaining measured HU values."""
    path = Path(
        os.environ.get(
            "MEDMLX_TEST_CT",
            str(Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz"),
        )
    )
    source = nib.Nifti1Image.from_filename(str(path))
    values = np.asarray(source.dataobj, dtype=np.float32)[100:228, 210:338, :128]
    factors = tuple(128 / length for length in values.shape)
    return np.asarray(ndimage.zoom(values, factors, order=0, prefilter=False), dtype=np.float32)


def ct_cube() -> NDArray[np.float32]:
    """Return independent pixels for a controlled-grid test, with no shared mutations."""
    return _source_cube().copy()
