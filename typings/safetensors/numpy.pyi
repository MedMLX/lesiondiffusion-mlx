"""Typed NumPy writer used by optional host-only benchmark captures."""

from os import PathLike

import numpy as np

def save_file(
    tensor_dict: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
    filename: str | PathLike[str],
    metadata: dict[str, str] | None = None,
) -> None: ...
