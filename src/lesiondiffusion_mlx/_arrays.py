"""Host arrays shared by conversion, NIfTI I/O and reference tooling."""

import numpy as np

type HostArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]
type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
type BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]
type NumericScalar = (
    np.bool_
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.float16
    | np.float32
    | np.float64
    | np.complex64
    | np.complex128
)
type NumericArray = np.ndarray[tuple[int, ...], np.dtype[NumericScalar]]
