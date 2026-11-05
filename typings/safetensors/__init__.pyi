"""Typed surface of safetensors 0.8 used here; its bundled stub leaves these untyped."""

from os import PathLike
from types import TracebackType
from typing import Literal, Self

import numpy as np

class SafetensorError(Exception): ...

class safe_open:
    def __init__(
        self,
        filename: str | PathLike[str],
        framework: Literal["np"],
        device: str = ...,
        *,
        backend: str = "mmap",
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def keys(self) -> list[str]: ...
    def get_tensor(self, name: str) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]: ...
