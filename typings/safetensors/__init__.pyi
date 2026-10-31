from os import PathLike
from types import TracebackType
from typing import Self

class SafetensorError(Exception): ...

class safe_open:
    def __init__(self, filename: str | PathLike[str], framework: str) -> None:
        """Open a safetensors file for the requested framework."""
        ...
    def __enter__(self) -> Self:
        """Enter the safetensors reader context."""
        ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the safetensors reader context."""
        ...
    def keys(self) -> list[str]:
        """List tensor names in the opened file."""
        ...
