"""Narrow reference-only adapters for incomplete Torch NumPy/save annotations."""

from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Protocol, cast

import torch

from lesiondiffusion_mlx._arrays import HostArray


# These are the original Torch functions, with their documented arguments and
# results. They never enter the package's inference or NumPy-only conversion path.
class TorchSerialization(Protocol):
    def from_numpy(self, array: HostArray) -> torch.Tensor:
        """Create a reference tensor from a host array."""
        ...
    def save(self, obj: object, f: Path | str | BinaryIO) -> None:
        """Serialize a reference object to a path or stream."""
        ...


_torch = cast(TorchSerialization, torch)
from_numpy: Callable[[HostArray], torch.Tensor] = _torch.from_numpy
save: Callable[[object, Path | str | BinaryIO], None] = _torch.save
