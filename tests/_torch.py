"""Narrow reference-only adapter for Torch's incomplete ``from_numpy`` annotation."""

from collections.abc import Callable
from typing import Protocol, cast

import torch

from lesiondiffusion_mlx._arrays import HostArray


# This is the original Torch function, with its documented argument and result. It
# never enters the package's inference or NumPy-only conversion path.
class TorchSerialization(Protocol):
    def from_numpy(self, array: HostArray) -> torch.Tensor:
        """Create a reference tensor from a host array."""
        ...


from_numpy: Callable[[HostArray], torch.Tensor] = cast(TorchSerialization, torch).from_numpy
