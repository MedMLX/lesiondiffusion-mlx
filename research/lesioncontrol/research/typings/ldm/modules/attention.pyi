from collections.abc import Callable, Sequence

from torch import Tensor

def checkpoint(
    func: Callable[..., Tensor],
    inputs: Sequence[Tensor | None],
    params: Sequence[Tensor],
    flag: bool,
) -> Tensor: ...
