"""Used MLX 0.32.4 neural-network APIs, checked against installed Python source."""

from collections.abc import Callable

from mlx.core import array

type ArrayTree = array | dict[str, ArrayTree] | list[ArrayTree] | tuple[ArrayTree, ...]
type Parameters = dict[str, ArrayTree]

class Module:
    def __init__(self) -> None: ...
    def parameters(self) -> Parameters: ...
    def trainable_parameters(self) -> Parameters: ...

class Linear(Module):
    weight: array
    bias: array
    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None: ...
    def __call__(self, x: array) -> array: ...

def silu(x: array) -> array: ...
def value_and_grad[**P, R](
    model: Module, fn: Callable[P, R]
) -> Callable[P, tuple[R, Parameters]]: ...
