"""Native array and dependency contracts used by the source implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from tokenizers import Encoding

from lesiondiffusion_mlx._arrays import HostArray as HostArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    import mlx.core as mx
    from medmlx_core.typing import FastRuntime as CoreFastRuntime

    type Array = mx.array
    type Dtype = mx.Dtype
    type ArrayValue = bool | int | float | complex | list[ArrayValue] | tuple[ArrayValue, ...]

    class ArrayConstructor(Protocol):
        def __call__(
            self,
            value: HostArray | Array | ArrayValue,
            /,
            dtype: Dtype | None = None,
        ) -> Array:
            """Construct a native array from values and an optional dtype."""
            ...

    class TokenizerContract(Protocol):
        def enable_truncation(self, max_length: int) -> None:
            """Enable tokenizer truncation at the requested length."""
            ...
        def enable_padding(self, *, length: int, pad_id: int, pad_token: str) -> None:
            """Configure tokenizer padding and its token ID."""
            ...
        def encode_batch(self, inputs: Sequence[str]) -> list[Encoding]:
            """Encode a batch of input strings."""
            ...

    class MetalKernel(Protocol):
        def __call__(
            self,
            *,
            inputs: Sequence[Array],
            template: Sequence[tuple[str, int | bool | Dtype]],
            grid: tuple[int, int, int],
            threadgroup: tuple[int, int, int],
            output_shapes: Sequence[Sequence[int]],
            output_dtypes: Sequence[Dtype],
        ) -> list[Array]:
            """Launch a Metal kernel and return its output arrays."""
            ...

    class Fast(CoreFastRuntime, Protocol):
        def scaled_dot_product_attention(
            self,
            q: Array,
            k: Array,
            v: Array,
            *,
            scale: float,
            mask: str | Array | None = None,
            sinks: Array | None = None,
            force_fused: bool = False,
        ) -> Array:
            """Compute native scaled dot-product attention."""
            ...

        def metal_kernel(
            self,
            name: str,
            input_names: Sequence[str],
            output_names: Sequence[str],
            source: str,
            header: str = "",
            ensure_row_contiguous: bool = True,
            atomic_outputs: bool = False,
            compile_options: object | None = None,
        ) -> MetalKernel:
            """Compile a named native Metal kernel."""
            ...

    class Random(Protocol):
        def seed(self, seed: int) -> None:
            """Set the native random generator seed."""
            ...
        def normal(self, shape: Sequence[int]) -> Array:
            """Draw normally distributed native values."""
            ...

    Mlx = Any
else:

    class ArrayConstructor(Protocol):
        pass

    class TokenizerContract(Protocol):
        pass

    class MetalKernel(Protocol):
        pass

    class Fast(Protocol):
        pass

    class Random(Protocol):
        pass

    Mlx = Any
