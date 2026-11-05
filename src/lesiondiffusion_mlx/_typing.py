"""Native array and dependency contracts used by the source implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from lesiondiffusion_mlx._arrays import HostArray as HostArray

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal, overload

    import mlx.core as mx
    from medmlx_core.typing import FastRuntime as CoreFastRuntime
    from medmlx_core.typing import MlxRuntime as CoreMlxRuntime
    from tokenizers import Encoding

    type Array = mx.array
    type Dtype = mx.Dtype
    type ArrayValue = bool | int | float | complex | list[ArrayValue] | tuple[ArrayValue, ...]

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
        """The part of ``mx.fast`` used here, with typed Metal-kernel construction."""

        scaled_dot_product_attention = staticmethod(mx.fast.scaled_dot_product_attention)

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
            """Compile a named Metal kernel with the supplied source."""
            ...

    class Random(Protocol):
        normal = staticmethod(mx.random.normal)
        seed = staticmethod(mx.random.seed)

    class Mlx(CoreMlxRuntime, Protocol):
        """Native ``mlx.core`` operations injected into the model and pipeline."""

        @property
        def fast(self) -> Fast:
            """Return native fast operations and Metal-kernel construction."""
            ...

        @property
        def random(self) -> Random:
            """Return native random generation operations."""
            ...

        @overload
        def load(
            self,
            file: str,
            /,
            format: Literal["safetensors"],
            return_metadata: Literal[False] = False,
        ) -> dict[str, Array]:
            """Load a safetensors file as a name-to-array mapping."""
            ...

        @overload
        def load(
            self, file: str, /, format: str | None = None, return_metadata: bool = False
        ) -> Array | dict[str, Array] | tuple[dict[str, Array], dict[str, object]]:
            """Load arrays, and optionally metadata, from a supported file format."""
            ...

        addmm = staticmethod(mx.addmm)
        all = staticmethod(mx.all)
        argmin = staticmethod(mx.argmin)
        clear_cache = staticmethod(mx.clear_cache)
        compile = staticmethod(mx.compile)
        contiguous = staticmethod(mx.contiguous)
        cos = staticmethod(mx.cos)
        device_info = staticmethod(mx.device_info)
        erf = staticmethod(mx.erf)
        exp = staticmethod(mx.exp)
        full = staticmethod(mx.full)
        isfinite = staticmethod(mx.isfinite)
        reciprocal = staticmethod(mx.reciprocal)
        sin = staticmethod(mx.sin)
        softmax = staticmethod(mx.softmax)
        sqrt = staticmethod(mx.sqrt)
        sum = staticmethod(mx.sum)


# Defined at runtime because callers pass it to ``typing.cast`` unquoted.
class ArrayConstructor(Protocol):
    def __call__(
        self,
        value: HostArray | Array | ArrayValue,
        /,
        dtype: Dtype | None = None,
    ) -> Array:
        """Construct a native array from values and an optional dtype."""
        ...
