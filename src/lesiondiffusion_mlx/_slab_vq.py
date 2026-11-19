"""Run the conditioned VQ on Metal with file-backed intermediate activations."""

from __future__ import annotations

import math
import shutil
from contextlib import ExitStack
from functools import lru_cache
from io import FileIO
from tempfile import TemporaryFile, gettempdir
from typing import TYPE_CHECKING, Self, cast
from weakref import WeakSet, finalize

import numpy as np
from medmlx_core.errors import InvalidInputError, ModelExecutionError
from medmlx_core.group_norm import group_norm_affine
from medmlx_core.ops import group_norm_ncdhw

from lesiondiffusion_mlx._nax_conv import nax_conv3d
from lesiondiffusion_mlx._source_half import group_norm_half
from lesiondiffusion_mlx._typing import ArrayConstructor
from lesiondiffusion_mlx.mlx_model import VqAutoencoder, vq_decode_layers, vq_encode_layers

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import TracebackType

    from lesiondiffusion_mlx._arrays import HostArray
    from lesiondiffusion_mlx._typing import Array, Dtype, MetalKernel, Mlx

_SLAB_VOXELS = 1 << 20
# One trained GroupNorm group holds six channels. Keep its FP32 read below
# 2 GiB, including on hosts whose file I/O requires signed 32-bit read sizes.
_GROUP_BYTES_LIMIT = 2 << 30
_VQ_CHANNELS = 96
_VQ_GROUPS = 16


def require_slab_capacity(shape: tuple[int, ...]) -> None:
    """Check the padded RAS shape and storage for the three live FP32 features."""
    voxels = math.prod(shape)
    feature_bytes = voxels * _VQ_CHANNELS * np.dtype(np.float32).itemsize
    if shape[0] * shape[1] > _SLAB_VOXELS or feature_bytes // _VQ_GROUPS >= _GROUP_BYTES_LIMIT:
        raise InvalidInputError(
            "LesionDiffusion slab VQ requires at most 1,048,576 voxels per axial plane "
            "and less than 2 GiB per FP32 normalization group"
        )
    required = 3 * feature_bytes
    if shutil.disk_usage(gettempdir()).free < required:
        raise ModelExecutionError(
            f"LesionDiffusion slab VQ needs {required / 1e9:.1f} GB of free temporary storage. "
            "Set TMPDIR to a volume with enough free space."
        )


class _Volume:
    """An unbuffered temporary file in channel-major D, H, W order, batch size one."""

    def __init__(
        self, file: FileIO, shape: tuple[int, int, int, int, int], dtype: np.dtype[np.floating]
    ) -> None:
        self.file = file
        _, self.channels, self.depth, self.height, self.width = shape
        self.dtype = dtype
        self.plane_bytes = self.height * self.width * dtype.itemsize

    def _transfer(
        self,
        data: HostArray,
        start: int,
        channel: int,
        operation: Callable[[memoryview], int | None],
    ) -> None:
        """Transfer adjacent groups in one call, or one depth interval per channel."""
        def transfer(block: HostArray, first_plane: int) -> None:
            buffer = memoryview(block).cast("B")
            self.file.seek(first_plane * self.plane_bytes)
            if operation(buffer) != buffer.nbytes:
                raise OSError("Incomplete VQ temporary activation transfer")

        if data.shape[2] == self.depth:
            transfer(data, channel * self.depth)
        else:
            for index in range(data.shape[1]):
                transfer(data[0, index], (channel + index) * self.depth + start)

    def read(self, start: int, stop: int, channel: int = 0, count: int | None = None) -> HostArray:
        """Read an axial slab or complete adjacent channels into one host buffer."""
        channels = self.channels if count is None else count
        out = np.empty((1, channels, stop - start, self.height, self.width), dtype=self.dtype)
        self._transfer(out, start, channel, self.file.readinto)
        return out

    def write(self, data: Array, start: int = 0, channel: int = 0) -> None:
        """Export an evaluated MLX slab to disk without changing its dtype."""
        values = np.ascontiguousarray(cast("HostArray", np.asarray(data)), dtype=self.dtype)
        self._transfer(values, start, channel, self.file.write)


# Keep the same FP32 affine and precise SiLU as medmlx-core's native GroupNorm.
# A depth halo can have an unaligned spatial size even when the full group is aligned.
_APPLY_SILU = r"""
const uint i = thread_position_in_grid.x, c = thread_position_in_grid.y;
const float scale = affine[c * 2], offset = affine[c * 2 + 1];
if constexpr (SPATIAL % 4 == 0) {
    if (i * 4 >= SPATIAL) return;
    const ulong index = ulong(c) * SPATIAL + i * 4;
    float4 y = float4(*reinterpret_cast<const device vec<T, 4>*>(x + index)) * scale + offset;
    y = y / (1.0f + metal::precise::exp(-y));
    *reinterpret_cast<device vec<T, 4>*>(out + index) = vec<T, 4>(y);
} else {
    if (i >= SPATIAL) return;
    const ulong index = ulong(c) * SPATIAL + i;
    const float y = float(x[index]) * scale + offset;
    out[index] = T(y / (1.0f + metal::precise::exp(-y)));
}
"""


@lru_cache(maxsize=1)
def _apply_kernel(mx: Mlx) -> MetalKernel:
    return mx.fast.metal_kernel(
        name="lesion_vq_slab_group_silu",
        input_names=["x", "affine"],
        output_names=["out"],
        source=_APPLY_SILU,
    )


class SlabVqAutoencoder(VqAutoencoder):
    """Conditioned VQ with whole-grid GroupNorm and layer-local depth halos."""

    def encode(self, x: Array) -> Array:
        """Encode all voxels, retaining only the four-channel latent in MLX."""
        with _SlabLayers(self, x.dtype) as layers:
            return layers.load(vq_encode_layers(layers, layers.store(x)))

    def decode(self, z: Array, *, quantize: bool = True) -> Array:
        """Use the reference codebook search and reconstruct the entire CT in slabs."""
        z = self.quantize(z) if quantize else z
        with _SlabLayers(self, z.dtype) as layers:
            return layers.load(vq_decode_layers(layers, layers.store(z)))

    def convolve_slab(
        self, x: Array, key: str, *, padding: int, affine: Array | None, residual: Array | None
    ) -> Array:
        """Run the reference convolution with optional global normalization and skip."""
        if affine is not None:
            packed = self._packed_weight(key)
            if packed is not None:
                # These VQ convolutions have padding one and a learned bias.
                return self._evaluated(nax_conv3d(
                    x, packed, self.weights[key + ".bias"], affine=affine, silu=True,
                    residual=residual, mx=self.mx,
                ))
            spatial = math.prod(x.shape[2:])
            channels = int(x.shape[1])
            x = _apply_kernel(self.mx)(
                inputs=[x, affine],
                template=[("T", x.dtype), ("SPATIAL", spatial)],
                grid=(spatial // 4 if spatial % 4 == 0 else spatial, channels, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[tuple(x.shape)],
                output_dtypes=[x.dtype],
            )[0]
        y = self.conv(x, key, padding=padding)
        return y if residual is None else self._evaluated(residual + y)


class _SlabLayers:
    """Stage activations on disk and run model arithmetic on the active Metal stream."""

    def __init__(self, model: SlabVqAutoencoder, dtype: Dtype) -> None:
        self.model = model
        self.mx = model.mx
        self.dtype = np.dtype(np.float16 if dtype == self.mx.float16 else np.float32)
        self._files: WeakSet[ExitStack] = WeakSet()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Also close volumes still referenced by an exception traceback.
        for lifetime in self._files:
            lifetime.close()
        if isinstance(exc, OSError):
            raise ModelExecutionError(
                f"LesionDiffusion slab VQ temporary storage failed: {exc}. Check TMPDIR."
            ) from exc

    def _empty(self, channels: int, depth: int, height: int, width: int) -> _Volume:
        with ExitStack() as opening:
            file = opening.enter_context(TemporaryFile(buffering=0))
            volume = _Volume(file, (1, channels, depth, height, width), self.dtype)
            lifetime = opening.pop_all()
        # Release completed layer files as soon as no later operation reads them.
        finalize(volume, lifetime.close)
        self._files.add(lifetime)
        return volume

    def store(self, x: Array) -> _Volume:
        """Stage the low-channel CT or quantized latent at the network boundary."""
        _, channels, depth, height, width = x.shape
        volume = self._empty(channels, depth, height, width)
        volume.write(x)
        return volume

    def load(self, x: _Volume) -> Array:
        """Collect a small-channel VQ input or output on the native device."""
        return cast(ArrayConstructor, self.mx.array)(x.read(0, x.depth))

    @staticmethod
    def _spans(depth: int, height: int, width: int, *, scale: int = 1) -> Iterator[tuple[int, int]]:
        step = max(1, _SLAB_VOXELS // (height * width * scale))
        for start in range(0, depth, step):
            yield start, min(start + step, depth)

    def conv(
        self, x: _Volume, key: str, *, padding: int = 1,
        affine: Array | None = None, residual: _Volume | None = None,
    ) -> _Volume:
        """Convolve each slab with neighboring planes, cropping only the halo outputs."""
        array = cast(ArrayConstructor, self.mx.array)
        channels = self.model.weights[key + ".weight"].shape[0]
        out = self._empty(channels, x.depth, x.height, x.width)
        for start, stop in self._spans(x.depth, x.height, x.width):
            lo, hi = max(0, start - padding), min(x.depth, stop + padding)
            tile = array(x.read(lo, hi))
            skip = array(residual.read(lo, hi)) if residual is not None else None
            y = self.model.convolve_slab(tile, key, padding=padding, affine=affine, residual=skip)
            out.write(y[:, :, start - lo : stop - lo], start)
            del tile, skip, y
        return out

    def downsample(self, x: _Volume, key: str) -> _Volume:
        """Pair output plane i with input planes 2i through 2i+2, including the end pad."""
        mx = self.mx
        depth, height, width = x.depth // 2, x.height // 2, x.width // 2
        out = self._empty(self.model.weights[key + ".weight"].shape[0], depth, height, width)
        for start, stop in self._spans(depth, x.height, x.width, scale=2):
            end = min(x.depth, 2 * stop + 1)
            tile = cast(ArrayConstructor, mx.array)(x.read(2 * start, end))
            tile = mx.pad(tile, [(0, 0), (0, 0), (0, 2 * stop + 1 - end), (0, 1), (0, 1)])
            out.write(self.model.conv(tile, key, padding=0, stride=2), start)
            del tile
        return out

    def upsample_conv(self, x: _Volume, key: str) -> _Volume:
        """Upsample whole pairs, retaining the nearest-neighbor phase across slabs."""
        depth, height, width = 2 * x.depth, 2 * x.height, 2 * x.width
        out = self._empty(self.model.weights[key + ".weight"].shape[0], depth, height, width)
        for start, stop in self._spans(x.depth, height, width, scale=2):
            lo, hi = max(0, start - 1), min(x.depth, stop + 1)
            tile = cast(ArrayConstructor, self.mx.array)(x.read(lo, hi))
            y = self.model.upsample_conv(tile, key)
            out.write(y[:, :, 2 * (start - lo) : 2 * (stop - lo)], 2 * start)
            del tile, y
        return out

    def norm_silu_conv(
        self, x: _Volume, norm_key: str, conv_key: str, *, groups: int, eps: float,
        residual: _Volume | None = None,
    ) -> _Volume:
        """Use the full spatial extent of each group before applying slab convolutions."""
        mx = self.mx
        weight = self.model.weights[norm_key + ".weight"]
        bias = self.model.weights[norm_key + ".bias"]
        width = x.channels // groups
        native = (x.depth * x.height * x.width) % 4 == 0
        normalized = None if native else self._empty(x.channels, x.depth, x.height, x.width)
        affines: list[Array] = []
        for start in range(0, x.channels, width):
            tile = cast(ArrayConstructor, mx.array)(x.read(0, x.depth, start, width))
            w, b = weight[start : start + width], bias[start : start + width]
            if normalized is None:
                affine = group_norm_affine(tile, w, b, groups=1, eps=eps, mx=mx)
                mx.eval(affine)
                affines.append(affine)
            else:
                # The whole-volume path also uses these kernels for an unaligned latent grid.
                norm = (
                    group_norm_half(tile, w, b, groups=1, eps=eps, mx=mx)
                    if tile.dtype == mx.float16
                    else group_norm_ncdhw(tile, w, b, num_groups=1, eps=eps, mx=mx)
                )
                normalized.write(self.model.silu(norm), channel=start)
                del norm
            del tile
        affine = mx.concatenate(affines, axis=0) if normalized is None else None
        return self.conv(
            normalized if normalized is not None else x, conv_key, affine=affine, residual=residual
        )

    def residual(self, x: _Volume, key: str) -> _Volume:
        """Keep the input until both convolutions have computed their output plus the skip."""
        h = self.norm_silu_conv(x, key + ".norm1", key + ".conv1", groups=16, eps=1e-6)
        if key + ".nin_shortcut.weight" in self.model.weights:
            x = self.conv(x, key + ".nin_shortcut", padding=0)
        return self.norm_silu_conv(
            h, key + ".norm2", key + ".conv2", groups=16, eps=1e-6, residual=x,
        )
