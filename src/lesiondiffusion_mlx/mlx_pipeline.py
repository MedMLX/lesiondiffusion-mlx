"""Array preprocessing for the pinned upstream LesionDiffusion pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, ArrayValue, HostArray, Mlx, TokenizerContract

import numpy as np
from medmlx_core.errors import InvalidInputError, ModelExecutionError

from lesiondiffusion_mlx._typing import ArrayConstructor
from lesiondiffusion_mlx.mlx_model import BiomedTextEncoder, VqAutoencoder

# Crop extents in source axis order (Z, Y, X), in voxels of the 1 mm grid: LMNet
# generates the mask on LMNET_CROP; LINet inpaints on LINET_CROP.
LMNET_CROP = (64, 152, 152)
LINET_CROP = (128, 128, 128)


def ras_to_source(array: HostArray) -> HostArray:
    """Convert a RAS-ordered ``(X, Y, Z)`` array to the source layout ``(Z, Y, X)``.

    The pinned upstream loader reorients to LAS before transposing, so the first
    (X) axis is also reversed. Expects a 3-D array and returns a C-contiguous copy.
    """
    return np.ascontiguousarray(array[::-1].transpose(2, 1, 0))


def source_to_ras(array: HostArray) -> HostArray:
    """Invert ``ras_to_source``: ``(Z, Y, X)`` LAS-ordered data back to RAS ``(X, Y, Z)``."""
    return np.ascontiguousarray(array.transpose(2, 1, 0)[::-1])


def source_crop(mask: HostArray, shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    """Return the fixed-size crop centred on a mask's bounding box, in source axes.

    Args:
        mask: Boolean or label volume in source ``(Z, Y, X)`` order; non-zero voxels
            define the extent (the box for LMNet, the lesion for LINet).
        shape: Positive crop size ``(Z, Y, X)`` in voxels, supplied by the caller.
            The helper assumes three positive sizes and does not validate them.

    Returns:
        Slices of length ``shape`` per axis, centred on the midpoint of the mask's
        extent (integer floor) and shifted to stay inside the volume.

    Raises:
        ModelExecutionError: If ``mask`` is empty.
        InvalidInputError: If any volume axis is shorter than the crop.
    """
    points = np.argwhere(mask)
    if points.size == 0:
        raise ModelExecutionError("LesionDiffusion cannot crop an empty lesion mask")
    slices: list[slice] = []
    for axis, requested in enumerate(shape):
        center = int((points[:, axis].min() + points[:, axis].max()) // 2)
        length = mask.shape[axis]
        if length < requested:
            raise InvalidInputError(
                f"Pinned LesionDiffusion crop requires source axis {axis} >= {requested}; "
                f"got {length}"
            )
        start = min(max(center - requested // 2, 0), length - requested)
        slices.append(slice(start, start + requested))
    return slices[0], slices[1], slices[2]


def piecewise_ct(array: Array, mx: Mlx, *, inverse: bool = False) -> Array:
    """Apply the source's four-knot CT intensity transform on the Metal GPU.

    Map HU landmarks (-1000, -200, 200, 1000) to (0, 0.2, 0.8, 1) with
    piecewise-linear interpolation and endpoint clipping. ``inverse=True``
    swaps those landmarks to return HU, also clipping outside its input range.
    The input array may have any shape; the result is float32 with the same shape.
    Clipping means HU outside [-1000, 1000] cannot survive a round trip.
    NaN inputs fall through all comparisons to the lower output endpoint; the
    caller must reject non-finite inputs if preserving missingness is required.
    """
    inputs, outputs = (
        ((0.0, 0.2, 0.8, 1.0), (-1000.0, -200.0, 200.0, 1000.0))
        if inverse
        else ((-1000.0, -200.0, 200.0, 1000.0), (0.0, 0.2, 0.8, 1.0))
    )
    result = mx.full(array.shape, outputs[0], dtype=mx.float32)
    for index in range(3):
        slope = (outputs[index + 1] - outputs[index]) / (inputs[index + 1] - inputs[index])
        result = mx.where(
            (array >= inputs[index]) & (array < inputs[index + 1]),
            slope * (array - inputs[index]) + outputs[index],
            result,
        )
    return mx.where(array >= inputs[-1], outputs[-1], result)


def encode_source(model: VqAutoencoder, array: Array, mx: Mlx) -> Array:
    """Encode a normalized volume ``(B, C, D, H, W)`` to its VQ latent, evaluated eagerly.

    ``array`` holds ``piecewise_ct`` output in [0, 1]. The latent has half the
    spatial size, rounded down, so decoding an odd extent returns one voxel fewer;
    the runner pads to even extents first. The latent is not quantized here;
    ``VqAutoencoder.decode`` does that.
    """
    latent = model.encode(array)
    mx.eval(latent)
    return latent


def encode_attributes(
    attributes: Mapping[str, object],
    model: BiomedTextEncoder,
    tokenizer_path: Path,
    mx: Mlx,
) -> Array:
    """Embed each attribute value as one BiomedCLIP text vector for cross-attention.

    Only the values of ``attributes`` are encoded, one sequence per entry in
    insertion order; keys are ignored. Tokenization runs on the host (truncated and
    padded to 256 tokens, pad id 0) and every learned text layer runs in MLX.

    Returns:
        Context of shape ``(1, len(attributes), D_ctx)``, in the text encoder's dtype.
    """
    from tokenizers import Tokenizer

    tokenizer = cast("TokenizerContract", Tokenizer.from_file(str(tokenizer_path)))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(length=256, pad_id=0, pad_token="[PAD]")
    tokens = tokenizer.encode_batch([str(value) for value in attributes.values()])
    # Tokenizer IDs are nested numeric lists; list invariance loses the recursive type.
    token_ids = cast(ArrayConstructor, mx.array)(
        cast("ArrayValue", [token.ids for token in tokens]), dtype=mx.int32
    )
    encoded = model(token_ids)
    mx.eval(encoded)
    return encoded[None]
