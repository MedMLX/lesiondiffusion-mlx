"""Seeded synthetic source-keyed weights shared by fixtures and replay tests.

Only weight shapes are constructed here; outputs come from pinned upstream code.
Seeds keep fixtures small, including the fixed-width 12-layer text encoder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import numpy as np

from lesiondiffusion_mlx._arrays import FloatArray, HostArray

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Mlx


class ArrayFactory[T, D](Protocol):
    @property
    def float32(self) -> D:
        """Return the runtime float32 dtype."""
        ...

    def array(self, value: HostArray, /, *, dtype: D) -> T:
        """Construct a runtime array from host values."""
        ...


class Weights:
    def __init__(self, seed: int):
        """Initialize a seeded fixture weight generator."""
        self.rng = np.random.default_rng(seed)
        self.arrays: dict[str, FloatArray] = {}

    def add(self, key: str, shape: tuple[int, ...], *, offset: float = 0) -> None:
        """Add a seeded tensor with the requested shape and offset."""
        self.arrays[key] = (
            self.rng.standard_normal(shape).astype(np.float32) * np.float32(0.02) + offset
        )

    def dense(self, key: str, inputs: int, outputs: int, *, bias: bool = True) -> None:
        """Add dense-layer weight and bias fixtures."""
        self.add(key + ".weight", (outputs, inputs))
        if bias:
            self.add(key + ".bias", (outputs,))

    def conv(self, key: str, inputs: int, outputs: int, kernel: int = 3) -> None:
        """Add convolution weight and bias fixtures."""
        self.add(key + ".weight", (outputs, inputs, kernel, kernel, kernel))
        self.add(key + ".bias", (outputs,))

    def norm(self, key: str, channels: int) -> None:
        """Add normalization scale and bias fixtures."""
        self.add(key + ".weight", (channels,), offset=1)
        self.add(key + ".bias", (channels,))

    def native[T, D](self, mx: ArrayFactory[T, D], dtype: D) -> dict[str, T]:
        """Convert fixture weights to the requested native dtype."""
        return {key: mx.array(value, dtype=dtype) for key, value in self.arrays.items()}


def unet_weights[T, D](
    seed: int, mx: ArrayFactory[T, D], dtype: D, *, skip_context: bool
) -> dict[str, T]:
    """Create seeded fixture weights for the denoising UNet."""
    w = Weights(seed)
    w.dense("time_embed.0", 32, 128)
    w.dense("time_embed.2", 128, 128)
    if not skip_context:
        w.dense("context_linear_projection", 6, 6, bias=False)

    def residual(key: str, inputs: int, outputs: int) -> None:
        """Add one residual block's fixture weights."""
        w.norm(key + ".in_layers.0", inputs)
        w.conv(key + ".in_layers.2", inputs, outputs)
        w.dense(key + ".emb_layers.1", 128, outputs)
        w.norm(key + ".out_layers.0", outputs)
        w.conv(key + ".out_layers.3", outputs, outputs)
        if inputs != outputs:
            w.conv(key + ".skip_connection", inputs, outputs, 1)

    def transformer(key: str, channels: int) -> None:
        """Add one transformer block's fixture weights."""
        w.norm(key + ".norm", channels)
        w.conv(key + ".proj_in", channels, channels, 1)
        w.conv(key + ".proj_out", channels, channels, 1)
        block = key + ".transformer_blocks.0"
        for name in ("norm1", "norm2", "norm3"):
            w.norm(block + "." + name, channels)
        for name in ("attn1", "attn2"):
            context_channels = channels if name == "attn1" or skip_context else 6
            w.dense(block + "." + name + ".to_q", channels, channels, bias=False)
            for projection in ("to_k", "to_v"):
                w.dense(
                    block + "." + name + "." + projection,
                    context_channels,
                    channels,
                    bias=False,
                )
            w.dense(block + "." + name + ".to_out.0", channels, channels)
        w.dense(block + ".ff.net.0.0", channels, channels * 2)
        w.dense(block + ".ff.net.2", channels * 2, channels)

    w.conv("input_blocks.0.0", 5 if skip_context else 4, 32)
    residual("input_blocks.1.0", 32, 32)
    transformer("input_blocks.1.1", 32)
    w.conv("input_blocks.2.0.op", 32, 32)
    residual("input_blocks.3.0", 32, 64)
    transformer("input_blocks.3.1", 64)
    residual("middle_block.0", 64, 64)
    transformer("middle_block.1", 64)
    residual("middle_block.2", 64, 64)
    for index, inputs, outputs in ((0, 128, 64), (1, 96, 64), (2, 96, 32), (3, 64, 32)):
        residual(f"output_blocks.{index}.0", inputs, outputs)
        transformer(f"output_blocks.{index}.1", outputs)
    w.conv("output_blocks.1.2.conv", 64, 64)
    w.norm("out.0", 32)
    w.conv("out.2", 32, 5 if skip_context else 1)
    return w.native(mx, dtype)


def vq_weights[T, D](seed: int, mx: ArrayFactory[T, D], dtype: D) -> dict[str, T]:
    """Create seeded fixture weights for the VQ autoencoder."""
    w = Weights(seed)

    def residual(key: str, inputs: int, outputs: int) -> None:
        """Add one VQ residual block's fixture weights."""
        w.norm(key + ".norm1", inputs)
        w.conv(key + ".conv1", inputs, outputs)
        w.norm(key + ".norm2", outputs)
        w.conv(key + ".conv2", outputs, outputs)
        if inputs != outputs:
            w.conv(key + ".nin_shortcut", inputs, outputs, 1)

    w.conv("encoder.conv_in", 1, 16)
    residual("encoder.down.0.block.0", 16, 16)
    w.conv("encoder.down.0.downsample.conv", 16, 16)
    residual("encoder.down.1.block.0", 16, 32)
    residual("encoder.down.2.block.0", 32, 32)
    for name in ("block_1", "block_2"):
        residual("encoder.mid." + name, 32, 32)
        residual("decoder.mid." + name, 32, 32)
    w.norm("encoder.norm_out", 32)
    w.conv("encoder.conv_out", 32, 4)
    w.conv("quant_conv", 4, 4, 1)
    w.add("quantize.embedding.weight", (8, 4))
    w.conv("post_quant_conv", 4, 4, 1)
    w.conv("decoder.conv_in", 4, 32)
    for level, inputs, outputs in ((2, 32, 32), (1, 32, 32), (0, 32, 16)):
        residual(f"decoder.up.{level}.block.0", inputs, outputs)
        residual(f"decoder.up.{level}.block.1", outputs, outputs)
    w.conv("decoder.up.1.upsample.conv", 32, 32)
    w.norm("decoder.norm_out", 16)
    w.conv("decoder.conv_out", 16, 1)
    return w.native(mx, dtype)


def text_weights[T, D](seed: int, mx: ArrayFactory[T, D]) -> dict[str, T]:
    """Create seeded fixture weights for the text encoder."""
    w = Weights(seed)
    prefix = "transformer.embeddings"
    w.add(prefix + ".word_embeddings.weight", (12, 768))
    w.add(prefix + ".position_embeddings.weight", (256, 768))
    w.add(prefix + ".token_type_embeddings.weight", (2, 768))
    w.norm(prefix + ".LayerNorm", 768)
    for index in range(12):
        key = f"transformer.encoder.layer.{index}"
        for name in ("query", "key", "value"):
            w.dense(key + ".attention.self." + name, 768, 768)
        w.dense(key + ".attention.output.dense", 768, 768)
        w.norm(key + ".attention.output.LayerNorm", 768)
        w.dense(key + ".intermediate.dense", 768, 16)
        w.dense(key + ".output.dense", 16, 768)
        w.norm(key + ".output.LayerNorm", 768)
    w.dense("proj.0", 768, 16, bias=False)
    w.dense("proj.2", 16, 6, bias=False)
    return w.native(mx, mx.float32)


def sampling_inputs(seed: int, *, inpainting: bool) -> dict[str, HostArray]:
    """Create seeded latent and noise arrays for diffusion fixtures."""
    rng = np.random.default_rng(seed)
    shape = (1, 4 if inpainting else 1, 2, 2, 2)
    result: dict[str, HostArray] = {
        "alphas": np.linspace(0.99, 0.1, 20, dtype=np.float32).astype(
            np.float16 if inpainting else np.float32
        ),
        "image": rng.standard_normal((1, 4 if inpainting else 3, 2, 2, 2)).astype(np.float32),
        "initial": rng.standard_normal(shape).astype(np.float32),
        "noise": rng.standard_normal((4, *shape)).astype(np.float32),
        "condition": rng.standard_normal((1, 2, 3)).astype(np.float32),
    }
    if inpainting:
        result["mask"] = np.array([0, 1, 1, 0, 1, 0, 1, 0], dtype=np.float32).reshape(1, 1, 2, 2, 2)
    return result


def sampling_denoiser(mx: Mlx) -> Callable[[Array, Array, Array | None], Array]:
    """Build a deterministic denoiser for sampling fixtures."""
    def denoiser(x: Array, timestep: Array, condition: Array | None) -> Array:
        """Predict deterministic noise from the sample and conditioning."""
        x = x[:, :1] if x.shape[1] == 4 else x
        assert condition is not None
        return (
            x * 0.125
            + timestep[:, None, None, None, None].astype(mx.float32) * 0.001
            + mx.sum(condition) * 0.01
        )

    return denoiser
