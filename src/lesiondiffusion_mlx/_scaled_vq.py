"""Scaled FP16 VQ decoding with FP32 nearest-codebook selection.

After upsampling, residual activations use one quarter of their original scale
to bring the known upsampling range inside half storage. GroupNorm on
scaled inputs uses s² epsilon, preserving its normalized value. Convolution
branches return to the scaled representation before adding a scaled residual;
the final normalization removes the representation scale. Half weights,
normalized convolution inputs and stored activations introduce rounding.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from medmlx_core.group_norm import group_norm_affine

from lesiondiffusion_mlx._nax_conv import nax_available, nax_conv3d
from lesiondiffusion_mlx.mlx_model import VqAutoencoder

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Mlx

_SCALE = 0.25


class ScaledVqDecoder(VqAutoencoder):
    """Decode with half convolution weights and an unchanged FP32 codebook.

    ``weights`` are the existing FP16 VQ weights; ``codebook`` is the original
    FP32 embedding matrix. The selected code vectors round to half only after
    FP32 quantization, retaining reference nearest-neighbor selection.
    """

    def __init__(self, weights: Mapping[str, Array], codebook: Array, mx: Mlx):
        """Require enabled NAX and retain the FP32 codebook with half network weights."""
        if not nax_available(mx):
            raise ValueError("Scaled FP16 VQ decoding requires enabled NAX")
        if codebook.dtype != mx.float32:
            raise ValueError("Scaled FP16 VQ decoding requires the original FP32 codebook")
        super().__init__(
            {**weights, "quantize.embedding.weight": codebook}, mx, source_arithmetic=False
        )

    def decode(self, z: Array, *, quantize: bool = True) -> Array:
        """Quantize in FP32, then decode the selected latent using scaled half activations."""
        if quantize:
            z = self.quantize(z.astype(self.mx.float32))
        return super().decode(z.astype(self.mx.float16), quantize=False)

    def upsample_conv(self, x: Array, key: str) -> Array:
        """Scale the trained upsampling convolution before its first half store."""
        if key != "decoder.up.1.upsample.conv":
            raise ValueError("Scaled FP16 VQ decoding requires the trained decoder layout")
        packed = self._packed_weight(key)
        if packed is None:
            raise ValueError("Scaled FP16 VQ decoding requires packable convolution weights")
        return self._evaluated(
            nax_conv3d(
                x,
                packed,
                self.weights[key + ".bias"],
                upsample=True,
                output_scale=_SCALE,
                mx=self.mx,
            )
        )

    def norm_silu_conv(
        self,
        x: Array,
        norm_key: str,
        conv_key: str,
        *,
        groups: int,
        eps: float,
        shift: Array | None = None,
        residual: Array | None = None,
    ) -> Array:
        """Normalize scaled inputs and scale only post-upsampling residual branches."""
        scaled_input = norm_key == "decoder.norm_out" or (
            norm_key.startswith("decoder.up.0.block.") and norm_key.endswith(".norm1")
        )
        if scaled_input:
            eps *= _SCALE**2
        scaled_branch = conv_key.startswith("decoder.up.0.block.") and conv_key.endswith(".conv2")
        if not scaled_branch:
            return super().norm_silu_conv(
                x, norm_key, conv_key, groups=groups, eps=eps, shift=shift, residual=residual
            )
        mx = self.mx
        x = mx.contiguous(x)
        affine = group_norm_affine(
            x,
            self.weights[norm_key + ".weight"],
            self.weights[norm_key + ".bias"],
            groups=groups,
            eps=eps,
            mx=mx,
        )
        bias = self.weights[conv_key + ".bias"]
        if shift is not None:
            bias = bias.astype(mx.float32) + shift.astype(mx.float32)
        packed = self._packed_weight(conv_key)
        if packed is None:
            raise ValueError("Scaled FP16 VQ decoding requires packable convolution weights")
        return self._evaluated(
            nax_conv3d(
                x,
                packed,
                bias,
                affine=affine,
                silu=True,
                residual=residual,
                output_scale=_SCALE,
                mx=mx,
            )
        )
