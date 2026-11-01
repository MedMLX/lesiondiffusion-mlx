"""Native Metal networks for LesionDiffusion source 880736e2.

Weights retain their source names/layouts. Shared MedMLX array primitives own
the NCDHW to MLX convolution boundary; the architecture here is the upstream
3-D LDM architecture, including its distinct normalization and attention rules.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Mlx

from medmlx_core.group_norm import (
    group_norm_affine,
    native_group_norm,
    native_group_norm_eligible,
)
from medmlx_core.ops import (
    conv3d_ncdhw,
    group_norm_ncdhw,
    linear,
    silu,
)

from lesiondiffusion_mlx._attention import flash_attention, flash_eligible
from lesiondiffusion_mlx._nax_conv import (
    nax_available,
    nax_conv3d,
    nax_eligible,
    pack_upsample_weight,
    pack_weight,
)
from lesiondiffusion_mlx._source_half import conv3d_half, group_norm_half
from lesiondiffusion_mlx._spatial import upsample_nearest2
from lesiondiffusion_mlx._typing import ArrayConstructor


@dataclass(frozen=True)
class UnetConfig:
    """Architecture of one upstream 3-D UNet (LMNet or LINet).

    Attributes:
        model_channels: Base channel width; also sets the sinusoidal time-embedding size.
        num_res_blocks: Residual blocks per resolution level.
        attention_resolutions: Downsampling factors (1, 2, 4, ...) at which residual
            blocks are followed by a spatial transformer. The middle block always has one.
        channel_mult: Per-level multipliers of ``model_channels``; each level after
            the first halves the spatial size.
        num_head_channels: Channels per attention head.
        skip_context: Drop the text context, so cross-attention layers attend to
            their own input (upstream's ``skip_context``).
    """

    model_channels: int
    num_res_blocks: int
    attention_resolutions: tuple[int, ...]
    channel_mult: tuple[int, ...] = (1, 2, 4)
    num_head_channels: int = 32
    skip_context: bool = False


LMNET_CONFIG = UnetConfig(16, 1, (4,))
# skip_context=True follows the pinned inference YAML; the runner overrides it per
# inference mode (context is enabled only in "conditioned" mode).
LINET_CONFIG = UnetConfig(128, 2, (4, 8), skip_context=True)

# Convolutions applied only to a nearest 2x upsampling: the UNet decoder's
# ``Upsample.conv`` and the VQ decoder's ``upsample.conv``.
_UPSAMPLE_CONV = re.compile(r"(?:^output_blocks\.\d+\.\d+|\.upsample)\.conv$")


class SourceLayers:
    """Source-keyed native MLX layers without a host execution path.

    ``weights`` maps upstream state-dict names (such as ``input_blocks.1.0.in_layers.2.weight``)
    to MLX arrays in upstream's NCDHW layout. With ``source_arithmetic=True``, float16
    convolutions use the Torch-CPU-ordered Metal kernel (``conv3d_half``); otherwise
    they use the shared MedMLX convolution.

    ``native_kernels`` (default: ``not source_arithmetic``) selects conditioned
    mode's faster kernels, which keep FP32 accumulation but not source operation
    order: NAX 3x3x3 convolution on M5-class GPUs (FP16, or near-FP32 exact mode
    for FP32 weights), pointwise convolution as a matmul, and single-pass group
    normalization with fused SiLU. Without it, float16 group normalization uses
    the source-ordered kernel.
    """

    def __init__(
        self,
        weights: Mapping[str, Array],
        mx: Mlx,
        *,
        source_arithmetic: bool = True,
        native_kernels: bool | None = None,
    ):
        """Initialize source-keyed weights and the selected native arithmetic."""
        self.weights = weights
        self.mx = mx
        self.source_arithmetic = source_arithmetic
        self.native_kernels = not source_arithmetic if native_kernels is None else native_kernels
        # Native 3x3x3 weights, prepacked for the NAX kernel. Packing here
        # keeps the reorder out of compiled denoisers, which would otherwise
        # repeat it on every call. Costs one copy of those weights (two FP16
        # planes for FP32). Convolutions that only ever follow a nearest 2x
        # upsampling get the folded ``(8, 8 * C, O)`` parity form instead; see
        # ``upsample_conv``. Source arithmetic governs only FP16 operations, so
        # FP32 weights are packed whenever native kernels are on.
        self.nax_weights: dict[str, Array] = {}
        if self.native_kernels and nax_available(mx):
            packed_dtypes = (mx.float32,) if source_arithmetic else (mx.float16, mx.float32)
            for name, weight in weights.items():
                if (
                    name.endswith(".weight")
                    and weight.dtype in packed_dtypes
                    and nax_eligible(weight)
                ):
                    key = name.removesuffix(".weight")
                    pack = pack_upsample_weight if _UPSAMPLE_CONV.search(key) else pack_weight
                    self.nax_weights[key] = pack(weight, mx)
            mx.eval(*self.nax_weights.values())

    def conv(self, x: Array, key: str, *, padding: int = 1, stride: int = 1) -> Array:
        """Apply a source-keyed 3-D convolution to channels-first ``(B, C, D, H, W)``.

        ``padding`` and ``stride`` apply equally to each spatial axis; bias is
        optional. The class-level arithmetic selection controls the FP16 path.
        Native padding-1 3x3x3 convolutions (stride 1 or 2) with a prepacked
        weight of the input's dtype use the NAX kernel, which adds bias before
        its single FP16 rounding, or in exact FP32 mode before its FP32 store.
        """
        mx = self.mx
        weight, bias = self.weights[key + ".weight"], self.weights.get(key + ".bias")
        if x.dtype == mx.float16 and self.source_arithmetic:
            return conv3d_half(x, weight, bias, padding=padding, stride=stride, mx=mx)
        packed = self.nax_weights.get(key)
        if (
            packed is not None
            and packed.ndim == 2
            and x.dtype == weight.dtype
            and padding == 1
            and stride in (1, 2)
        ):
            if bias is None:
                bias = mx.zeros((int(weight.shape[0]),), dtype=x.dtype)
            return nax_conv3d(x, packed, bias, stride=stride, mx=mx)
        if self.native_kernels and weight.shape[2:] == (1, 1, 1) and padding == 0 and stride == 1:
            # A pointwise NCDHW convolution is W (O, C) @ x (C, DHW) per batch
            # item, which needs no layout transposes. Like conv3d, the product
            # is rounded to the input dtype before the separate bias add, and
            # FP32 follows the same MLX_ENABLE_TF32 policy as conv3d.
            batch, channels = int(x.shape[0]), int(x.shape[1])
            output = mx.reshape(weight, tuple(weight.shape[:2])) @ mx.reshape(
                x, (batch, channels, -1)
            )
            if bias is not None:
                output = output + bias[:, None]
            return mx.reshape(output, (batch, int(weight.shape[0]), *x.shape[2:]))
        return conv3d_ncdhw(
            x,
            weight,
            bias,
            padding=padding,
            stride=stride,
            mx=mx,
        )

    def dense(self, x: Array, key: str) -> Array:
        """Apply the source-keyed linear layer on the last axis, returning ``x``'s dtype.

        FP16 inputs and weights are promoted to FP32 for the affine and rounded
        once back to FP16, including the optional bias. Native kernels instead
        run an FP16 GEMM, which accumulates and adds the bias in FP32 before the
        same single rounding, without the FP32 matmul.
        """
        mx = self.mx
        weight, bias = self.weights[key + ".weight"], self.weights.get(key + ".bias")
        if self.native_kernels and x.dtype == mx.float16 and weight.dtype == mx.float16:
            if bias is None:
                return x @ mx.transpose(weight)
            return mx.addmm(bias, x, mx.transpose(weight))
        dtype = x.dtype
        x, weight, bias = self._affine_inputs(x, key)
        return linear(x, weight, bias, self.mx).astype(dtype)

    def _affine_inputs(self, x: Array, key: str) -> tuple[Array, Array, Array | None]:
        """Promote half inputs and weights for source-compatible affine accumulation."""
        weight = self.weights[key + ".weight"]
        bias = self.weights.get(key + ".bias")
        if x.dtype == self.mx.float16:
            # Torch's CPU half kernels add bias to the FP32 accumulator before
            # rounding once. A half convolution/matmul followed by bias rounds twice.
            x, weight = x.astype(self.mx.float32), weight.astype(self.mx.float32)
            if bias is not None:
                bias = bias.astype(self.mx.float32)
        return x, weight, bias

    def norm(self, x: Array, key: str, *, groups: int, eps: float) -> Array:
        """Apply affine group normalization to NCDHW input, returning its dtype.

        ``groups`` must divide the channel count; ``eps`` is the variance epsilon.
        Both scale and bias must be present under ``key``.
        """
        if self.native_kernels and native_group_norm_eligible(x, groups):
            return self._native_norm(x, key, groups=groups, eps=eps, silu=False)
        if x.dtype == self.mx.float16:
            return group_norm_half(
                x,
                self.weights[key + ".weight"],
                self.weights[key + ".bias"],
                groups=groups,
                eps=eps,
                mx=self.mx,
            )
        # Torch GroupNorm accumulates statistics and applies its affine in
        # float32 before rounding the result back to the input precision.
        return group_norm_ncdhw(
            x.astype(self.mx.float32),
            self.weights[key + ".weight"].astype(self.mx.float32),
            self.weights[key + ".bias"].astype(self.mx.float32),
            num_groups=groups,
            eps=eps,
            mx=self.mx,
        ).astype(x.dtype)

    def norm_silu(self, x: Array, key: str, *, groups: int, eps: float) -> Array:
        """Group-normalize, then apply this class's SiLU; native kernels fuse both.

        The fused form applies SiLU to the FP32 normalized value and rounds once,
        instead of rounding the normalized value and then the activation.
        """
        if self.native_kernels and native_group_norm_eligible(x, groups):
            return self._native_norm(x, key, groups=groups, eps=eps, silu=True)
        return self.silu(self.norm(x, key, groups=groups, eps=eps))

    def upsample_conv(self, x: Array, key: str) -> Array:
        """Nearest 2x upsampling of NCDHW ``x``, then the padding-1 conv ``key``.

        With a folded NAX weight the upsampled tensor is never formed: each
        output parity is a 2x2x2 convolution of ``x`` (8/27 of the products).
        Folded FP16 weights round each merged tap sum once; FP32 weights keep
        the FP32 sums.
        """
        mx = self.mx
        packed = self.nax_weights.get(key)
        weight = self.weights[key + ".weight"]
        if packed is None or packed.ndim != 3 or x.dtype != weight.dtype:
            return self.conv(self.resize_nearest(x, tuple(2 * int(s) for s in x.shape[2:])), key)
        bias = self.weights.get(key + ".bias")
        if bias is None:
            bias = mx.zeros((int(weight.shape[0]),), dtype=x.dtype)
        return nax_conv3d(x, packed, bias, upsample=True, mx=mx)

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
        """``residual + (conv(norm_silu(x)) + shift)`` with a padding-1 convolution.

        ``shift`` is an optional per-batch, per-output-channel ``(B, O)`` term
        broadcast over space, such as a projected time embedding; ``residual``
        is an optional NCDHW tensor of the output's shape. On the NAX path the
        normalize-SiLU result is never materialized: group statistics become a
        per-channel affine that the convolution's input packing applies before
        rounding to FP16 once, matching ``norm_silu``. The convolution, bias,
        shift and residual are then summed in FP32 and rounded once, where the
        separate operations round after each addition. FP32 networks take the
        same path in exact mode, which splits the prepared input instead of
        rounding it and stores the FP32 sum.
        """
        mx = self.mx
        packed = self.nax_weights.get(conv_key)
        if (
            packed is None
            or packed.ndim != 2
            or x.dtype != self.weights[conv_key + ".weight"].dtype
            or not self.native_kernels
            or not native_group_norm_eligible(x, groups)
        ):
            y = self.conv(self.norm_silu(x, norm_key, groups=groups, eps=eps), conv_key)
            if shift is not None:
                y = y + shift[:, :, None, None, None]
            return y if residual is None else residual + y
        x = mx.contiguous(x)
        affine = group_norm_affine(
            x,
            self.weights[norm_key + ".weight"],
            self.weights[norm_key + ".bias"],
            groups=groups,
            eps=eps,
            mx=mx,
        )
        out_channels = int(self.weights[conv_key + ".weight"].shape[0])
        bias = self.weights.get(conv_key + ".bias")
        if bias is None:
            bias = mx.zeros((out_channels,), dtype=x.dtype)
        if shift is not None:
            bias = bias.astype(mx.float32) + shift.astype(mx.float32)
        return nax_conv3d(x, packed, bias, affine=affine, silu=True, residual=residual, mx=mx)

    def silu(self, x: Array) -> Array:
        """SiLU in the input dtype; the UNet overrides this with one FP32 rounding."""
        return silu(x, self.mx)

    def _native_norm(self, x: Array, key: str, *, groups: int, eps: float, silu: bool) -> Array:
        """Apply the native group-normalization kernel with optional SiLU."""
        return native_group_norm(
            x,
            self.weights[key + ".weight"],
            self.weights[key + ".bias"],
            groups=groups,
            eps=eps,
            silu=silu,
            mx=self.mx,
        )

    def layer_norm(self, x: Array, key: str, *, eps: float = 1e-5) -> Array:
        """Apply source-keyed affine layer normalization over the final axis."""
        return self.mx.fast.layer_norm(
            x,
            self.weights[key + ".weight"],
            self.weights[key + ".bias"],
            eps,
        )

    def gelu(self, x: Array) -> Array:
        """Evaluate the erf GELU in FP32 and return the input dtype."""
        # Source nn.GELU uses the exact erf formula, not tanh approximation.
        dtype = x.dtype
        x = x.astype(self.mx.float32)
        return (x * (1 + self.mx.erf(x / math.sqrt(2))) * 0.5).astype(dtype)

    def resize_nearest(self, x: Array, shape: tuple[int, ...]) -> Array:
        """Nearest-neighbour resize of ``(B, C, D, H, W)`` to spatial ``shape`` (D, H, W).

        Output index ``i`` reads input index ``floor(i * in_size / out_size)``.
        """
        if len(shape) != 3:
            raise ValueError("LesionDiffusion nearest resizing needs three spatial axes")
        if x.ndim == 5 and min(x.shape) > 0 and shape == tuple(2 * s for s in x.shape[2:]):
            # Both source decoders double every spatial axis. A single strided
            # copy preserves their floor-index rule without three intermediates.
            return upsample_nearest2(x, mx=self.mx)
        for axis, target in enumerate(shape, start=2):
            indices = self.mx.floor(self.mx.arange(target) * (x.shape[axis] / target))
            x = self.mx.take(x, indices.astype(self.mx.int32), axis=axis)
        return x


class LdmUnet(SourceLayers):
    """LMNet/LINet: upstream's 3-D OpenAI-style UNet with spatial transformers.

    ``fused_attention=True`` replaces explicit softmax attention with Metal's fused
    kernel (used for both denoisers in "conditioned" mode); source mode keeps the
    explicit form to stay on upstream's arithmetic.
    """

    def __init__(
        self,
        weights: Mapping[str, Array],
        mx: Mlx,
        config: UnetConfig,
        *,
        source_arithmetic: bool = True,
        fused_attention: bool = False,
        native_kernels: bool | None = None,
    ):
        """Initialize the source UNet and its attention and arithmetic policies."""
        super().__init__(
            weights, mx, source_arithmetic=source_arithmetic, native_kernels=native_kernels
        )
        self.config = config
        self.fused_attention = fused_attention

    def silu(self, x: Array) -> Array:
        # Source nn.SiLU rounds once after the activation. VQ's separate
        # sigmoid/product deliberately retains its two half rounding steps.
        """Apply SiLU with FP32 intermediate rounding and restore the input dtype."""
        return silu(x.astype(self.mx.float32), self.mx).astype(x.dtype)

    def softmax(self, x: Array) -> Array:
        # Torch softmax accumulates in float32, then returns the input dtype.
        """Apply source-compatible precise softmax on the final axis."""
        return self.mx.softmax(x, axis=-1, precise=True)

    def residual(self, x: Array, emb: Array, key: str) -> Array:
        """Apply a time-conditioned NCDHW residual block.

        The projected embedding is broadcast over the spatial axes; an optional
        learned skip projection handles channel changes.
        """
        channels = x.shape[1]
        groups = 32 if channels % 32 == 0 else channels
        h = self.norm_silu_conv(
            x,
            key + ".in_layers.0",
            key + ".in_layers.2",
            groups=groups,
            eps=1e-5,
            shift=self.dense(self.silu(emb), key + ".emb_layers.1"),
        )
        groups = 32 if h.shape[1] % 32 == 0 else h.shape[1]
        if key + ".skip_connection.weight" in self.weights:
            x = self.conv(x, key + ".skip_connection", padding=0)
        return self.norm_silu_conv(
            h, key + ".out_layers.0", key + ".out_layers.3", groups=groups, eps=1e-5, residual=x
        )

    def attention(self, x: Array, context: Array | None, key: str, heads: int) -> Array:
        """Attend from ``x`` to ``context`` sequences, or to ``x`` when context is absent.

        Both inputs use ``(B, sequence, features)``; the projected feature size
        must divide evenly across ``heads``. The result has the shape of ``x``.
        """
        mx = self.mx
        context = x if context is None else context
        q, k, v = (
            self.dense(x, key + ".to_q"),
            self.dense(context, key + ".to_k"),
            self.dense(context, key + ".to_v"),
        )
        depth = q.shape[-1] // heads
        if self.fused_attention and depth == 32 and flash_eligible(q, k, heads, mx):
            # FP32 32-wide heads: unpadded flash kernel in the projection layout.
            result = flash_attention(q, k, v, heads=heads, scale=depth**-0.5, mx=mx)
            return self.dense(result, key + ".to_out.0")

        def split(tensor: Array) -> Array:
            """Split projected tokens into attention heads."""
            return mx.transpose(
                mx.reshape(tensor, (tensor.shape[0], tensor.shape[1], heads, depth)), (0, 2, 1, 3)
            )

        if self.fused_attention:
            q, k, v = split(q), split(k), split(v)
            # Metal's fused full-attention kernel starts at 64-wide heads.
            # Zero padding retains the 32-wide model's original dot products
            # and scale; discard the padded value channels after attention.
            if depth < 64:
                padding = [(0, 0), (0, 0), (0, 0), (0, 64 - depth)]
                q, k, v = (mx.pad(tensor, padding) for tensor in (q, k, v))
            result = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=depth**-0.5, force_fused=True
            )[..., :depth]
        else:
            # Source half attention rounds similarity and probability tensors;
            # LINet retains those boundaries rather than using fused SDPA.
            similarity = (split(q) @ mx.swapaxes(split(k), -1, -2)) * depth**-0.5
            result = self.softmax(similarity) @ split(v)
        result = mx.reshape(mx.transpose(result, (0, 2, 1, 3)), x.shape)
        return self.dense(result, key + ".to_out.0")

    def transformer(self, x: Array, context: Array | None, key: str) -> Array:
        """Apply self-attention, optional context attention and a feed-forward block.

        ``x`` is NCDHW; spatial voxels are flattened into the attention sequence
        and restored to the original shape before the residual addition.
        """
        residual = x
        x = self.conv(self.norm(x, key + ".norm", groups=32, eps=1e-6), key + ".proj_in", padding=0)
        shape = x.shape
        x = self.mx.reshape(self.mx.transpose(x, (0, 2, 3, 4, 1)), (shape[0], -1, shape[1]))
        heads = shape[1] // self.config.num_head_channels
        block = key + ".transformer_blocks.0"
        x = x + self.attention(self.layer_norm(x, block + ".norm1"), None, block + ".attn1", heads)
        x = x + self.attention(
            self.layer_norm(x, block + ".norm2"), context, block + ".attn2", heads
        )
        ff = self.gelu(self.dense(self.layer_norm(x, block + ".norm3"), block + ".ff.net.0.0"))
        x = x + self.dense(ff, block + ".ff.net.2")
        x = self.mx.transpose(self.mx.reshape(x, (shape[0], *shape[2:], shape[1])), (0, 4, 1, 2, 3))
        return residual + self.conv(x, key + ".proj_out", padding=0)

    def __call__(self, x: Array, timesteps: Array, context: Array | None) -> Array:
        """Predict noise for one denoising step.

        Args:
            x: Network input ``(B, C_in, D, H, W)``, channels first. Spatial sizes
                must be divisible by ``2 ** (levels - 1)`` (4 for the shipped configs).
            timesteps: Integer diffusion timesteps, shape ``(B,)``.
            context: Text embeddings ``(B, N, D_ctx)``, or ``None``. Ignored when the
                config sets ``skip_context``.

        Returns:
            Prediction ``(B, C_out, D, H, W)``, computed in the weights' dtype
            (float32 for LMNet, float16 for LINet in the runner).
        """
        mx, cfg = self.mx, self.config
        half = cfg.model_channels // 2
        frequency = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        phase = timesteps[:, None].astype(mx.float32) * frequency[None]
        emb = mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=1)
        dtype = self.weights["time_embed.0.weight"].dtype
        emb = self.dense(self.silu(self.dense(emb.astype(dtype), "time_embed.0")), "time_embed.2")
        if context is not None:
            context = (
                None
                if cfg.skip_context
                else self.dense(context.astype(dtype), "context_linear_projection")
            )
        x = self.conv(x.astype(dtype), "input_blocks.0.0")
        skips = [x]
        block_index, resolution = 1, 1
        for level in range(len(cfg.channel_mult)):
            for _ in range(cfg.num_res_blocks):
                key = f"input_blocks.{block_index}"
                x = self.residual(x, emb, key + ".0")
                if resolution in cfg.attention_resolutions:
                    x = self.transformer(x, context, key + ".1")
                skips.append(x)
                block_index += 1
            if level < len(cfg.channel_mult) - 1:
                x = self.conv(x, f"input_blocks.{block_index}.0.op", stride=2)
                skips.append(x)
                block_index += 1
                resolution *= 2
        x = self.residual(x, emb, "middle_block.0")
        x = self.transformer(x, context, "middle_block.1")
        x = self.residual(x, emb, "middle_block.2")
        block_index = 0
        for level in reversed(range(len(cfg.channel_mult))):
            for block in range(cfg.num_res_blocks + 1):
                key = f"output_blocks.{block_index}"
                x = self.residual(mx.concatenate([x, skips.pop()], axis=1), emb, key + ".0")
                has_attention = resolution in cfg.attention_resolutions
                if has_attention:
                    x = self.transformer(x, context, key + ".1")
                if level and block == cfg.num_res_blocks:
                    x = self.upsample_conv(x, key + f".{2 if has_attention else 1}.conv")
                    resolution //= 2
                block_index += 1
        return self.norm_silu_conv(
            x, "out.0", "out.2", groups=32 if x.shape[1] % 32 == 0 else x.shape[1], eps=1e-5
        )


class VqAutoencoder(SourceLayers):
    """Upstream's 3-D VQ autoencoder, as used by LINet: factor-two spatial latent.

    Tensors are channels first. ``encode`` maps ``(B, C, D, H, W)`` to a latent of
    half the spatial size (rounded down); ``decode`` snaps latent vectors to their
    nearest codebook entry and returns twice the latent's spatial size.
    """

    def residual(self, x: Array, key: str) -> Array:
        """Apply one VQ autoencoder residual block."""
        h = self.norm_silu_conv(x, key + ".norm1", key + ".conv1", groups=16, eps=1e-6)
        if key + ".nin_shortcut.weight" in self.weights:
            x = self.conv(x, key + ".nin_shortcut", padding=0)
        return self.norm_silu_conv(
            h, key + ".norm2", key + ".conv2", groups=16, eps=1e-6, residual=x
        )

    def encode(self, x: Array) -> Array:
        """Return the unquantized NCDHW latent with each spatial axis halved, rounded down."""
        x = self.conv(x, "encoder.conv_in")
        for level in range(3):
            x = self.residual(x, f"encoder.down.{level}.block.0")
            if level == 0:
                x = self.mx.pad(x, [(0, 0), (0, 0), (0, 1), (0, 1), (0, 1)])
                x = self.conv(x, "encoder.down.0.downsample.conv", padding=0, stride=2)
        x = self.residual(x, "encoder.mid.block_1")
        x = self.residual(x, "encoder.mid.block_2")
        x = self.norm_silu_conv(x, "encoder.norm_out", "encoder.conv_out", groups=16, eps=1e-6)
        return self.conv(x, "quant_conv", padding=0)

    def quantize(self, z: Array, *, chunk_size: int = 4096) -> Array:
        """Select a nearest codebook entry for each latent vector of NCDHW ``z``.

        Nearest means smallest computed squared Euclidean distance; positions are
        processed in chunks of ``chunk_size`` to bound the distance matrix. The
        source's straight-through expression ``z + (selected - z)`` is retained,
        so rounding can make the result differ from the selected entry. Selection
        is discontinuous, so tiny rounding differences can change a chosen code.
        """
        mx = self.mx
        shape = z.shape
        flat = mx.reshape(mx.transpose(z, (0, 2, 3, 4, 1)), (-1, shape[1]))
        codebook = self.weights["quantize.embedding.weight"]
        # Source half multiplication rounds each square, then torch.sum
        # accumulates in float32 and rounds once back to half.
        codebook_norm = mx.sum((codebook * codebook).astype(mx.float32), axis=1).astype(
            codebook.dtype
        )
        pieces: list[Array] = []
        for start in range(0, flat.shape[0], chunk_size):
            part = flat[start : start + chunk_size]
            distance = (
                mx.sum((part * part).astype(mx.float32), axis=1, keepdims=True).astype(part.dtype)
                + codebook_norm[None]
                - 2 * (part @ codebook.T)
            )
            selected = codebook[mx.argmin(distance, axis=1)]
            # Upstream VectorQuantizer retains its straight-through arithmetic
            # during inference too; fp16 cancellation changes the returned values.
            pieces.append(part + (selected - part))
        return mx.transpose(
            mx.reshape(mx.concatenate(pieces), (shape[0], *shape[2:], shape[1])),
            (0, 4, 1, 2, 3),
        )

    def decode(self, z: Array, *, quantize: bool = True) -> Array:
        """Decode a latent ``(B, C, d, h, w)`` to a volume ``(B, C_out, 2d, 2h, 2w)``.

        Set ``quantize=False`` to skip the codebook lookup and decode ``z`` as given.
        """
        x = self.conv(self.quantize(z) if quantize else z, "post_quant_conv", padding=0)
        x = self.conv(x, "decoder.conv_in")
        x = self.residual(x, "decoder.mid.block_1")
        x = self.residual(x, "decoder.mid.block_2")
        for level in reversed(range(3)):
            for block in range(2):
                x = self.residual(x, f"decoder.up.{level}.block.{block}")
            if level == 1:
                x = self.upsample_conv(x, "decoder.up.1.upsample.conv")
        return self.norm_silu_conv(x, "decoder.norm_out", "decoder.conv_out", groups=16, eps=1e-6)


class BiomedTextEncoder(SourceLayers):
    """PubMedBERT CLS pooling and OpenCLIP MLP projection on Metal."""

    def __call__(self, token_ids: Array) -> Array:
        """Embed token id sequences ``(N, L)`` as vectors ``(N, D_ctx)``.

        Padding positions (id 0) are masked out of attention; the vector is the
        projected first-token (CLS) state. ``L`` must not exceed the position-embedding
        table (256 for BiomedCLIP).
        """
        mx = self.mx
        prefix = "transformer.embeddings"
        x = self.weights[prefix + ".word_embeddings.weight"][token_ids]
        x = (
            x
            + self.weights[prefix + ".position_embeddings.weight"][mx.arange(token_ids.shape[1])][
                None
            ]
        )
        x = x + self.weights[prefix + ".token_type_embeddings.weight"][0]
        x = self.layer_norm(x, prefix + ".LayerNorm", eps=1e-12)
        mask = mx.where(
            token_ids[:, None, None, :] != 0,
            cast(ArrayConstructor, mx.array)(0, dtype=x.dtype),
            cast(ArrayConstructor, mx.array)(-float("inf"), dtype=x.dtype),
        )
        for index in range(12):
            key = f"transformer.encoder.layer.{index}"
            q, k, v = (
                self.dense(x, key + ".attention.self." + name) for name in ("query", "key", "value")
            )

            def split(tensor: Array) -> Array:
                """Split text-attention projections into heads."""
                return mx.transpose(
                    mx.reshape(tensor, (tensor.shape[0], tensor.shape[1], 12, 64)), (0, 2, 1, 3)
                )

            h = mx.fast.scaled_dot_product_attention(
                split(q), split(k), split(v), scale=0.125, mask=mask
            )
            h = mx.reshape(mx.transpose(h, (0, 2, 1, 3)), x.shape)
            x = self.layer_norm(
                x + self.dense(h, key + ".attention.output.dense"),
                key + ".attention.output.LayerNorm",
                eps=1e-12,
            )
            h = self.gelu(self.dense(x, key + ".intermediate.dense"))
            x = self.layer_norm(
                x + self.dense(h, key + ".output.dense"), key + ".output.LayerNorm", eps=1e-12
            )
        return self.dense(self.gelu(self.dense(x[:, 0], "proj.0")), "proj.2")
