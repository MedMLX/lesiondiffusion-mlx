"""Executable pinned-source oracles for native LesionDiffusion Metal blocks."""

from __future__ import annotations

import ast
import importlib
import math
import os
import tempfile
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import torch as torch_types
from torch import nn

from scripts._torch import from_numpy

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Mlx


import numpy as np
import pytest

from lesiondiffusion_mlx import DdimSampler, lesion_diffusion_timesteps
from lesiondiffusion_mlx._arrays import NumericArray
from lesiondiffusion_mlx._typing import ArrayConstructor
from lesiondiffusion_mlx.mlx_model import (
    LMNET_CONFIG,
    LdmUnet,
    SourceLayers,
    VqAutoencoder,
)


class PathPatcher(Protocol):
    def syspath_prepend(self, path: str) -> None:
        """Prepend a source path for the reference import."""
        ...


class TorchModule(Protocol):
    nn: TorchNn

    def manual_seed(self, seed: int) -> torch_types.Generator:
        """Seed the reference Torch generator."""
        ...
    def no_grad(self) -> AbstractContextManager[None]:
        """Return the reference no-gradient context."""
        ...
    def randn(self, *size: int) -> torch_types.Tensor:
        """Draw reference normally distributed tensors."""
        ...

    tensor = staticmethod(torch_types.tensor)
    any = staticmethod(torch_types.any)
    cat = staticmethod(torch_types.cat)
    sqrt = staticmethod(torch_types.sqrt)


class TorchNn(Protocol):
    Module: type[nn.Module]
    Conv3d: type[nn.Conv3d]


SOURCE_ROOT = Path(
    os.environ.get(
        "LESIONDIFFUSION_UPSTREAM_ROOT", str(Path(tempfile.gettempdir()) / "lesion-upstream")
    )
)
SOURCE = SOURCE_ROOT / "pipeline"


@pytest.fixture(scope="session")
def admitted_source() -> Path:
    """Provide the external upstream source when available."""
    if not SOURCE.is_dir():
        pytest.fail(
            "Missing pinned source; run scripts/fetch_reference.py as in docs/technical.md"
        )
    return SOURCE


@pytest.fixture
def oracle(
    monkeypatch: PathPatcher, mx: Mlx, admitted_source: Path
) -> tuple[TorchModule, Mlx]:
    """Provide reference Torch and native MLX runtimes for comparison."""
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(admitted_source))
    return cast(TorchModule, torch), mx


def mlx_weights(model: nn.Module, mx: Mlx) -> dict[str, Array]:
    """Convert source model weights to native arrays."""
    return {
        key: cast(ArrayConstructor, mx.array)(value.detach().numpy())
        for key, value in model.state_dict().items()
    }


@pytest.mark.parametrize("fused_attention", [False, True])
def test_unet_matches_pinned_source_nonzero_weights(
    oracle: tuple[TorchModule, Mlx], fused_attention: bool
) -> None:
    """Compare native UNet output with pinned source using nonzero weights."""
    torch, mx = oracle
    module = importlib.import_module("ldm.modules.diffusionmodules.openaimodel")
    torch.manual_seed(42)
    source = module.UNetModel(
        image_size=64,
        in_channels=4,
        out_channels=1,
        model_channels=16,
        attention_resolutions=[4],
        num_res_blocks=1,
        channel_mult=[1, 2, 4],
        num_head_channels=32,
        use_spatial_transformer=True,
        context_dim=512,
        use_checkpoint=False,
    ).eval()
    # The source zero-initializes residual/output convolutions; replace only
    # zero parameters so the fixture exercises the complete network.
    with torch.no_grad():
        for parameter in source.parameters():
            if not torch.any(parameter):
                parameter.normal_(0, 0.02)
    x = torch.randn(1, 4, 8, 12, 16)
    condition = torch.randn(1, 5, 512)
    timesteps = torch.tensor([501])
    with torch.no_grad():
        expected = source(x, timesteps, condition).numpy()
    native = LdmUnet(mlx_weights(source, mx), mx, LMNET_CONFIG, fused_attention=fused_attention)
    actual = native(
        cast(ArrayConstructor, mx.array)(x.numpy()),
        cast(ArrayConstructor, mx.array)(timesteps.numpy()),
        cast(ArrayConstructor, mx.array)(condition.numpy()),
    )
    mx.eval(actual)
    np.testing.assert_allclose(cast(NumericArray, np.array(actual)), expected, atol=3e-5, rtol=3e-4)


def test_vq_encoder_decoder_match_source(oracle: tuple[TorchModule, Mlx]) -> None:
    """Compare native VQ encoding and decoding with the source."""
    torch, mx = oracle
    module = importlib.import_module("ldm.modules.diffusionmodules.model")
    torch.manual_seed(23)
    config = dict(
        ch=32,
        out_ch=1,
        ch_mult=[1, 1, 1],
        num_res_blocks=1,
        attn_resolutions=[],
        in_channels=1,
        resolution=[8, 12, 16],
        z_channels=4,
        double_z=False,
        sample_depth=[0],
        attn_type="none",
        dims=3,
    )
    encoder, decoder = module.Encoder(**config).eval(), module.Decoder(**config).eval()
    quant_conv, post_quant = torch.nn.Conv3d(4, 4, 1), torch.nn.Conv3d(4, 4, 1)
    weights: dict[str, Array] = {}
    for prefix, model in (
        ("encoder", encoder),
        ("decoder", decoder),
        ("quant_conv", quant_conv),
        ("post_quant_conv", post_quant),
    ):
        weights.update({prefix + "." + key: value for key, value in mlx_weights(model, mx).items()})
    native = VqAutoencoder(weights, mx)
    x = torch.randn(1, 1, 8, 12, 16)
    with torch.no_grad():
        z = quant_conv(encoder(x))
        expected = decoder(post_quant(z), use_checkpoint=False).numpy()
    actual_z = native.encode(cast(ArrayConstructor, mx.array)(x.numpy()))
    actual = native.decode(actual_z, quantize=False)
    mx.eval(actual, actual_z)
    np.testing.assert_allclose(
        cast(NumericArray, np.array(actual_z)), z.detach().numpy(), atol=5e-6, rtol=1e-4
    )
    np.testing.assert_allclose(cast(NumericArray, np.array(actual)), expected, atol=1e-5, rtol=2e-4)


@pytest.mark.parametrize(
    ("channels", "outputs", "kernel", "padding", "stride", "with_bias"),
    [
        (2, 31, (1, 1, 1), 0, 1, True),
        (1, 3, (3, 3, 3), 1, 1, True),
        (2, 5, (3, 3, 3), 1, 2, False),
        (3, 7, (3, 1, 3), 1, 1, True),
        (4, 9, (2, 3, 2), 0, 1, False),
        (95, 35, (3, 3, 3), 1, 1, True),
        (96, 32, (3, 3, 3), 1, 2, True),
        (128, 33, (1, 1, 1), 0, 1, True),
        (512, 5, (1, 1, 1), 0, 1, False),
    ],
)
def test_source_half_convolution_preserves_source_rounding(
    mx: Mlx,
    channels: int,
    outputs: int,
    kernel: tuple[int, int, int],
    padding: int,
    stride: int,
    with_bias: bool,
) -> None:
    """Preserve source half-precision convolution rounding."""
    rng = np.random.default_rng(47)
    x = rng.normal(size=(2, channels, 5, 7, 9)).astype(np.float16)
    weight = rng.normal(size=(outputs, channels, *kernel)).astype(np.float16)
    bias = rng.normal(size=outputs).astype(np.float16) if with_bias else None
    with torch_types.no_grad():
        expected = nn.functional.conv3d(
            from_numpy(x),
            from_numpy(weight),
            from_numpy(bias) if bias is not None else None,
            padding=padding,
            stride=stride,
        ).numpy()
    weights = {"conv.weight": cast(ArrayConstructor, mx.array)(weight)}
    if bias is not None:
        weights["conv.bias"] = cast(ArrayConstructor, mx.array)(bias)
    native = SourceLayers(weights, mx)
    actual = native.conv(
        cast(ArrayConstructor, mx.array)(x), "conv", padding=padding, stride=stride
    )
    mx.eval(actual)
    np.testing.assert_array_equal(np.array(actual).view(np.uint16), expected.view(np.uint16))


@pytest.mark.parametrize(
    "shape",
    [
        (1, 96, 8, 12, 16),
        (2, 96, 8, 32, 32),
        (2, 16, 1, 1, 1),
        (2, 16, 1, 1, 7),
        (1, 16, 1, 1, 16377),
        (1, 16, 1, 1, 16384),
        (1, 16, 1, 1, 16391),
        (1, 16, 1, 1, 16392),
        (1, 16, 1, 1, 32760),
        (1, 16, 1, 1, 32768),
        (1, 16, 1, 1, 32775),
        (1, 16, 1, 1, 32776),
    ],
)
def test_source_half_group_norm_rounds_after_affine(
    oracle: tuple[TorchModule, Mlx], shape: tuple[int, int, int, int, int]
) -> None:
    """Match source half group-normalization rounding after affine terms."""
    torch, mx = oracle
    module = importlib.import_module("ldm.modules.diffusionmodules.model")
    rng = np.random.default_rng(42)
    x = rng.normal(size=shape).astype(np.float16)
    weight = rng.normal(size=shape[1]).astype(np.float16)
    bias = rng.normal(size=shape[1]).astype(np.float16)
    source = module.Normalize(shape[1]).half().eval()
    source.load_state_dict({"weight": from_numpy(weight), "bias": from_numpy(bias)})
    with torch.no_grad():
        expected = source(from_numpy(x)).float().numpy()
    native = SourceLayers(
        {
            "norm.weight": cast(ArrayConstructor, mx.array)(weight),
            "norm.bias": cast(ArrayConstructor, mx.array)(bias),
        },
        mx,
    )
    actual = native.norm(cast(ArrayConstructor, mx.array)(x), "norm", groups=16, eps=1e-6)
    mx.eval(actual)
    np.testing.assert_allclose(
        cast(NumericArray, np.array(actual, dtype=np.float32)), expected, atol=1e-3, rtol=0
    )


@pytest.mark.parametrize("reverse_width", [False, True])
def test_source_half_group_norm_keeps_strided_group_membership(
    mx: Mlx, reverse_width: bool
) -> None:
    """Shared contiguous inputs must preserve groups across batch and reversed views."""
    rng = np.random.default_rng(42)
    host = rng.normal(size=(2, 96, 8, 32, 32)).astype(np.float16)
    weight = rng.normal(size=96).astype(np.float16)
    bias = rng.normal(size=96).astype(np.float16)
    create = cast(ArrayConstructor, mx.array)
    x = mx.transpose(create(np.ascontiguousarray(host.transpose(0, 2, 3, 4, 1))), (0, 4, 1, 2, 3))
    if reverse_width:
        x, host = x[..., ::-1], host[..., ::-1]
    with torch_types.no_grad():
        expected = nn.functional.group_norm(
            from_numpy(np.ascontiguousarray(host)),
            32,
            from_numpy(weight),
            from_numpy(bias),
            eps=1e-5,
        ).numpy()
    native = SourceLayers({"norm.weight": create(weight), "norm.bias": create(bias)}, mx)
    actual = native.norm(x, "norm", groups=32, eps=1e-5)
    mx.eval(actual)
    np.testing.assert_allclose(
        cast(NumericArray, np.array(actual, dtype=np.float32)), expected, atol=1e-3, rtol=0
    )


def test_half_quantization_matches_pinned_source(oracle: tuple[TorchModule, Mlx]) -> None:
    """Compare half-precision quantization with the pinned source."""
    torch, mx = oracle
    source_file = SOURCE / "ldm/models/autoencoder.py"
    source_class = next(
        node
        for node in ast.parse(source_file.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "VectorQuantizer"
    )
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": torch.nn,
        "rearrange": importlib.import_module("einops").rearrange,
    }
    # Execute the pinned source class without loading its unrelated trainer.
    exec(
        compile(ast.Module(body=[source_class], type_ignores=[]), str(source_file), "exec"),
        namespace,
    )
    constructor = cast(Callable[[int, int, float], nn.Module], namespace["VectorQuantizer"])
    source = constructor(2048, 4, 0.25).half().eval()
    rng = np.random.default_rng(43)
    z = rng.normal(size=(1, 4, 8, 6, 8)).astype(np.float16)
    codebook = rng.normal(size=(2048, 4)).astype(np.float16)
    source.load_state_dict({"embedding.weight": from_numpy(codebook)})
    with torch.no_grad():
        expected = source(from_numpy(z))[0].numpy()
    native = VqAutoencoder(
        {"quantize.embedding.weight": cast(ArrayConstructor, mx.array)(codebook)}, mx
    )
    actual = native.quantize(cast(ArrayConstructor, mx.array)(z), chunk_size=37)
    mx.eval(actual)
    np.testing.assert_array_equal(cast(NumericArray, np.array(actual)), expected)


def test_ddim_matches_source_equations_and_preserves_unmasked_latent(
    oracle: tuple[TorchModule, Mlx],
) -> None:
    """Match DDIM source equations and preserve the unmasked latent."""
    torch, mx = oracle
    rng = np.random.default_rng(71)
    betas = np.linspace(math.sqrt(0.0015), math.sqrt(0.0195), 1000, dtype=np.float64) ** 2
    alphas = np.cumprod(1 - betas).astype(np.float32)
    x0 = rng.normal(size=(1, 4, 4, 4, 4)).astype(np.float32)
    initial = rng.normal(size=x0.shape).astype(np.float32)
    mask = np.zeros((1, 1, 4, 4, 4), dtype=np.float32)
    mask[:, :, 1:3, 1:3, 1:3] = 1
    noise = tuple(rng.normal(size=x0.shape).astype(np.float32) for _ in range(10))
    sampler = DdimSampler(cast(ArrayConstructor, mx.array)(alphas), mx, 10)

    def denoiser(x: Array, t: Array, context: Array | None) -> Array:
        """Return deterministic noise for the DDIM reference case."""
        del context
        return x * 0.13 + t[:, None, None, None, None] * 0.0001

    actual = sampler(
        denoiser,
        None,
        cast(ArrayConstructor, mx.array)(x0),
        initial_noise=cast(ArrayConstructor, mx.array)(initial),
        step_noise=tuple(cast(ArrayConstructor, mx.array)(n) for n in noise),
        mask=cast(ArrayConstructor, mx.array)(mask),
    )
    x, reference, mask_tensor = (
        from_numpy(initial),
        from_numpy(x0),
        from_numpy(mask),
    )
    for iteration, index in enumerate(reversed(range(10))):
        step = lesion_diffusion_timesteps(10)[index]
        a = torch.tensor(alphas[step])
        previous = torch.tensor(
            alphas[lesion_diffusion_timesteps(10)[index - 1]] if index else alphas[0]
        )
        sigma = torch.sqrt((1 - previous) / (1 - a) * (1 - a / previous))
        x = x * mask_tensor + (1 - mask_tensor) * reference
        eps = (torch.cat((x, mask_tensor), dim=1) * 0.13 + step * 0.0001)[:, :4]
        predicted = (x - torch.sqrt(1 - a) * eps) / torch.sqrt(a)
        x = (
            torch.sqrt(previous) * predicted
            + torch.sqrt(1 - previous - sigma**2) * eps
            + sigma * from_numpy(noise[iteration])
        )
    x = x * mask_tensor + (1 - mask_tensor) * reference
    mx.eval(actual)
    np.testing.assert_allclose(
        cast(NumericArray, np.array(actual)), x.numpy(), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_array_equal(
        cast(NumericArray, np.array(actual)) * (1 - mask), x0 * (1 - mask)
    )


def test_ddim_fp16_schedule_matches_pinned_source_parameters(
    oracle: tuple[TorchModule, Mlx],
) -> None:
    """Compare FP16 DDIM schedule values with pinned source parameters."""
    torch, mx = oracle
    source = importlib.import_module("ldm.modules.diffusionmodules.util")
    alphas = torch.tensor(np.linspace(0.998, 0.0004, 1000, dtype=np.float32)).half()
    with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
        sigma, selected, _ = source.make_ddim_sampling_parameters(
            alphas,
            cast(NumericArray, np.array(lesion_diffusion_timesteps(10))),
            1.0,
            verbose=False,
        )
    native = DdimSampler(cast(ArrayConstructor, mx.array)(alphas.numpy()), mx, 10)
    np.testing.assert_allclose(
        cast(NumericArray, np.array(native.sigmas)), np.asarray(sigma), atol=1e-7, rtol=1e-6
    )
    np.testing.assert_array_equal(
        cast(NumericArray, np.array(native.sqrt_one_minus)),
        torch.sqrt(1 - selected).float().numpy(),
    )
