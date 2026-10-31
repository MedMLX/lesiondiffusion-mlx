"""Verified upstream source loading for offline fixture recording only.

The AST loader executes named definitions verbatim. It excludes optional training,
plotting, segmentation and service imports; it never substitutes numerical bodies.
"""

from __future__ import annotations

import ast
import importlib
import platform
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import numpy as np
import torch
from _torch import from_numpy
from einops import rearrange
from torch import nn

from lesiondiffusion_mlx._arrays import HostArray
from lesiondiffusion_mlx._reference import REFERENCE as REFERENCE

REPO_ROOT = Path(__file__).resolve().parents[1]


class UnetNetwork(Protocol):
    def __call__(
        self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor | None
    ) -> torch.Tensor:
        """Run the upstream conditioned denoising network."""
        ...


class VqNetwork(Protocol):
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image into the upstream VQ latent space."""
        ...
    def decode(self, z: torch.Tensor, force_not_quantize: bool = False) -> torch.Tensor:
        """Decode an upstream VQ latent."""
        ...
    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, object]:
        """Quantize an upstream latent against the codebook."""
        ...


class SourceSampler(Protocol):
    ddim_timesteps: HostArray
    ddim_alphas: torch.Tensor
    ddim_alphas_prev: HostArray
    ddim_sqrt_one_minus_alphas: torch.Tensor
    ddim_sigmas: torch.Tensor


class InpaintingNetwork(Protocol):
    device: torch.device
    is_conditional_first_stage: bool
    conditioning_key: str | None
    first_stage_key: str
    first_stage_model: VqNetwork
    save_dataset: bool
    eval_scheme: list[str]
    encode_first_stage: Callable[[torch.Tensor, object], torch.Tensor]
    get_first_stage_encoding: Callable[[torch.Tensor], torch.Tensor]
    decode_first_stage: Callable[..., torch.Tensor]
    ema_scope: Callable[..., object]
    sample_log: Callable[..., tuple[torch.Tensor, None]]

    def log_images(
        self, batch: dict[str, object], *, ddim_steps: int
    ) -> tuple[object, dict[str, torch.Tensor]]:
        """Run upstream inpainting and return recorded image tensors."""
        ...


def recording_directory() -> Path:
    """Only Apple Silicon Macs produce the package's reference recordings."""
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise RuntimeError("Reference recording requires an Apple Silicon Mac (darwin-arm64)")
    directory = REPO_ROOT / "tests/fixtures/darwin-arm64"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def definitions(
    root: Path, path: str, names: tuple[str, ...], scope: dict[str, object]
) -> SimpleNamespace:
    """Load selected definitions from the pinned upstream source."""
    source = root / path
    tree = ast.parse(source.read_text(), filename=str(source))
    selected: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name in names
    ]
    if {node.name for node in selected if isinstance(node, ast.FunctionDef | ast.ClassDef)} != set(
        names
    ):
        raise ValueError(f"Missing upstream definitions in {source}: {names}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


def prepare(
    root: Path, repository: Path | None = None, text_repository: Path | None = None
) -> None:
    """Prepare the fetched upstream source for direct numerical comparisons."""
    sys.path.insert(0, str(root / "pipeline"))
    torch.set_num_threads(1)


def unet(root: Path, weights: dict[str, torch.Tensor], *, inpainting: bool) -> UnetNetwork:
    """Construct the upstream denoising network with supplied weights."""
    attention = importlib.import_module("ldm.modules.attention")
    FeedForward = cast(type[nn.Module], attention.FeedForward)
    UNetModel = cast(
        type[nn.Module],
        importlib.import_module("ldm.modules.diffusionmodules.openaimodel").UNetModel,
    )

    network = UNetModel(
        image_size=4,
        in_channels=5 if inpainting else 4,
        model_channels=32,
        out_channels=5 if inpainting else 1,
        num_res_blocks=1,
        attention_resolutions=(1, 2),
        channel_mult=(1, 2),
        dims=3,
        num_head_channels=16,
        use_checkpoint=False,
        use_fp16=inpainting,
        use_spatial_transformer=True,
        context_dim=6,
        skip_context=inpainting,
    )
    for module in network.modules():
        if hasattr(module, "ff"):
            module.ff = FeedForward(
                cast(nn.LayerNorm, module.norm3).normalized_shape[0], mult=2, glu=False
            )
        if inpainting and hasattr(module, "attn2"):
            CrossAttention = cast(type[nn.Module], attention.CrossAttention)

            width = cast(nn.LayerNorm, module.norm2).normalized_shape[0]
            module.attn2 = CrossAttention(width, heads=width // 16, dim_head=16)
    # The source creates an unused projection even when skip_context is true.
    if inpainting:
        weights = {**network.state_dict(), **weights}
        network.half()
    network.load_state_dict(weights, strict=True)
    return cast(UnetNetwork, network.eval())


def vq(root: Path, weights: dict[str, torch.Tensor]) -> VqNetwork:
    """Construct the upstream VQ network with supplied weights."""
    model_module = importlib.import_module("ldm.modules.diffusionmodules.model")
    Decoder, Encoder = model_module.Decoder, model_module.Encoder
    instantiate_from_config = importlib.import_module("ldm.util").instantiate_from_config

    scope: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "np": np,
        "rearrange": rearrange,
        "pl": SimpleNamespace(LightningModule=nn.Module),
        "Encoder": Encoder,
        "Decoder": Decoder,
        "instantiate_from_config": instantiate_from_config,
        "contextmanager": contextmanager,
    }
    classes = definitions(
        root,
        "pipeline/ldm/models/autoencoder.py",
        ("VectorQuantizer", "VQModel", "VQModelInterface"),
        scope,
    )
    config = dict(
        ch=16,
        out_ch=1,
        ch_mult=(1, 2, 2),
        num_res_blocks=1,
        attn_resolutions=(),
        in_channels=1,
        resolution=(4, 4, 4),
        z_channels=4,
        double_z=False,
        attn_type="none",
        sample_depth=(0,),
        dims=3,
    )
    network = classes.VQModelInterface(
        embed_dim=4,
        n_embed=8,
        ddconfig=config,
        lossconfig={"target": "torch.nn.Identity"},
    )
    network.to(dtype=next(iter(weights.values())).dtype)
    network.load_state_dict(weights, strict=True)
    return cast(VqNetwork, network.eval())


def register_pooler[T](cls: T) -> T:
    """Register the upstream text pooling class."""
    return cls


def text(root: Path, weights: dict[str, torch.Tensor]) -> nn.Module:
    """Construct the upstream text encoder with supplied weights."""
    transformers = importlib.import_module("transformers")
    BertConfig, BertModel = transformers.BertConfig, transformers.BertModel
    output_types = importlib.import_module("transformers.modeling_outputs")
    BaseModelOutput = output_types.BaseModelOutput
    BaseModelOutputWithPooling = output_types.BaseModelOutputWithPooling
    BaseModelOutputWithPoolingAndCrossAttentions = (
        output_types.BaseModelOutputWithPoolingAndCrossAttentions
    )

    scope: dict[str, object] = dict(
        nn=nn,
        torch=torch,
        TensorType=torch.Tensor,
        BaseModelOutput=BaseModelOutput,
        BaseModelOutputWithPooling=BaseModelOutputWithPooling,
        BaseModelOutputWithPoolingAndCrossAttentions=BaseModelOutputWithPoolingAndCrossAttentions,
        PretrainedConfig=object,
    )
    classes = definitions(
        root,
        "open_clip/hf_model.py",
        ("ClsPooler", "ClsLastHiddenStatePooler", "HFTextEncoder"),
        {**scope, "register_pooler": register_pooler},
    )
    network = classes.HFTextEncoder.__new__(classes.HFTextEncoder)
    nn.Module.__init__(network)
    network.config = BertConfig(
        vocab_size=12,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=16,
        max_position_embeddings=256,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        attn_implementation="eager",
    )
    network.output_tokens = False
    network.transformer = BertModel(network.config, add_pooling_layer=False)
    network.pooler = classes.ClsLastHiddenStatePooler()
    network.proj = nn.Sequential(
        nn.Linear(768, 16, bias=False), nn.GELU(), nn.Linear(16, 6, bias=False)
    )
    network.load_state_dict(weights, strict=True)
    return network.eval()


def data_functions(root: Path) -> SimpleNamespace:
    """Load the upstream image and mask data helpers."""
    import nibabel

    orientations = importlib.import_module("nibabel.orientations")
    axcodes2ornt = cast(Callable[[tuple[str, ...]], HostArray], orientations.axcodes2ornt)
    ornt_transform = cast(Callable[[HostArray, HostArray], HostArray], orientations.ornt_transform)

    return definitions(
        root,
        "pipeline/ldm/data/data_process_func.py",
        ("reorient_func", "load_nifty_volume_as_array", "img_multi_thresh_normalized"),
        dict(np=np, nibabel=nibabel, axcodes2ornt=axcodes2ornt, ornt_transform=ornt_transform),
    )


def sampler(
    root: Path,
    alphas: HostArray,
    denoiser: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor],
    image: torch.Tensor,
    condition: torch.Tensor | None,
    *,
    steps: int,
    initial: torch.Tensor,
    noise: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[SourceSampler, torch.Tensor]:
    """Construct the upstream deterministic diffusion sampler."""
    ddim = importlib.import_module("ldm.models.diffusion.ddim")

    class ReferenceSampler(ddim.InpaintingSampler if mask is not None else ddim.MaskSampler):
        def register_buffer(self, name: str, value: object) -> None:
            # Upstream hard-codes CUDA here; storage alone is moved to CPU.
            """Store a reference sampler buffer."""
            setattr(self, name, value)

    def apply_model(x: torch.Tensor, *args: torch.Tensor | None) -> torch.Tensor:
        """Apply the upstream denoising model during sampling."""
        rest = args
        if mask is None:
            z, c, timestep = rest
            z = cast(torch.Tensor, z)
            network_input = torch.cat((x, z), dim=1)
        else:
            supplied, timestep, c = rest
            supplied = cast(torch.Tensor, supplied)
            network_input = torch.cat((x, supplied), dim=1)
        return denoiser(network_input, cast(torch.Tensor, timestep), c)

    model = SimpleNamespace(
        num_timesteps=len(alphas),
        dims=3,
        device=torch.device("cpu"),
        betas=torch.zeros(len(alphas)),
        alphas_cumprod=from_numpy(alphas),
        alphas_cumprod_prev=from_numpy(np.r_[alphas[:1], alphas[:-1]]),
        apply_model=apply_model,
    )
    reference = ReferenceSampler(model)
    reference.make_schedule(steps, ddim_eta=1, verbose=False)
    from unittest.mock import patch

    noises = iter(noise)

    def next_noise(*args: object) -> torch.Tensor:
        """Return the next predetermined sampling noise tensor."""
        return next(noises)

    with patch.object(ddim, "noise_like", next_noise):
        result, _ = reference.ddim_sampling(
            condition, initial.shape, x_T=initial, x0=image, mask=mask, verbose=False
        )
    return cast(SourceSampler, reference), cast(
        torch.Tensor, result[:, :4] if mask is not None else result
    )


def inference_classes(
    root: Path, mask_base: type[object], inpainting_base: type[object]
) -> SimpleNamespace:
    """Load the pinned upstream inference classes."""
    import random
    import re

    from scipy import ndimage

    scope: dict[str, object] = dict(
        torch=torch,
        np=np,
        ndm=ndimage,
        random=random,
        re=re,
        MaskDiffusion=mask_base,
        InpaintingDiffusion_v2=inpainting_base,
        ComputeMetrics=type("ComputeMetrics", (), {}),
        AlignDataset=type("AlignDataset", (), {}),
        totalseg_class=organ_labels(root),
    )
    return definitions(
        root,
        "pipeline/inference/models.py",
        ("InferMaskDiffusion", "InferInpaintingDiffusion_v2"),
        scope,
    )


def crop_reference(root: Path, mask: HostArray, shape: tuple[int, int, int]) -> HostArray:
    """Observe the original inference crop through an auxiliary coordinate tensor."""
    observed: list[HostArray] = []

    class MaskBase:
        def log_images(
            self, batch: dict[str, torch.Tensor], *args: object, **kwargs: object
        ) -> dict[str, torch.Tensor]:
            """Record image tensors and crop coordinates for the source fixture."""
            coordinates = batch["coordinates"][0].numpy()
            starts = coordinates[:, 0, 0, 0]
            stops = coordinates[:, -1, -1, -1] + 1
            observed.append(np.stack((starts, stops), axis=1))
            return {"samples": torch.zeros_like(batch["seg"]), "inputs": batch["seg"]}

    classes = inference_classes(root, MaskBase, type("InpaintingBase", (), {}))
    network = classes.InferMaskDiffusion.__new__(classes.InferMaskDiffusion)
    network.save_dataset = False
    network.eval_scheme = []
    network.log_images(
        dict(
            max_mask_num={"max_mask_num": 1},
            crop_size={"crop_size": shape},
            image_path={"image_path": "demo/liver/tumor/ct.nii.gz"},
            label=from_numpy(mask[None, None].astype(np.uint8) * 2),
            seg=torch.ones((1, 1, *mask.shape)),
            coordinates=from_numpy(np.indices(mask.shape)[None]),
        )
    )
    return observed[0]


def organ_labels(root: Path) -> dict[str, str]:
    """Read the upstream organ-label definitions."""
    source = root / "pipeline/ldm/data/Torchio_contrast_dataloader.py"
    tree = ast.parse(source.read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "totalseg_class"
            for target in node.targets
        )
    )
    return cast(dict[str, str], ast.literal_eval(node.value))


def placement(root: Path) -> SimpleNamespace:
    """Compute source-reference lesion placement within the organ mask."""
    from scipy.ndimage import binary_erosion, generate_binary_structure

    return definitions(
        root,
        "pipeline/preprocess.py",
        ("generate_random_bbox_from_mask", "create_bbox_mask"),
        dict(
            np=np,
            binary_erosion=binary_erosion,
            generate_binary_structure=generate_binary_structure,
            nib=importlib.import_module("nibabel"),
            os=importlib.import_module("os"),
        ),
    )
