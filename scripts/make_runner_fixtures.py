"""Record upstream preprocessing, threshold QC and full seeded generation on CPU.

The original inference methods execute verbatim. Model hooks supply seeded tiny
Torch networks, explicit noise and CPU storage; no package implementation is imported.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import TokenizerContract

import numpy as np
import torch
from _torch import from_numpy
from _upstream import (
    REPO_ROOT,
    InpaintingNetwork,
    UnetNetwork,
    VqNetwork,
    data_functions,
    definitions,
    inference_classes,
    organ_labels,
    placement,
    prepare,
    recording_directory,
    sampler,
    text,
    unet,
    vq,
)
from make_reference_fixtures import HostArrays, save, tensors
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from lesiondiffusion_mlx._arrays import BoolArray, HostArray, NumericArray
from lesiondiffusion_mlx._nifti import nib

sys.path.insert(0, str(REPO_ROOT))
from tests import _runner_cases as cases
from tests._fixture_models import text_weights, unet_weights, vq_weights


def report() -> dict[str, object]:
    """Construct source-style lesion attributes for a fixture."""
    return {
        "5": [
            "irregular,",
            "",
            {
                "enhancement status": "non-contrast CT,",
                "lesion location": "5",
                "shape": "irregular,",
                "size": "[[14 14 10]] pixel,",
                "density": "hypodense lesion,",
                "density variations": "heterogeneous,",
                "CT value": "Mean density of the lesion region is approximately 0.0 HU,",
                "surface characteristics": "ill-defined margin,",
                "relationship with adjacent organs": (
                    "no close relationship with surrounding organs,"
                ),
                "specific features": "",
                "cavity": "within the parenchymal organ,",
            },
        ]
    }


def record_host(root: Path, directory: Path) -> None:
    """Record host-side geometry and report behavior."""
    import tempfile

    arrays: dict[str, object] = {"organ_labels": json.dumps(organ_labels(root))}
    upstream = placement(root)
    for name, (mode, shape, size, requested, seed) in cases.CENTER_CASES.items():
        if requested is not None:
            continue  # Explicit centers are a file-runner boundary, tested directly.
        cavity = {
            "inside_parenchyma": "within the parenchymal organ",
            "protruding_from_parenchyma": "protruding from the parenchymal organ",
            "within_lumen": "within the lumen of a hollow organ",
            "wall_thickening": "wall thickening",
        }[mode]
        organ = "colon" if mode in {"within_lumen", "wall_thickening"} else "liver"
        np.random.seed(seed)
        center = upstream.generate_random_bbox_from_mask(
            cases.organ_shape(shape), "1", np.array(size), cavity, organ
        )
        arrays[f"center.{name}"] = center
        # Oversized boxes are rejected at the public boundary, not silently truncated.
        if np.any(np.array(size) > np.array(cases.CT_SHAPE)):
            continue
        with tempfile.TemporaryDirectory() as scratch:
            image = nib.Nifti1Image(cases.organ_shape(shape), np.eye(4))
            path = upstream.create_bbox_mask(image, center, np.array(size), scratch)
            arrays[f"box.{name}"] = np.asanyarray(nib.load(path).dataobj)
    save(directory, "runner_host", **arrays)


def selection(root: Path, samples: HostArray, box: HostArray) -> HostArray:
    """Construct the source lesion-selection fixture."""
    class MaskBase:
        def log_images(
            self, batch: dict[str, object], *args: object, **kwargs: object
        ) -> dict[str, object]:
            """Return fixture lesion samples and their inputs."""
            return {"samples": from_numpy(samples[None, None]), "inputs": batch["seg"]}

    classes = inference_classes(root, MaskBase, type("InpaintingBase", (), {}))
    network = classes.InferMaskDiffusion.__new__(classes.InferMaskDiffusion)
    network.save_dataset = False
    network.eval_scheme = []
    batch: dict[str, object] = dict(
        max_mask_num={"max_mask_num": 1},
        crop_size={"crop_size": samples.shape},
        image_path={"image_path": "demo/liver/tumor/ct.nii.gz"},
        label=from_numpy(box[None, None].astype(np.uint8) * 2),
        seg=torch.ones((1, 1, *samples.shape)),
    )
    _, logs = network.log_images(batch)
    return np.asarray(logs["samples"])[0, 0] == 2


def record_selection(root: Path, directory: Path) -> None:
    """Record source lesion-component selection outputs."""
    samples, box = cases.lesion_samples()
    save(
        directory,
        "selection",
        samples=samples,
        box=box,
        selected=selection(root, samples, box),
        empty=selection(root, samples * 0.1, box),
    )


def record_generation(root: Path, directory: Path) -> None:
    """Record source generation stages for a fixture."""
    from importlib import import_module

    make_beta_schedule = import_module("ldm.modules.diffusionmodules.util").make_beta_schedule

    rng = np.random.default_rng(4)
    betas = make_beta_schedule("linear", 1000, linear_start=0.00085, linear_end=0.012)
    alphas = np.cumprod(1 - betas).astype(np.float32)
    lmnet = unet(
        root,
        tensors(unet_weights(11, HostArrays(), np.float32, skip_context=False)),
        inpainting=False,
    )
    linet = unet(
        root,
        tensors(unet_weights(12, HostArrays(), np.float16, skip_context=True)),
        inpainting=True,
    )
    autoencoder = vq(root, tensors(vq_weights(13, HostArrays(), np.float16)))
    encoder = text(root, tensors(text_weights(14, HostArrays())))
    tokenizer = Tokenizer(
        WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "liver": 2, "lesion": 3, "irregular": 4, ",": 5}, "[UNK]"
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    cast("TokenizerContract", tokenizer).enable_truncation(max_length=256)
    cast("TokenizerContract", tokenizer).enable_padding(length=256, pad_id=0, pad_token="[PAD]")
    records: dict[str, object] = {
        "alphas": alphas,
        "tokenizer": tokenizer.to_str(),
        "input_seed": 4,
        "lmnet_weight_seed": 11,
        "linet_weight_seed": 12,
        "vq_weight_seed": 13,
        "text_weight_seed": 14,
    }
    np.random.seed(4)
    center = placement(root).generate_random_bbox_from_mask(
        cases.organ_block(), "1", np.array((14, 14, 10)), "within the parenchymal organ", "liver"
    )
    box = np.zeros(cases.CT_SHAPE, np.uint8)
    half = np.array((7, 7, 5))
    low, high = center - half, center + half
    box[tuple(slice(a, b) for a, b in zip(low, high, strict=True))] = 2
    records.update(center=center, box=box)
    image = cases.ct_data()[::-1].transpose(2, 1, 0).copy()
    organ = cases.organ_block()[::-1].transpose(2, 1, 0).copy() * 5
    source_box = box[::-1].transpose(2, 1, 0).copy()
    records["ct"] = cases.ct_data()

    lesion = generate_mask(root, records, lmnet, encoder, tokenizer, alphas, rng, organ, source_box)
    records["lesion"] = lesion.transpose(2, 1, 0)[::-1].copy().astype(np.uint8)
    records["output"] = generate_image(
        root, records, linet, autoencoder, alphas, rng, image, organ, lesion
    )
    save(directory, "generation", **records)


def generate_mask(
    root: Path,
    records: dict[str, object],
    lmnet: UnetNetwork,
    encoder: torch.nn.Module,
    tokenizer: Tokenizer,
    alphas: HostArray,
    rng: np.random.Generator,
    organ: HostArray,
    source_box: HostArray,
) -> BoolArray:
    """Run and record the source lesion-mask stage."""
    class MaskBase:
        def log_images(
            self, batch: dict[str, object], *args: object, **kwargs: object
        ) -> dict[str, object]:
            """Capture mask-stage conditioning and return fixture samples."""
            attrs = cast(list[dict[str, str]], batch["text"])[0]
            ids = torch.tensor([item.ids for item in tokenizer.encode_batch(list(attrs.values()))])
            condition = encoder(ids)[None]
            conditioning = torch.stack(
                (
                    cast(torch.Tensor, batch["seg"])[0, 0],
                    cast(torch.Tensor, batch["target"])[0, 0],
                    cast(torch.Tensor, batch["label"])[0, 0],
                )
            )[None].float()
            attempt = len([key for key in records if key.startswith("lm_initial.")])
            shape = (1, 1, 8, 12, 12)
            initial = rng.standard_normal(shape).astype(np.float32)
            noise = rng.standard_normal((10, *shape)).astype(np.float32)
            records.update(
                {
                    f"lm_initial.{attempt}": initial,
                    f"lm_noise.{attempt}": noise,
                    "lm_attributes": json.dumps(attrs),
                    "lm_condition": condition,
                }
            )
            _, result = sampler(
                root,
                alphas,
                lmnet,
                conditioning,
                condition,
                steps=10,
                initial=from_numpy(initial),
                noise=from_numpy(noise),
            )
            return {"samples": result, "inputs": batch["seg"]}

    classes = inference_classes(root, MaskBase, type("InpaintingBase", (), {}))
    mask_model = classes.InferMaskDiffusion.__new__(classes.InferMaskDiffusion)
    mask_model.save_dataset = False
    mask_model.eval_scheme = []
    batch: dict[str, object] = dict(
        max_mask_num={"max_mask_num": 1},
        crop_size={"crop_size": (8, 12, 12)},
        image_path={"image_path": "demo/liver/tumor/ct.nii.gz"},
        label=from_numpy(source_box[None, None]),
        seg=from_numpy(organ[None, None]),
        coarseg=from_numpy(organ[None, None]),
        type=copy.deepcopy(report()),
    )
    _, logs = mask_model.log_images(batch)
    lesion = np.asarray(logs["samples"])[0, 0] == 2
    return cast(BoolArray, lesion)


def source_default[T](value: T | None, other: T) -> T:
    """Return a source default when an optional value is absent."""
    return other if value is None else value


def identity_latent(z: torch.Tensor) -> torch.Tensor:
    """Return a latent unchanged for reference recording."""
    return z


def inpainting_class(root: Path) -> type[InpaintingNetwork]:
    """Construct the upstream inpainting reference class."""
    class InpaintingBase:
        pass

    # Execute v2 get_input/log_images with framework hooks, preserving its crop,
    # nearest mask resizing, latent assignment and full-volume reconstruction.
    from scipy import ndimage

    v2 = definitions(
        root,
        "pipeline/ldm/models/diffusion/ddpm.py",
        ("InpaintingDiffusion_v2",),
        dict(
            torch=torch,
            np=np,
            ndm=ndimage,
            LatentDiffusion=InpaintingBase,
            InpaintingDiffusion=InpaintingBase,
            rearrange=__import__("einops").rearrange,
            default=source_default,
            totalseg_class=organ_labels(root),
        ),
    )
    return cast(
        type[InpaintingNetwork],
        inference_classes(
            root, type("MaskBase", (), {}), v2.InpaintingDiffusion_v2
        ).InferInpaintingDiffusion_v2,
    )


def inpainting_network(
    root: Path, records: dict[str, object], autoencoder: VqNetwork
) -> InpaintingNetwork:
    """Construct an inpainting network instrumented for fixture recording."""
    network_class = inpainting_class(root)
    lin_model = network_class.__new__(network_class)
    lin_model.device = torch.device("cpu")
    lin_model.is_conditional_first_stage = False
    lin_model.conditioning_key = None
    lin_model.first_stage_key = "image"
    lin_model.first_stage_model = autoencoder

    def encode(x: torch.Tensor, c: object) -> torch.Tensor:
        """Encode a fixture image into the reference latent space."""
        del c
        records["normalized"] = x.clone()
        result = autoencoder.encode(x)
        records["encoded"] = result.clone()
        return result

    lin_model.encode_first_stage = encode
    lin_model.get_first_stage_encoding = identity_latent
    lin_model.ema_scope = nullcontext
    lin_model.save_dataset = False
    lin_model.eval_scheme = []

    def decode(z: torch.Tensor, c: object, **kwargs: object) -> torch.Tensor:
        """Decode a reference latent for fixture recording."""
        del c, kwargs
        records["final_latent"] = z.clone()
        records["quantized"] = autoencoder.quantize(z)[0]
        return autoencoder.decode(z)

    lin_model.decode_first_stage = decode
    return lin_model


def generate_image(
    root: Path,
    records: dict[str, object],
    linet: UnetNetwork,
    autoencoder: VqNetwork,
    alphas: HostArray,
    rng: np.random.Generator,
    image: NumericArray,
    organ: HostArray,
    lesion: NumericArray,
) -> HostArray:
    """Run and record the source lesion-image stage."""
    lin_model = inpainting_network(root, records, autoencoder)

    def sample_log(**kwargs: object) -> tuple[torch.Tensor, None]:
        """Record sampling inputs and return fixture latent output."""
        latent, mask = cast(torch.Tensor, kwargs["x0"]), cast(torch.Tensor, kwargs["mask"])
        shape = tuple(latent.shape)
        initial = rng.standard_normal(shape).astype(np.float32)
        noise = rng.standard_normal((10, *shape)).astype(np.float32)
        records.update(
            lin_initial=initial, lin_noise=noise, latent_crop=latent.clone(), latent_mask=mask
        )
        _, result = sampler(
            root,
            alphas.astype(np.float16),
            linet,
            latent,
            None,
            steps=10,
            initial=from_numpy(initial),
            noise=from_numpy(noise),
            mask=mask,
        )
        records["sampled_crop"] = result.clone()
        return torch.cat((result, mask), dim=1), None

    lin_model.sample_log = sample_log
    data = data_functions(root)
    normalized = data.img_multi_thresh_normalized(image, [-1000, -200, 200, 1000], [0, 0.2, 0.8, 1])
    batch: dict[str, object] = dict(
        image=from_numpy(normalized[None, None]),
        label=from_numpy(lesion[None, None] * 2),
        coarseg=from_numpy(organ[None, None]),
        type=[copy.deepcopy(report())],
        crop_size=[{"crop_size": (16, 16, 16)}],
        max_mask_num=[{"max_mask_num": 1}],
        image_path=[{"image_path": "ct.nii.gz"}],
    )
    _, logs = lin_model.log_images(batch, ddim_steps=10)
    result = data.img_multi_thresh_normalized(
        logs["samples"].numpy()[0, 0], [0, 0.2, 0.8, 1], [-1000, -200, 200, 1000]
    )
    return result.transpose(2, 1, 0)[::-1].copy()


def main() -> None:
    """Parse options and record source runner fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.upstream_root)
    directory = recording_directory()
    with torch.no_grad():
        record_host(args.upstream_root, directory)
        record_selection(args.upstream_root, directory)
        record_generation(args.upstream_root, directory)


if __name__ == "__main__":
    main()
