"""Record seeded upstream Torch CPU fixtures; no package inference imports.

Requires a source directory matching reference.json, including OpenCLIP 2.24.0's
hf_model.py and hf_configs.py in open_clip/. Fetch it with scripts/fetch_reference.py;
see "Regenerating reference fixtures" in docs/technical.md.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import TokenizerContract

import numpy as np
import torch
from _torch import from_numpy
from _upstream import (
    REFERENCE,
    REPO_ROOT,
    crop_reference,
    data_functions,
    prepare,
    recording_directory,
    sampler,
    text,
    unet,
    vq,
)
from numpy.typing import ArrayLike
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from lesiondiffusion_mlx._arrays import HostArray

sys.path.insert(0, str(REPO_ROOT))
from tests._fixture_models import (
    sampling_inputs,
    text_weights,
    unet_weights,
    vq_weights,
)


class HostArrays:
    """Weight builder output adapter; all random arrays retain the original seeds."""

    float32: type[np.float16] | type[np.float32] = np.float32
    float16: type[np.float16] | type[np.float32] = np.float16

    @staticmethod
    def array(value: HostArray, /, *, dtype: type[np.float16] | type[np.float32]) -> HostArray:
        """Construct a host fixture array with an optional dtype."""
        return np.asarray(value, dtype=dtype)


def tensors(weights: dict[str, HostArray]) -> dict[str, torch.Tensor]:
    """Convert fixture weights to reference Torch tensors."""
    return {key: from_numpy(value) for key, value in weights.items()}


def save(directory: Path, name: str, **arrays: object) -> None:
    """Write reference arrays with their version and source metadata."""
    from importlib.metadata import version

    payload = dict(
        schema_version=np.array(1),
        upstream_revision=np.array(REFERENCE["revision"]),
        text_revision=np.array(REFERENCE["text_revision"]),
        backend=np.array("torch-cpu"),
        recording_platform=np.array("darwin-arm64"),
        torch_version=np.array(torch.__version__),
        numpy_version=np.array(np.__version__),
        transformers_version=np.array(version("transformers")),
        **arrays,
    )
    path = directory / (name + ".npz")
    np.savez_compressed(
        path,
        allow_pickle=False,
        **{
            key: value.detach().numpy()
            if isinstance(value, torch.Tensor)
            else cast(ArrayLike, value)
            for key, value in payload.items()
        },
    )
    if path.stat().st_size >= 1_000_000:
        raise ValueError(f"Fixture exceeds size budget: {path}")
    print(f"Recorded {name}: {path.stat().st_size} bytes", flush=True)


def record_networks(root: Path, directory: Path) -> None:
    """Record seeded upstream network outputs."""
    for inpainting in (False, True):
        seed = 102 if inpainting else 101
        dtype = np.float16 if inpainting else np.float32
        rng = np.random.default_rng(seed + 1000)
        weights = tensors(unet_weights(seed, HostArrays(), dtype, skip_context=inpainting))
        network = unet(root, weights, inpainting=inpainting)
        x = rng.standard_normal((1, 5 if inpainting else 4, 4, 4, 4)).astype(np.float32)
        context = rng.standard_normal((1, 3, 6)).astype(np.float32)
        timesteps = np.array([17], dtype=np.int32)
        output = network(from_numpy(x), from_numpy(timesteps), from_numpy(context))
        config = dict(
            model_channels=32,
            num_res_blocks=1,
            attention_resolutions=[1, 2],
            channel_mult=[1, 2],
            num_head_channels=16,
            skip_context=inpainting,
        )
        save(
            directory,
            "unet_inpainting" if inpainting else "unet_mask",
            weight_seed=seed,
            config=json.dumps(config),
            dtype="float16" if inpainting else "float32",
            x=x,
            context=context,
            timesteps=timesteps,
            output=output,
        )
    for dtype in (np.float32, np.float16):
        rng = np.random.default_rng(1201)
        network = vq(root, tensors(vq_weights(201, HostArrays(), dtype)))
        x = rng.standard_normal((1, 1, 4, 4, 4)).astype(np.float32)
        z = rng.standard_normal((1, 4, 2, 2, 2)).astype(np.float32)
        tx, tz = from_numpy(x.astype(dtype)), from_numpy(z.astype(dtype))
        save(
            directory,
            "vq_" + np.dtype(dtype).name,
            weight_seed=201,
            dtype=np.dtype(dtype).name,
            x=x,
            z=z,
            encoded=network.encode(tx),
            quantized=network.quantize(tz)[0],
            decoded=network.decode(tz),
            decoded_unquantized=network.decode(tz, force_not_quantize=True),
        )


def record_activations(directory: Path) -> None:
    """Record source activation outputs across input values."""
    x = torch.linspace(-8, 8, 4096).half()
    logits = from_numpy(np.random.default_rng(107).normal(size=(2, 4096)).astype(np.float16))
    save(
        directory,
        "unet_activations",
        silu_input=x,
        silu_output=torch.nn.SiLU()(x),
        softmax_input=logits,
        softmax_output=torch.nn.Softmax(dim=-1)(logits),
    )


def record_text(root: Path, directory: Path) -> None:
    """Record seeded source text-encoder outputs."""
    network = text(root, tensors(text_weights(301, HostArrays())))
    tokens = np.array([[2, 3, 4, 0], [4, 3, 0, 0]], dtype=np.int32)
    save(directory, "text", weight_seed=301, tokens=tokens, output=network(from_numpy(tokens)))
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "liver": 2, "stone": 3}, "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    cast("TokenizerContract", tokenizer).enable_truncation(max_length=256)
    cast("TokenizerContract", tokenizer).enable_padding(length=256, pad_id=0, pad_token="[PAD]")
    attributes = {"location": "liver", "description": "stone " * 300}
    ids = torch.tensor([token.ids for token in tokenizer.encode_batch(list(attributes.values()))])
    save(
        directory,
        "attributes",
        weight_seed=301,
        attributes=json.dumps(attributes),
        tokenizer=tokenizer.to_str(),
        output=network(ids)[None],
    )
    del network
    gc.collect()


def record_sampling(root: Path, directory: Path) -> None:
    """Record seeded source diffusion sampling trajectories."""
    for inpainting in (False, True):
        inputs = sampling_inputs(401, inpainting=inpainting)

        def denoiser(
            x: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor | None
        ) -> torch.Tensor:
            """Apply the fixture denoiser to a conditioned sample."""
            assert condition is not None
            x = x[:, :1] if x.shape[1] == 4 else x
            return (
                x * 0.125
                + timestep[:, None, None, None, None].float() * 0.001
                + condition.sum() * 0.01
            )

        reference, output = sampler(
            root,
            inputs["alphas"],
            denoiser,
            from_numpy(inputs["image"]),
            from_numpy(inputs["condition"]),
            steps=4,
            initial=from_numpy(inputs["initial"]),
            noise=from_numpy(inputs["noise"]),
            mask=from_numpy(inputs["mask"]) if inpainting else None,
        )
        save(
            directory,
            "ddim_inpainting" if inpainting else "ddim_mask",
            **inputs,
            steps=4,
            output=output,
            timesteps=reference.ddim_timesteps,
            alphas_selected=reference.ddim_alphas.float(),
            previous=reference.ddim_alphas_prev.astype(np.float32),
            sqrt_one_minus=reference.ddim_sqrt_one_minus_alphas.float(),
            sigmas=reference.ddim_sigmas.float(),
        )


def record_pipeline(root: Path, directory: Path) -> None:
    """Record the source pipeline on a synthetic fixture."""
    import tempfile

    from lesiondiffusion_mlx._nifti import nib

    data = data_functions(root)
    rng = np.random.default_rng(501)
    orientation = rng.standard_normal((3, 4, 5)).astype(np.float32)
    crop_mask = np.zeros((7, 8, 9), dtype=bool)
    crop_mask[0:2, 6:8, 4:6] = True
    ct = np.array([-1100, -1000, -999, -200, -199, 0, 199, 200, 201, 999, 1000, 1100], np.float32)
    normalized = np.array([-0.1, 0, 0.1, 0.2, 0.5, 0.8, 0.9, 1, 1.1], np.float32)
    odd = rng.standard_normal((1, 1, 5, 3, 3)).astype(np.float32)
    long = rng.standard_normal((1, 1, 97, 2, 2)).astype(np.float32)
    network = vq(root, tensors(vq_weights(201, HostArrays(), np.float16)))
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "ct.nii.gz"
        nib.save(nib.Nifti1Image(orientation, np.eye(4)), path)
        source = data.load_nifty_volume_as_array(str(path))["data"].astype(np.float32)
    save(
        directory,
        "pipeline",
        weight_seed=201,
        orientation=orientation,
        source=source,
        crop_mask=crop_mask,
        crop_shape=np.array((4, 4, 4)),
        crop=crop_reference(root, crop_mask, (4, 4, 4)),
        ras=source.transpose(2, 1, 0)[::-1].copy(),
        ct=ct,
        normalized=normalized,
        ct_output=data.img_multi_thresh_normalized(ct, [-1000, -200, 200, 1000], [0, 0.2, 0.8, 1]),
        inverse_output=data.img_multi_thresh_normalized(
            normalized, [0, 0.2, 0.8, 1], [-1000, -200, 200, 1000]
        ),
        odd_input=odd,
        odd_output=network.encode(from_numpy(odd).half()),
        long_input=long,
        long_output=network.encode(from_numpy(long).half()),
    )


def main() -> None:
    """Parse recorder options and write reference fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.upstream_root)
    directory = recording_directory()
    with torch.no_grad():
        record_networks(args.upstream_root, directory)
        record_activations(directory)
        record_text(args.upstream_root, directory)
        record_sampling(args.upstream_root, directory)
        record_pipeline(args.upstream_root, directory)


if __name__ == "__main__":
    main()
