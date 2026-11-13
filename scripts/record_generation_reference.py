"""Record reduced upstream LesionDiffusion generation on an HCC-TACE CT crop.

The adapter supplies constructors for the pinned source networks and CPU DDIM;
model parameters and noise are the same reproducible inputs used by the tests.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from tests._fixture_models import unet_weights, vq_weights
from tests._runner_cases import ras_affine, real_scan_data

REVISION = "880736e2b8a4d9becdb6594da7f8af6db19d04d1"


class HostFactory:
    """Supply NumPy arrays to the shared deterministic weight generator."""

    float32 = np.float32

    def array(self, value: np.ndarray, *, dtype: type[np.float32]) -> np.ndarray:
        return np.asarray(value, dtype=dtype)


def crop_window(mask: np.ndarray, shape: tuple[int, int, int]) -> tuple[slice, ...]:
    """Match the upstream foreground-center crop and clamp it to the image."""
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("The source crop mask is empty")
    center = (points.min(0) + points.max(0)) // 2
    starts = [
        min(max(int(c) - width // 2, 0), length - width)
        for c, width, length in zip(center, shape, mask.shape, strict=True)
    ]
    return tuple(slice(start, start + width) for start, width in zip(starts, shape, strict=True))


def main() -> None:
    """Run pinned upstream placement, networks, sampling, quantization and decoding."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / ".cache/medmlx/test-data/lesiondiffusion-source",
    )
    parser.add_argument("--reference-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("work/real-reference"))
    args = parser.parse_args()
    revision = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if revision != REVISION:
        raise ValueError(f"Expected {REVISION}, got {revision}")
    subprocess.run(
        ["git", "-C", str(args.source), "diff", "--exit-code", "HEAD", "--", "pipeline"],
        check=True,
        capture_output=True,
    )
    sys.path[:0] = [str(args.reference_adapter), str(args.source / "pipeline")]
    upstream = importlib.import_module("_upstream")
    torch.set_num_threads(1)
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/darwin-arm64/generation.npz"
    with np.load(fixture, allow_pickle=False) as archive:
        old = dict(archive)
    arrays = {
        key: value
        for key, value in old.items()
        if key
        not in {
            "center",
            "box",
            "ct",
            "lesion",
            "normalized",
            "encoded",
            "latent_crop",
            "latent_mask",
            "sampled_crop",
            "final_latent",
            "quantized",
            "output",
        }
    }
    source_ct, source_organ, _ = real_scan_data()
    placement = upstream.placement(args.source)
    np.random.seed(4)
    center = placement.generate_random_bbox_from_mask(
        source_organ, 1, np.array((14, 14, 10)), "within the parenchymal organ", "liver"
    )
    affine = ras_affine((30.5, -42.0, 117.0))
    with tempfile.TemporaryDirectory(prefix="lesion-reference-") as temporary:
        root = Path(temporary)
        ct_path, organ_path = root / "ct.nii.gz", root / "organ.nii.gz"
        nib.save(nib.Nifti1Image(source_ct, affine), ct_path)
        nib.save(nib.Nifti1Image(source_organ, affine), organ_path)
        box_path = placement.create_bbox_mask(
            nib.load(organ_path), center, np.array((14, 14, 10)), temporary
        )
        box_ras = np.asarray(nib.load(box_path).dataobj, dtype=np.uint8)
        source = upstream.data_functions(args.source)
        ct = source.load_nifty_volume_as_array(ct_path)["data"].astype(np.float32)
        organ = source.load_nifty_volume_as_array(organ_path)["data"] > 0
        box = source.load_nifty_volume_as_array(box_path)["data"] == 2
    lm_crop = crop_window(box, (8, 12, 12))
    conditioning = np.stack(
        (organ[lm_crop].astype(np.float64) * 5 / 104, organ[lm_crop], box[lm_crop])
    )[None].astype(np.float32)
    factory = HostFactory()
    def convert(values: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """Move deterministic reference weights into Torch CPU tensors."""
        return {key: torch.from_numpy(value) for key, value in values.items()}
    lm_weights = convert(unet_weights(11, factory, np.float32, skip_context=False))
    lmnet = upstream.unet(args.source, lm_weights, inpainting=False)
    condition = torch.from_numpy(old["lm_condition"])
    rng = np.random.default_rng(12345)
    lesion = np.zeros_like(box, dtype=bool)
    for attempt in range(2):
        initial_key, noise_key = f"lm_initial.{attempt}", f"lm_noise.{attempt}"
        if initial_key not in arrays:
            arrays[initial_key] = rng.standard_normal(old["lm_initial.0"].shape, dtype=np.float32)
            arrays[noise_key] = rng.standard_normal(old["lm_noise.0"].shape, dtype=np.float32)
        _, samples = upstream.sampler(
            args.source,
            old["alphas"],
            lmnet,
            torch.from_numpy(conditioning),
            condition,
            steps=10,
            initial=torch.from_numpy(arrays[initial_key]),
            noise=torch.from_numpy(arrays[noise_key]),
        )
        values = samples.detach().numpy()[0, 0]
        best, difference = None, float("inf")
        for threshold in np.arange(0.9, 0.2, -0.1):
            candidate = values > threshold
            labels, count = ndimage.label(candidate)
            if count:
                sizes = ndimage.sum(candidate, labels, range(count + 1))
                candidate = labels == int(np.argmax(sizes[1:]) + 1)
            current = abs(int(candidate.sum()) - int(box[lm_crop].sum()))
            if current < difference:
                best, difference = candidate.copy(), current
        lesion[lm_crop] = best
        if lesion.sum() > 400:
            break
    if not lesion.any():
        raise RuntimeError("Upstream LMNet produced no lesion")
    vq_arrays = vq_weights(13, factory, np.float32)
    vq = upstream.vq(args.source, {key: value.half() for key, value in convert(vq_arrays).items()})
    normalized = source.img_multi_thresh_normalized(ct, [-1000, -200, 200, 1000], [0, 0.2, 0.8, 1])
    with torch.no_grad():
        input_tensor = torch.from_numpy(normalized[None, None]).half()
        encoded = vq.encode(input_tensor)
        latent_mask = torch.nn.functional.interpolate(
            torch.from_numpy(lesion[None, None].astype(np.float32)),
            size=encoded.shape[2:],
            mode="nearest",
        ).half()
        lin_crop = crop_window(lesion, (16, 16, 16))
        selection = (
            slice(None),
            slice(None),
            *(slice(part.start // 2, part.stop // 2) for part in lin_crop),
        )
        latent_crop, mask_crop = encoded[selection].clone(), latent_mask[selection]
        linet = upstream.unet(
            args.source,
            convert(unet_weights(12, factory, np.float32, skip_context=True)),
            inpainting=True,
        )
        _, sampled = upstream.sampler(
            args.source,
            old["alphas"],
            linet,
            latent_crop,
            None,
            steps=10,
            initial=torch.from_numpy(old["lin_initial"]),
            noise=torch.from_numpy(old["lin_noise"]),
            mask=mask_crop,
        )
        final_latent = encoded.clone()
        final_latent[selection] = sampled.half()
        quantized, _, _ = vq.quantize(final_latent)
        decoded = vq.decode(quantized, force_not_quantize=True).float().numpy()[0, 0]
    output_source = source.img_multi_thresh_normalized(
        decoded, [0, 0.2, 0.8, 1], [-1000, -200, 200, 1000]
    )
    output_ras = np.ascontiguousarray(output_source.transpose(2, 1, 0)[::-1])
    lesion_ras = np.ascontiguousarray(lesion.transpose(2, 1, 0)[::-1]).astype(np.uint8)
    real = dict(
        center=center,
        box=box_ras,
        ct=source_ct,
        lesion=lesion_ras,
        normalized=input_tensor.numpy(),
        encoded=encoded.numpy(),
        latent_crop=latent_crop.numpy(),
        latent_mask=mask_crop.numpy(),
        sampled_crop=sampled.numpy(),
        final_latent=final_latent.numpy(),
        quantized=quantized.numpy(),
        output=output_ras,
        upstream_revision=np.asarray(revision),
        torch_version=np.asarray(torch.__version__),
        numpy_version=np.asarray(np.__version__),
        input_source=np.asarray("HCC-TACE-Seg HCC_063 contrast CT, liver and tumor"),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "generation.npz", **real)
    np.savez_compressed(fixture, **arrays)


if __name__ == "__main__":
    main()
