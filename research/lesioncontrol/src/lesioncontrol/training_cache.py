"""Cache real HCC features, frozen MLX VQ latents and ordered LINet context.

The CUDA research trainer consumes these backend-neutral arrays. Only real
training crops are encoded here; clean-host generation inputs stay separate.
Features use liver-relative HU, while the numeric text token retains the actual
absolute tumor mean. Feature scaling is fitted on training patients only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import mlx.core as mx
import nibabel as nib
import numpy as np
from lesiondiffusion_mlx.medmlx import load_stage
from lesiondiffusion_mlx.mlx_model import BiomedTextEncoder, VqAutoencoder
from lesiondiffusion_mlx.mlx_pipeline import (
    encode_attributes,
    encode_source,
    piecewise_ct,
    ras_to_source,
)

from lesioncontrol.backend import require_metal
from lesioncontrol.learned import CONTEXT_FIELDS
from lesioncontrol.profile import build_profile, default_settings, save_profile

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Mlx


class PreparedCase(TypedDict):
    """Prepared-data manifest fields used by the frozen encoder boundary."""

    id: str
    patient: str
    partition: Literal["train", "selection"]
    real_ct: str
    real_mask: str
    real_liver_mask: str


class BundleCase(TypedDict):
    """One portable training sample; all paths are relative to the bundle."""

    patient_id: str
    lesion_id: str
    split: Literal["train", "selection"]
    file: str


def _attributes(mean_hu: float) -> dict[str, object]:
    """Use the ten training fields in the production report's insertion order."""
    fields: dict[str, object] = {
        "enhancement status": "enhanced CT,",
        "lesion location": "liver",
        "shape": "irregular,",
        "density": "isodense lesion,",
        "density variations": "heterogeneous,",
        "CT value": f"{float(mean_hu)} HU,",
        "surface characteristics": "well-defined margin,",
        "relationship with adjacent organs": "no close relationship with surrounding organs,",
        "specific features": "",
        "cavity": "within the parenchymal organ,",
    }
    if tuple(fields) != CONTEXT_FIELDS:
        raise ValueError("Cache context differs from the declared ten-field training order")
    return fields


def _file_record(path: Path) -> dict[str, object]:
    """Identify a local file, including replacements that preserve size and mtime."""
    path = path.resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime": stat.st_mtime_ns,
        "ctime": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _model_identity(path: Path) -> dict[str, object]:
    """Stream a functional cache key once for each consumed checkpoint/tokenizer file."""
    path = path.resolve()
    identity = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            identity.update(block)
    return {"path": str(path), "content_key": identity.hexdigest()}


def _bundle_provenance(weights: Path) -> dict[str, object]:
    """Record the frozen encoder, tokenizer, extractor and preprocessing contract once."""
    from lesiondiffusion_mlx import mlx_model, mlx_pipeline

    from lesioncontrol import profile

    return {
        "schema": "lesioncontrol.training_cache_provenance.v2",
        "model_files": [
            _model_identity(weights / name) for name in ("model.safetensors", "tokenizer.json")
        ],
        "implementation_files": [
            _file_record(Path(__file__)),
            *[
                _file_record(Path(module.__file__))
                for module in (mlx_model, mlx_pipeline, profile)
                if module.__file__ is not None
            ],
        ],
        "extractor_settings": asdict(default_settings()),
        "target_space": "liver_contrast",
        "context_fields": list(CONTEXT_FIELDS),
        "preprocessing": "RAS to source; piecewise CT; FP16 input and conditioned VQ encoder",
        "latent_quantized": False,
        "saved_dtype": "float32",
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "mlx",
                "mlxradiomics",
                "lesiondiffusion-mlx",
                "numpy",
                "safetensors",
                "tokenizers",
            )
        },
    }


def cache_training_data(
    manifest_path: Path, weights: Path, output: Path, maximum_cases: int | None = None
) -> Path:
    """Encode an owned prepared dataset and fit equal-patient training feature scales.

    Completed samples require unchanged input/model/tokenizer file identities,
    encoder/preprocessing code, extraction settings and package versions. A capped
    cache is explicitly marked incomplete and is intended only for real-data CUDA
    preflight, never full adapter training.
    """
    runtime = cast("Mlx", mx)
    payload = cast(dict[str, object], json.loads(manifest_path.read_text()))
    if payload["schema"] != "lesioncontrol.real_hcc_training_data.v1":
        raise ValueError("Unsupported prepared HCC training manifest")
    cases = cast(list[PreparedCase], payload["cases"])
    selected = cases if maximum_cases is None else cases[:maximum_cases]
    if not selected or maximum_cases is not None and maximum_cases < 1:
        raise ValueError("The cache requires at least one prepared real tumor")
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "provenance.json"
    # Normalize tuples to the same JSON arrays read on resume. Model content keys
    # are functional cache identity; medical inputs retain filesystem provenance.
    provenance = cast(dict[str, object], json.loads(json.dumps(_bundle_provenance(weights))))
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError(
                "Existing training cache has different model or preprocessing provenance"
            )
    elif any((output / f"{case['id']}.json").exists() for case in selected):
        raise ValueError("Existing training cache lacks bundle provenance; rebuild it")
    else:
        provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    records: list[BundleCase] = []
    vectors: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
    feature_names: list[str] = []
    encoder: VqAutoencoder | None = None
    text: BiomedTextEncoder | None = None
    tokenizer = weights / "tokenizer.json"
    for case in selected:
        if case["partition"] not in ("train", "selection"):
            raise ValueError("Only the registered train/selection cohorts may be cached")
        paths = [Path(case[name]) for name in ("real_ct", "real_mask", "real_liver_mask")]
        inputs = [_file_record(path) for path in paths]
        destination = output / f"{case['id']}.npz"
        record_path = output / f"{case['id']}.json"
        if record_path.exists():
            completed = cast(dict[str, object], json.loads(record_path.read_text()))
            if completed["input_files"] != inputs or completed["weights"] != str(weights.resolve()):
                raise ValueError("Existing encoded sample has different input or model provenance")
            feature_names = cast(list[str], completed["feature_names"])
            with np.load(destination, allow_pickle=False) as cached:
                vector = np.asarray(cached["features"], dtype=np.float64)
        else:
            if destination.exists():
                raise FileExistsError(
                    f"Incomplete encoded sample requires inspection: {destination}"
                )
            if encoder is None or text is None:
                require_metal()
                encoder = VqAutoencoder(
                    load_stage(
                        weights,
                        "vq",
                        runtime,
                        dtype=mx.float16,
                        prefixes=("encoder.", "quant_conv."),
                    ),
                    runtime,
                    source_arithmetic=False,
                )
                text = BiomedTextEncoder(
                    load_stage(weights, "linet_text", runtime, dtype=mx.float32), runtime
                )
                runtime.eval(*encoder.weights.values(), *text.weights.values())
            profile = build_profile(*paths, target_space="liver_contrast")
            save_profile(profile, output / f"{case['id']}.profile.json")
            feature_names = sorted(profile.features)
            image = np.asarray(
                nib.Nifti1Image.from_filename(str(paths[0])).dataobj, dtype=np.float32
            )
            mask = np.asarray(nib.Nifti1Image.from_filename(str(paths[1])).dataobj) > 0
            mean_hu = float(np.mean(image[mask], dtype=np.float64))
            source_image = np.asarray(ras_to_source(image), dtype=np.float32)
            source_mask = np.asarray(ras_to_source(mask), dtype=np.bool_)
            fine = source_mask[::2, ::2, ::2][None, None].astype(np.float32)
            if not fine.any():
                raise ValueError("Real lesion disappeared on the half-resolution latent grid")
            # Upstream forms its loss-normalizing box after nearest downsampling.
            points = np.argwhere(fine[0, 0] > 0)
            low, high = points.min(axis=0), points.max(axis=0) + 1
            coarse = np.zeros_like(fine)
            coarse[0, 0, low[0] : high[0], low[1] : high[1], low[2] : high[2]] = 1
            latent = encode_source(
                encoder,
                piecewise_ct(mx.array(source_image[None, None]), runtime).astype(mx.float16),
                runtime,
            ).astype(mx.float32)
            context = encode_attributes(_attributes(mean_hu), text, tokenizer, runtime)
            runtime.eval(latent, context)
            if latent.shape != (1, 4, 64, 64, 64) or context.shape != (1, 10, 512):
                raise ValueError("Encoded real tumor differs from the pinned LINet training shapes")
            arrays = {
                "latent": np.array(latent, dtype=np.float32),
                "fine_mask": fine,
                "coarse_mask": coarse,
                "context": np.array(context, dtype=np.float32),
                "features": np.asarray(
                    [profile.features[name] for name in feature_names], dtype=np.float32
                ),
            }
            if not all(np.isfinite(array).all() for array in arrays.values()):
                raise ValueError("Real training cache contains non-finite arrays")
            temporary = destination.with_suffix(".partial")
            with temporary.open("wb") as stream:
                np.savez(
                    stream,
                    latent=arrays["latent"],
                    fine_mask=arrays["fine_mask"],
                    coarse_mask=arrays["coarse_mask"],
                    context=arrays["context"],
                    features=arrays["features"],
                    allow_pickle=False,
                )
            temporary.replace(destination)
            completed = {
                "input_files": inputs,
                "weights": str(weights.resolve()),
                "feature_names": feature_names,
                "target_space": "liver_contrast",
                "context_fields": list(CONTEXT_FIELDS),
                "context_attributes": _attributes(mean_hu),
                "encoder_dtype": "float16, matching production",
                "saved_dtype": "float32",
                "latent_quantized": False,
            }
            record_path.write_text(json.dumps(completed, indent=2, allow_nan=False) + "\n")
            vector = arrays["features"].astype(np.float64)
        records.append(
            {
                "patient_id": case["patient"],
                "lesion_id": case["id"],
                "split": case["partition"],
                "file": destination.name,
            }
        )
        vectors.append(vector)
        print(f"CACHED {case['id']} {case['partition']}", flush=True)
    train = [index for index, record in enumerate(records) if record["split"] == "train"]
    if not train:
        raise ValueError("No registered training patients are present")
    patient_counts = Counter(records[index]["patient_id"] for index in train)
    sample_weights = np.asarray(
        [1 / patient_counts[records[index]["patient_id"]] for index in train]
    )
    matrix = np.stack([vectors[index] for index in train])
    center = np.average(matrix, axis=0, weights=sample_weights)
    scale = np.sqrt(np.average((matrix - center) ** 2, axis=0, weights=sample_weights))
    constant = scale == 0
    scale[constant] = 1
    schedule_path = output / "alphas_cumprod.npy"
    if not schedule_path.exists():
        require_metal()
        schedule = load_stage(
            weights, "linet", runtime, dtype=mx.float32, prefixes=("alphas_cumprod",)
        )["alphas_cumprod"]
        runtime.eval(schedule)
        np.save(schedule_path, np.array(schedule, dtype=np.float32), allow_pickle=False)
    index = {
        "schema": "lesioncontrol.adapter_training_bundle.v1",
        "dataset_complete": maximum_cases is None,
        "source_manifest": str(manifest_path.resolve()),
        "provenance_file": provenance_path.name,
        "target_space": "liver_contrast",
        "feature_names": feature_names,
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
        "constant_features": [
            name for name, fixed in zip(feature_names, constant, strict=True) if fixed
        ],
        "normalization": (
            "train-only mean/std; equal total weight per training patient; "
            "constant scales one; no clipping"
        ),
        "context_fields": list(CONTEXT_FIELDS),
        "schedule_file": "alphas_cumprod.npy",
        "cases": records,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlxradiomics", "lesiondiffusion-mlx", "numpy")
        },
    }
    result = output / "index.json"
    result.write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-cases", type=int)
    args = parser.parse_args()
    print(cache_training_data(args.manifest, args.weights, args.output, args.maximum_cases))


if __name__ == "__main__":
    main()
