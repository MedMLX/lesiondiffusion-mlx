"""Frozen cache provenance and completed-resume model lifetime contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Protocol, cast

import mlx.core as mx
import nibabel as nib
import numpy as np
import pytest
from lesiondiffusion_mlx.mlx_model import BiomedTextEncoder, VqAutoencoder

from lesioncontrol import training_cache
from lesioncontrol.profile import ReferenceProfile, default_settings

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Mlx


class _TensorWriter(Protocol):
    def save_safetensors(self, file: str, arrays: dict[str, mx.array]) -> None: ...


class _Writable(Protocol):
    def to_filename(self, filename: str) -> None: ...


@pytest.mark.parametrize("replacement", [None, "model.safetensors", "tokenizer.json"])
def test_completed_cache_avoids_models_and_rejects_replaced_assets(
    replacement: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed cache needs no GPU model, and same-path replacements invalidate it."""
    weights = tmp_path / "weights"
    weights.mkdir()
    cast(_TensorWriter, mx).save_safetensors(
        str(weights / "model.safetensors"),
        {"vq.encoder.unused": mx.zeros((1,)), "linet.alphas_cumprod": mx.ones((1000,))},
    )
    (weights / "tokenizer.json").write_text('{"fixture":true}')
    image = np.full((128, 128, 128), 80, dtype=np.float32)
    mask = np.zeros_like(image)
    mask[56:72, 56:72, 56:72] = 1
    paths = [tmp_path / f"{name}.nii.gz" for name in ("ct", "mask", "liver")]
    for path, array in zip(paths, (image, mask, np.ones_like(image)), strict=True):
        cast(_Writable, nib.Nifti1Image(array, np.eye(4))).to_filename(str(path))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "lesioncontrol.real_hcc_training_data.v1",
                "cases": [
                    {
                        "id": "tumor",
                        "patient": "patient",
                        "partition": "train",
                        "real_ct": str(paths[0]),
                        "real_mask": str(paths[1]),
                        "real_liver_mask": str(paths[2]),
                    }
                ],
            }
        )
    )
    settings = default_settings()
    features = {f"original_{name}": float(i + 1) for i, name in enumerate(settings.features)}
    profile = ReferenceProfile(features, "liver_contrast", settings, {})

    def fixed_profile(*_paths: Path, target_space: str) -> ReferenceProfile:
        assert target_space == "liver_contrast"
        return profile

    def frozen_encoder(model: VqAutoencoder, image: mx.array, runtime: Mlx) -> mx.array:
        return runtime.full((1, 4, 64, 64, 64), 1.5, dtype=runtime.float32)

    def frozen_context(
        attributes: dict[str, object], text: BiomedTextEncoder, tokenizer: Path, runtime: Mlx
    ) -> mx.array:
        assert attributes["CT value"] == "80.0 HU,"
        return runtime.full((1, 10, 512), 2.5, dtype=runtime.float32)

    monkeypatch.setattr(training_cache, "build_profile", fixed_profile)
    monkeypatch.setattr(training_cache, "encode_source", frozen_encoder)
    monkeypatch.setattr(training_cache, "encode_attributes", frozen_context)
    output = tmp_path / "cache"
    result = training_cache.cache_training_data(manifest, weights, output)
    first = json.loads(result.read_text())
    expected = [features[name] for name in sorted(features)]
    assert first["feature_center"] == expected
    assert first["feature_scale"] == [1.0] * len(expected)
    with np.load(output / "tumor.npz", allow_pickle=False) as sample:
        np.testing.assert_array_equal(sample["latent"], np.full((1, 4, 64, 64, 64), 1.5))
    if replacement is not None:
        path = weights / replacement
        stat = path.stat()
        path.write_bytes(path.read_bytes()[::-1])
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("completed cache initialized Metal or loaded model weights")

    monkeypatch.setattr(training_cache, "load_stage", forbidden)
    monkeypatch.setattr(training_cache, "require_metal", forbidden)
    if replacement is not None:
        with pytest.raises(ValueError, match="different model or preprocessing provenance"):
            training_cache.cache_training_data(manifest, weights, output)
    else:
        resumed = training_cache.cache_training_data(manifest, weights, output)
        assert cast(dict[str, object], json.loads(resumed.read_text())) == first
