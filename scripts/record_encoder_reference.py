"""Record pinned upstream VQ and image-axis references on HCC-TACE CT crops."""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Protocol, TypedDict, cast

import numpy as np
import torch

from lesiondiffusion_mlx._arrays import HostArray
from lesiondiffusion_mlx._nifti import nib
from lesiondiffusion_mlx._reference import REFERENCE
from tests._fixture_models import vq_weights
from tests._torch import from_numpy


class VqNetwork(Protocol):
    def encode(self, image: torch.Tensor) -> torch.Tensor: ...
    def decode(self, latent: torch.Tensor, *, force_not_quantize: bool = False) -> torch.Tensor: ...
    def quantize(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, object]: ...


class ImageData(TypedDict):
    data: HostArray


class DataFunctions(Protocol):
    def load_nifty_volume_as_array(self, path: Path) -> ImageData: ...
    def img_multi_thresh_normalized(
        self, image: HostArray, thresholds: list[float], levels: list[float],
    ) -> HostArray: ...


class ReferenceAdapter(Protocol):
    def vq(self, root: Path, weights: dict[str, torch.Tensor]) -> VqNetwork: ...
    def data_functions(self, root: Path) -> DataFunctions: ...


class RecordingOptions(Protocol):
    source: Path
    reference_adapter: Path
    ct: Path
    output: Path


class TorchFactory:
    """Construct the shared seeded parameters directly in the reference dtype."""

    float32 = torch.float32

    def array(self, value: HostArray, /, *, dtype: torch.dtype) -> torch.Tensor:
        """Keep NumPy's parameter sequence and cast once, as in native replay."""
        return from_numpy(value).to(dtype=dtype)


def load_arrays(path: Path) -> dict[str, HostArray]:
    """Load numeric/string arrays and require the recorded source and Torch revision."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: cast(HostArray, archive[key]) for key in archive.files}
    if str(arrays["upstream_revision"].item()) != REFERENCE["revision"]:
        raise ValueError(f"Unexpected upstream revision in {path}")
    if str(arrays["torch_version"].item()) != torch.__version__.split("+", 1)[0]:
        raise ValueError(f"Use the recorded Torch version for {path}")
    return arrays


def main() -> None:
    """Rebuild image-derived outputs with the unchanged pinned source on Torch CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-adapter", type=Path, required=True)
    parser.add_argument("--ct", type=Path,
                        default=Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz")
    parser.add_argument("--output", type=Path, default=Path("work/real-reference"))
    args = cast(RecordingOptions, parser.parse_args())
    revision = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != REFERENCE["revision"]:
        raise ValueError(f"Expected upstream {REFERENCE['revision']}, got {revision}")
    subprocess.run(
        ["git", "-C", str(args.source), "diff", "--exit-code", "HEAD", "--", "pipeline"],
        check=True, capture_output=True,
    )
    sys.path[:0] = [str(args.reference_adapter), str(args.source / "pipeline")]
    adapter = cast(ReferenceAdapter, importlib.import_module("_upstream"))
    source = adapter.data_functions(args.source)
    ct = np.asarray(nib.load(args.ct).dataobj, dtype=np.float32)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    fixtures = Path(__file__).resolve().parents[1] / "tests/fixtures/darwin-arm64"

    def crop(shape: tuple[int, int, int]) -> HostArray:
        """Read native-order source pixels; test geometry controls their axis meaning."""
        image = ct[100:100 + shape[0], 210:210 + shape[1], 55:55 + shape[2]].copy()
        if image.shape != shape:
            raise ValueError("The CT does not cover the requested encoder input")
        return image

    def normalized(shape: tuple[int, int, int]) -> HostArray:
        """Use the source's four HU thresholds before encoding."""
        return np.asarray(source.img_multi_thresh_normalized(
            crop(shape), [-1000, -200, 200, 1000], [0, 0.2, 0.8, 1],
        ), dtype=np.float32)[None, None]

    for name in ("vq_float32", "vq_float16", "pipeline"):
        arrays = load_arrays(fixtures / f"{name}.npz")
        dtype = torch.float32 if name == "vq_float32" else torch.float16
        network = adapter.vq(args.source, vq_weights(
            int(arrays["weight_seed"].item()), TorchFactory(), dtype,
        ))
        with torch.inference_mode():
            if name.startswith("vq_"):
                arrays["x"] = normalized((4, 4, 4))
                arrays["encoded"] = network.encode(from_numpy(arrays["x"]).to(dtype)).numpy()
                latent = from_numpy(arrays["z"]).to(dtype)
                arrays["quantized"] = network.quantize(latent)[0].numpy()
                arrays["decoded"] = network.decode(latent).numpy()
                arrays["decoded_unquantized"] = network.decode(
                    latent, force_not_quantize=True,
                ).numpy()
            else:
                orientation = crop((3, 4, 5))
                arrays["orientation"] = orientation
                with tempfile.TemporaryDirectory(prefix="lesion-vq-orientation-") as temporary:
                    path = Path(temporary) / "ct.nii.gz"
                    nib.save(nib.Nifti1Image(orientation, np.eye(4)), path)
                    arrays["source"] = np.asarray(
                        source.load_nifty_volume_as_array(path)["data"], dtype=np.float32,
                    )
                arrays.pop("ras", None)  # Superseded by the tested source-to-RAS round trip.
                for prefix, shape in (("odd", (5, 3, 3)), ("long", (97, 2, 2))):
                    image = normalized(shape)
                    arrays[f"{prefix}_input"] = image
                    arrays[f"{prefix}_output"] = network.encode(from_numpy(image).half()).numpy()
                arrays["ct_output"] = source.img_multi_thresh_normalized(
                    arrays["ct"], [-1000, -200, 200, 1000], [0, 0.2, 0.8, 1],
                )
                arrays["inverse_output"] = source.img_multi_thresh_normalized(
                    arrays["normalized"], [0, 0.2, 0.8, 1], [-1000, -200, 200, 1000],
                )
        arrays["image_source"] = np.asarray("HCC-TACE-Seg HCC_063 contrast CT")
        arrays["torch_version"] = np.asarray(torch.__version__)
        arrays["numpy_version"] = np.asarray(np.__version__)
        np.savez_compressed(args.output / f"{name}.npz", allow_pickle=False, **arrays)
        print(f"Recorded {name}")


if __name__ == "__main__":
    main()
