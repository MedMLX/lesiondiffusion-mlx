"""Compare one LMNet precision with the captured 200-step FP32 trajectory.

The input conditioning, MLX seed, stochastic DDIM schedule and FP32 coefficient
buffers are fixed. Only learned-network arithmetic changes. Each repetition
restarts the same random stream. This measures the first mask-generation attempt;
its QC retry decision is reported instead of replacing it with a different draw.
Run each dtype in a separate process under the shared GPU lock.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from components import save_arrays
from pipeline import source_patch
from reduced_steps import compare_masks

from lesiondiffusion_mlx._arrays import FloatArray
from lesiondiffusion_mlx.medmlx import load_stage, select_lesion_component
from lesiondiffusion_mlx.mlx_model import LMNET_CONFIG, LdmUnet
from lesiondiffusion_mlx.mlx_sampling import DdimSampler

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Mlx


class Arguments(argparse.Namespace):
    weights: Path
    fixture: Path
    reference: Path
    output: Path
    dtype: str
    seed: int
    repetitions: int


def main() -> None:
    """Measure a complete seeded LMNet trajectory and compare selected lesions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path, help="Captured lmnet.safetensors")
    parser.add_argument(
        "--reference", required=True, type=Path, help="Captured lmnet-sample.safetensors"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), required=True)
    parser.add_argument("--seed", default=23, type=int)
    parser.add_argument("--repetitions", default=1, type=int)
    args = parser.parse_args(namespace=Arguments())
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    args.output.mkdir(parents=True, exist_ok=False)

    import mlx.core as mx

    runtime = cast("Mlx", mx)
    fixture = runtime.load(str(args.fixture), format="safetensors")
    expected = runtime.load(str(args.reference), format="safetensors")["sample"]
    dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[args.dtype]
    original_weights = load_stage(args.weights, "lmnet", runtime, dtype=mx.float32)
    weights = {
        key: value if key == "alphas_cumprod" else value.astype(dtype)
        for key, value in original_weights.items()
    }
    del original_weights
    model = LdmUnet(
        weights,
        runtime,
        LMNET_CONFIG,
        source_arithmetic=args.dtype == "float32",
        fused_attention=True,
        native_kernels=True,
    )
    denoiser = runtime.compile(model.__call__)
    sampler = DdimSampler(weights["alphas_cumprod"], runtime, 200)
    condition, image = fixture["context"], fixture["conditioning_image"]
    runtime.eval(*weights.values(), *fixture.values(), expected)
    runtime.random.seed(args.seed)
    probe = runtime.random.normal((image.shape[0], 1, *image.shape[2:]))
    runtime.eval(probe)
    np.testing.assert_array_equal(
        np.array(probe, dtype=np.float32),
        np.array(fixture["network_input"][:, :1], dtype=np.float32),
    )
    del probe
    bbox = np.asarray(image[0, 2], dtype=np.float32) > 0
    expected_host = cast(FloatArray, np.array(expected))
    expected_mask = select_lesion_component(expected_host[0, 0], bbox)
    root = Path(__file__).resolve().parents[1]
    (args.output / "source.patch").write_text(source_patch(root))
    report: dict[str, object] = {
        "patch": "source.patch",
        "lockfile": "uv.lock",
        "command": sys.argv,
        "revision": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "dtype": args.dtype,
        "schedule_dtype": str(weights["alphas_cumprod"].dtype),
        "steps": len(sampler.timesteps),
        "eta": 1,
        "seed": args.seed,
        "initial_noise_matches_reference": True,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_extension": mx.__file__,
        "device": dict(mx.device_info()),
        "reference": str(args.reference.resolve()),
        "timings": [],
    }
    timings: list[dict[str, object]] = []
    report["timings"] = timings
    for iteration in range(args.repetitions):
        runtime.random.seed(args.seed)
        mx.synchronize()
        mx.reset_peak_memory()
        started, cpu_started = time.perf_counter(), time.process_time()
        sampled = sampler(denoiser, condition, image)
        runtime.eval(sampled)
        elapsed = time.perf_counter() - started
        timing: dict[str, object] = {
            "phase": "first_trajectory" if iteration == 0 else "warm_trajectory",
            "elapsed_s": elapsed,
            "ms_per_step": elapsed * 1000.0 / len(sampler.timesteps),
            "cpu_s": time.process_time() - cpu_started,
            "peak_mlx_bytes": mx.get_peak_memory(),
        }
        host = cast(FloatArray, np.array(sampled))
        if not np.isfinite(host).all():
            raise ValueError(f"LMNet {args.dtype} produced non-finite samples")
        mask = select_lesion_component(host[0, 0], bbox)
        timing["qc_requires_retry"] = int(mask.sum()) <= 400
        if np.any(mask):
            timing["mask"] = compare_masks(expected_mask, mask, np.eye(4), (1.0, 1.0, 1.0))
        else:
            timing["mask"] = {"lesion_voxels": 0, "mask_dice": 0.0}
        difference = host.astype(np.float64) - expected_host.astype(np.float64)
        timing["sample_max_abs_error"] = float(np.abs(difference).max())
        timing["sample_rmse"] = float(np.sqrt(np.mean(difference * difference)))
        timings.append(timing)
        save_arrays(args.output / f"sample-{iteration}.safetensors", {"sample": sampled})
        (args.output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
