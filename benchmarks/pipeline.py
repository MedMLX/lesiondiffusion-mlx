"""Measure the public NIfTI pipeline with unchanged inference options.

Run this entrypoint under a shared GPU lock (for example ``perfkit gpu lock
run``). A run includes weight loading, preparation, both diffusion stages,
decoding, and output writing. The first call is recorded separately from
subsequent same-process calls; every call still reloads models as ``run`` does.
Supply ``--lesion-mask`` to measure LINet without LMNet mask generation.
Use an external command benchmark to measure interpreter startup as well.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Mlx


def source_patch(root: Path) -> str:
    """Capture tracked edits and untracked development code relative to HEAD."""
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root, text=True)
    untracked = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "src",
            "benchmarks",
            "tests",
        ],
        cwd=root,
        text=True,
    )
    for name in untracked.split("\0"):
        if not name:
            continue
        difference = subprocess.run(
            ["git", "diff", "--binary", "--no-index", "--", "/dev/null", name],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if difference.returncode not in (0, 1):
            difference.check_returncode()
        patch += difference.stdout
    return patch


def runtime_diagnostics(
    inference_mode: str, execution: str = "candidate", decoder_mode: str = "reference"
) -> dict[str, object]:
    """Report configured native paths; dispatch tracing verifies their execution."""
    import mlx.core as mx

    from lesiondiffusion_mlx._nax_conv import DISABLE_ENV, nax_available

    available = nax_available(cast("Mlx", mx))
    conditioned = inference_mode == "conditioned"
    native_half = conditioned and available and execution == "candidate"
    return {
        "device": dict(mx.device_info()),
        "mlx_extension": mx.__file__,
        "medmlx_core_version": importlib.metadata.version("medmlx-core"),
        "execution": execution,
        "decoder_mode": decoder_mode,
        "nax_available": available,
        "nax_convolution_enabled": conditioned and available,
        "nax_attention_enabled": conditioned and available,
        "environment": {
            name: os.environ.get(name)
            for name in (
                DISABLE_ENV,
                "MLX_DISABLE_COMPILE",
                "MLX_ENABLE_TF32",
                "MTL_CAPTURE_ENABLED",
            )
        },
        "precision": {
            "lmnet": "float32",
            "linet": "float16",
            "vq_encoder": "float16",
            "vq_decoder": ("float32" if conditioned and decoder_mode == "reference" else "float16"),
        },
        "attention": {
            "lmnet": "flash_fp32_self_mlx_fused_padded_cross" if conditioned else "source_explicit",
            "linet": (
                "flash_fp16_self_mlx_fused_padded_cross"
                if native_half
                else "mlx_fused_padded_fp16"
                if conditioned
                else "source_explicit"
            ),
        },
    }


class Arguments(argparse.Namespace):
    """Command-line workload and output locations."""

    image: Path
    organ_mask: Path
    anatomy_mask: Path | None
    lesion_mask: Path | None
    weights: Path
    output: Path
    repetitions: int
    steps: int
    lmnet_steps: int | None
    linet_steps: int | None
    sampler: str
    decoder_mode: str
    bbox_size_mm: list[float] | None
    seed: int
    output_region: str
    inference_mode: str
    center: list[int] | None
    capture_components: bool
    execution: str


def main() -> None:
    """Run complete inference and save synchronized samples beside its outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--organ-mask", required=True, type=Path)
    parser.add_argument("--anatomy-mask", type=Path)
    parser.add_argument("--lesion-mask", type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", default=1, type=int)
    parser.add_argument("--steps", default=200, type=int)
    parser.add_argument("--lmnet-steps", type=int, help="Override LMNet's shared --steps count")
    parser.add_argument("--linet-steps", type=int, help="Override LINet's shared --steps count")
    parser.add_argument("--sampler", choices=("ddim", "dpmpp_2m"), default="ddim")
    parser.add_argument("--decoder-mode", choices=("reference", "scaled_fp16"), default="reference")
    parser.add_argument("--bbox-size-mm", nargs=3, type=float, help="Default: 30 30 20")
    parser.add_argument("--seed", default=23, type=int)
    parser.add_argument("--output-region", choices=("full", "box", "lesion"), default="lesion")
    parser.add_argument(
        "--inference-mode", choices=("conditioned", "source"), default="conditioned"
    )
    parser.add_argument("--center", nargs=3, type=int)
    parser.add_argument("--capture-components", action="store_true")
    parser.add_argument("--execution", choices=("baseline", "candidate"), default="candidate")
    args = parser.parse_args(namespace=Arguments())
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.lesion_mask is not None and args.center is not None:
        parser.error("--lesion-mask defines the box; omit --center")
    if args.lesion_mask is not None and args.bbox_size_mm is not None:
        parser.error("--lesion-mask defines the box; omit --bbox-size-mm")
    args.output.mkdir(parents=True, exist_ok=False)

    import mlx.core as mx
    from baseline import baseline_execution
    from components import capture_components, save_arrays

    from lesiondiffusion_mlx.medmlx import run

    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    (args.output / "source.patch").write_text(source_patch(root))
    report: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "revision": revision,
        "patch": "source.patch",
        "lockfile": "uv.lock",
        "command": sys.argv,
        "python": sys.version,
        "executable": sys.executable,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_extension": mx.__file__,
        "macos": platform.mac_ver()[0],
        "device": dict(mx.device_info()),
        "runtime": runtime_diagnostics(args.inference_mode, args.execution, args.decoder_mode),
        "timing_boundary": "public run() through mx.synchronize(); imports excluded",
        "steps_requested": args.steps,
        "sampler": args.sampler,
        "decoder_mode": args.decoder_mode,
        "bbox_size_mm": (args.bbox_size_mm or [30, 30, 20]) if args.lesion_mask is None else None,
        "bbox_center_voxel": (args.center or [80, 80, 64]) if args.lesion_mask is None else None,
        "stage_steps_requested": {
            "lmnet": args.steps if args.lmnet_steps is None else args.lmnet_steps,
            "linet": args.steps if args.linet_steps is None else args.linet_steps,
        },
        "seed": args.seed,
        "image": str(args.image.resolve()),
        "organ_mask": str(args.organ_mask.resolve()),
        "lesion_mask": str(args.lesion_mask.resolve()) if args.lesion_mask is not None else None,
        "weights": str(args.weights.resolve()),
        "inference_mode": args.inference_mode,
        "output_region": args.output_region,
        "capture_components": args.capture_components,
        "execution": args.execution,
        "samples": [],
    }
    samples: list[dict[str, object]] = []
    report["samples"] = samples
    report_path = args.output / "measurements.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"status": "started", "revision": revision, "output": str(args.output)}),
        flush=True,
    )
    for index in range(args.repetitions):
        mx.synchronize()
        mx.reset_peak_memory()
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        capture = (
            capture_components(args.output / "components")
            if args.capture_components and index == 0
            else nullcontext(dict[str, float]())
        )
        execution = baseline_execution() if args.execution == "baseline" else nullcontext()
        with execution, capture as export_times:
            result = run(
                image_path=args.image,
                organ_mask_path=args.organ_mask,
                anatomy_mask_path=args.anatomy_mask,
                lesion_mask_path=args.lesion_mask,
                weights_path=args.weights,
                output_dir=args.output / f"run-{index}",
                organ="liver",
                lesion_type="tumor",
                bbox_size_mm=cast(
                    tuple[float, float, float], tuple(args.bbox_size_mm or [30, 30, 20])
                )
                if args.lesion_mask is None
                else None,
                bbox_center_voxel=(args.center or [80, 80, 64])
                if args.lesion_mask is None
                else None,
                placement="inside_parenchyma",
                diffusion_steps=args.steps,
                lmnet_diffusion_steps=args.lmnet_steps,
                linet_diffusion_steps=args.linet_steps,
                sampler=args.sampler,
                decoder_mode=args.decoder_mode,
                seed=args.seed,
                inference_mode=args.inference_mode,
                output_region=args.output_region,
            )
        mx.synchronize()
        sample: dict[str, object] = {
            "index": index,
            "phase": "first_call" if index == 0 else "subsequent_call",
            "wall_s": time.perf_counter() - wall_start,
            "cpu_s": time.process_time() - cpu_start,
            "peak_mlx_bytes": mx.get_peak_memory(),
            "active_mlx_bytes": mx.get_active_memory(),
            "cache_mlx_bytes": mx.get_cache_memory(),
            "capture_export_s": export_times,
            "stage_steps_actual": {
                stage: record.get("diffusion_steps_actual")
                for stage, record in result["details"]["stages"].items()
            },
            "result": result,
        }
        save_arrays(
            args.output / f"rng-after-{index}.safetensors",
            {str(i): state for i, state in enumerate(cast(list[mx.array], mx.random.state))},
        )
        samples.append(sample)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({key: value for key, value in sample.items() if key != "result"}), flush=True
        )


if __name__ == "__main__":
    main()
