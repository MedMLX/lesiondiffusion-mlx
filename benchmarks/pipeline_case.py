"""Expose complete NIfTI inference to ``perfkit benchmark call``.

Use ``pipeline_case:case`` as the input factory and ``pipeline_case:run_case``
as the target. Required environment variables are ``LESION_BENCH_IMAGE``,
``LESION_BENCH_ORGAN_MASK`` and ``LESION_BENCH_WEIGHTS``. Optional suffixes
are ``LESION_MASK``, ``ANATOMY_MASK``, ``OUTPUT_REGION`` (lesion), ``MODE``
(conditioned), ``STEPS`` (200), ``LMNET_STEPS`` / ``LINET_STEPS`` (shared count),
``SAMPLER`` (ddim), ``DECODER_MODE`` (reference), ``SEED`` (23), and ``VARIANT`` (candidate).
The factory loads no model: cold and warm calls both include model loading,
compilation, sampling and NIfTI writing, as the public API currently does.

Every call writes into a new temporary directory, cleaned at process exit.
Set ``LESION_BENCH_EVIDENCE`` to retain a per-process workload description and
complete development-code patch outside the temporary inference outputs.
Per-call stage records are collected in memory and written once at process exit.
Perfkit owns timing, warmup, repeated calls and memory measurements.

For exact endpoint comparison, use ``parity_case`` as the factory and
``parity_reference`` / ``parity_candidate`` as the two parity targets. They
return every saved NIfTI array and affine plus the final MLX RNG state.
"""

from __future__ import annotations

import atexit
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from lesiondiffusion_mlx._arrays import HostArray
    from lesiondiffusion_mlx.medmlx import RunResult


def case() -> dict[str, object]:
    """Prepare one public-pipeline workload without performing inference."""
    return _case(os.environ.get("LESION_BENCH_VARIANT", "candidate"))


def _case(variant: str) -> dict[str, object]:
    """Bind one explicit implementation to the configured workload."""
    from baseline import baseline_execution
    from pipeline import runtime_diagnostics, source_patch

    from lesiondiffusion_mlx.medmlx import run

    image = Path(os.environ["LESION_BENCH_IMAGE"])
    organ_mask = Path(os.environ["LESION_BENCH_ORGAN_MASK"])
    weights = Path(os.environ["LESION_BENCH_WEIGHTS"])
    lesion_name = os.environ.get("LESION_BENCH_LESION_MASK")
    lesion = Path(lesion_name) if lesion_name else None
    anatomy_name = os.environ.get("LESION_BENCH_ANATOMY_MASK")
    anatomy = Path(anatomy_name) if anatomy_name else None
    region = os.environ.get("LESION_BENCH_OUTPUT_REGION", "lesion")
    mode = os.environ.get("LESION_BENCH_MODE", "conditioned")
    sampler = os.environ.get("LESION_BENCH_SAMPLER", "ddim")
    decoder_mode = os.environ.get("LESION_BENCH_DECODER_MODE", "reference")
    steps = int(os.environ.get("LESION_BENCH_STEPS", "200"))
    lmnet_steps_name = os.environ.get("LESION_BENCH_LMNET_STEPS")
    linet_steps_name = os.environ.get("LESION_BENCH_LINET_STEPS")
    lmnet_steps = int(lmnet_steps_name) if lmnet_steps_name is not None else None
    linet_steps = int(linet_steps_name) if linet_steps_name is not None else None
    seed = int(os.environ.get("LESION_BENCH_SEED", "23"))
    if variant not in {"baseline", "candidate"}:
        raise ValueError("LESION_BENCH_VARIANT must be baseline or candidate")
    if region not in {"full", "box", "lesion"} or mode not in {"conditioned", "source"}:
        raise ValueError("Choose a supported LESION_BENCH_OUTPUT_REGION and LESION_BENCH_MODE")
    if lesion is not None and anatomy is not None:
        raise ValueError("LESION_BENCH_ANATOMY_MASK applies only to generated-mask workflows")
    directory = TemporaryDirectory(prefix="lesiondiffusion-benchmark-")
    atexit.register(directory.cleanup)
    output_root = Path(directory.name)
    call_index = 0
    calls: list[dict[str, object]] = []
    root = Path(__file__).resolve().parents[1]
    evidence_name = os.environ.get("LESION_BENCH_EVIDENCE")
    if evidence_name:
        evidence = Path(evidence_name) / f"{variant}-{os.getpid()}"
        evidence.mkdir(parents=True, exist_ok=False)
        (evidence / "source.patch").write_text(source_patch(root))
        metadata: dict[str, object] = {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "patch": "source.patch",
            "python": sys.version,
            "executable": sys.executable,
            "mlx_version": importlib.metadata.version("mlx"),
            "macos": platform.mac_ver()[0],
            "runtime": runtime_diagnostics(mode, variant, decoder_mode),
            "image": str(image.resolve()),
            "organ_mask": str(organ_mask.resolve()),
            "lesion_mask": str(lesion.resolve()) if lesion else None,
            "anatomy_mask": str(anatomy.resolve()) if anatomy else None,
            "weights": str(weights.resolve()),
            "output_region": region,
            "inference_mode": mode,
            "sampler": sampler,
            "decoder_mode": decoder_mode,
            "execution": variant,
            "steps": steps,
            "stage_steps_requested": {
                "lmnet": steps if lmnet_steps is None else lmnet_steps,
                "linet": steps if linet_steps is None else linet_steps,
            },
            "seed": seed,
            "timing_boundary": (
                "public run() through mx.synchronize(); imports and factory excluded"
            ),
        }
        (evidence / "workload.json").write_text(json.dumps(metadata, indent=2) + "\n")

        def save_calls() -> None:
            """Write retained stage records after perfkit has finished timing calls."""
            (evidence / "calls.json").write_text(json.dumps(calls, indent=2) + "\n")

        atexit.register(save_calls)

    def pipeline() -> RunResult:
        """Run and synchronize one independent full pipeline including NIfTI writes."""
        nonlocal call_index
        import mlx.core as mx

        output = output_root / f"run-{call_index}"
        call_index += 1
        execution = baseline_execution() if variant == "baseline" else nullcontext()
        with execution:
            result = run(
                image_path=image,
                organ_mask_path=organ_mask,
                lesion_mask_path=lesion,
                anatomy_mask_path=anatomy,
                weights_path=weights,
                output_dir=output,
                organ="liver",
                lesion_type="tumor",
                bbox_size_mm=(30, 30, 20) if lesion is None else None,
                bbox_center_voxel=(80, 80, 64) if lesion is None else None,
                placement="inside_parenchyma",
                diffusion_steps=steps,
                lmnet_diffusion_steps=lmnet_steps,
                linet_diffusion_steps=linet_steps,
                sampler=sampler,
                decoder_mode=decoder_mode,
                seed=seed,
                inference_mode=mode,
                output_region=region,
            )
        mx.synchronize()
        if evidence_name:
            details = result["details"]
            generation = details["generation_spec"]
            native = generation["native_runtime"]
            calls.append(
                {
                    "index": call_index - 1,
                    "reported_elapsed_s": details["elapsed_s"],
                    "diffusion_steps_requested": details["diffusion_steps"],
                    "sampler": details["sampler"],
                    "decoder_mode": details["decoder_mode"],
                    "stage_steps_requested": {
                        "lmnet": steps if lmnet_steps is None else lmnet_steps,
                        "linet": steps if linet_steps is None else linet_steps,
                    },
                    "diffusion_steps_actual": {
                        stage: record.get("diffusion_steps_actual")
                        for stage, record in details["stages"].items()
                    },
                    "lesion_mask_source": native["lesion_mask_source"],
                    "lmnet_attempts": native["lmnet_attempts"],
                    "output_validation": generation["output_validation"],
                    "stages": details["stages"],
                }
            )
        return result

    mask_source = "supplied" if lesion is not None else "generated"
    lm_count = steps if lmnet_steps is None else lmnet_steps
    lin_count = steps if linet_steps is None else linet_steps
    return {
        "name": (
            f"pipeline-{variant}-{sampler}-{decoder_mode}-{mode}-{mask_source}-{region}"
            f"-lm{lm_count}-lin{lin_count}-seed{seed}"
        ),
        "args": (pipeline,),
    }


def run_case(pipeline: Callable[[], RunResult]) -> RunResult:
    """Return the completed pipeline result for perfkit's cold and warm calls."""
    return pipeline()


def parity_case() -> dict[str, object]:
    """Select the environment-configured real workload for independent parity workers."""
    region = os.environ.get("LESION_BENCH_OUTPUT_REGION", "lesion")
    mode = os.environ.get("LESION_BENCH_MODE", "conditioned")
    sampler = os.environ.get("LESION_BENCH_SAMPLER", "ddim")
    decoder_mode = os.environ.get("LESION_BENCH_DECODER_MODE", "reference")
    steps = os.environ.get("LESION_BENCH_STEPS", "200")
    lmnet_steps = os.environ.get("LESION_BENCH_LMNET_STEPS", steps)
    linet_steps = os.environ.get("LESION_BENCH_LINET_STEPS", steps)
    seed = os.environ.get("LESION_BENCH_SEED", "23")
    mask_source = "supplied" if os.environ.get("LESION_BENCH_LESION_MASK") else "generated"
    return {
        "name": (
            f"pipeline-{sampler}-{decoder_mode}-{mode}-{mask_source}-{region}"
            f"-lm{lmnet_steps}-lin{linet_steps}-seed{seed}"
        ),
        "args": (),
    }


def _parity_outputs(variant: str) -> dict[str, HostArray]:
    """Materialize saved output values before the owned temporary directory is cleaned."""
    import mlx.core as mx
    import numpy as np

    from lesiondiffusion_mlx._nifti import nib

    prepared = _case(variant)
    pipeline = cast("tuple[Callable[[], RunResult]]", prepared["args"])[0]
    result = pipeline()
    outputs: dict[str, HostArray] = {}
    for role, path in result["outputs"].items():
        image = nib.load(path)
        outputs[role] = np.asarray(image.dataobj)
        outputs[role + "_affine"] = image.affine
    for index, state in enumerate(cast(list[mx.array], mx.random.state)):
        outputs[f"rng_{index}"] = np.array(state)
    return outputs


def parity_reference() -> dict[str, HostArray]:
    """Run the retained original execution and return saved arrays, affines and RNG."""
    return _parity_outputs("baseline")


def parity_candidate() -> dict[str, HostArray]:
    """Run current execution and return saved arrays, affines and RNG."""
    return _parity_outputs("candidate")
