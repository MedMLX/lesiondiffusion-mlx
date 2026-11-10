"""Capture a warmed denoiser forward using retained pipeline inputs.

Run with ``MTL_CAPTURE_ENABLED=1`` under the shared GPU lock. The input
safetensors file contains ``network_input``, ``timestep`` and ``context`` from
one real conditioned-mode call. Capture timings include instrumentation and
are not latency benchmarks; use the pipeline benchmark for speed comparisons.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Mlx


def main() -> None:
    """Warm the selected denoiser, then run evaluated forwards with optional capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lmnet", "linet"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--calls", type=int, default=1)
    parser.add_argument("--cached-context", action="store_true")
    args = parser.parse_args()
    if args.calls < 1:
        parser.error("--calls must be positive")
    if args.capture is not None and args.capture.exists():
        parser.error("--capture must be a new .gputrace path")
    if args.capture is not None and args.capture.suffix != ".gputrace":
        parser.error("--capture must end in .gputrace")

    # medmlx-core sets the numerical policy before importing the MLX extension.
    from medmlx_core.runtime import import_mlx

    from lesiondiffusion_mlx.medmlx import load_stage
    from lesiondiffusion_mlx.mlx_model import LINET_CONFIG, LMNET_CONFIG, LdmUnet

    mx = cast("Mlx", import_mlx())
    import mlx.core as native
    from mlx.core.metal import start_capture, stop_capture

    inputs = mx.load(str(args.inputs), format="safetensors")
    network_input, timestep, context = (
        inputs["network_input"],
        inputs["timestep"],
        inputs["context"],
    )
    is_lmnet = args.stage == "lmnet"
    model = LdmUnet(
        load_stage(args.weights, args.stage, mx, dtype=mx.float32 if is_lmnet else mx.float16),
        mx,
        LMNET_CONFIG if is_lmnet else replace(LINET_CONFIG, skip_context=False),
        source_arithmetic=is_lmnet,
        fused_attention=True,
        native_kernels=True,
    )
    denoiser: Callable[[native.array, native.array], native.array]
    if args.cached_context:
        denoiser = mx.compile(model.prepare_denoiser(context))
    else:
        compiled = mx.compile(model.__call__)

        def original_denoiser(x: native.array, timesteps: native.array) -> native.array:
            """Keep context as an input to the original compiled denoiser."""
            return compiled(x, timesteps, context)

        denoiser = original_denoiser

    mx.eval(network_input, timestep, context, *model.weights.values())
    for _ in range(2):
        mx.eval(denoiser(network_input, timestep))

    if args.capture is not None:
        args.capture.parent.mkdir(parents=True, exist_ok=True)
        start_capture(str(args.capture))
    started_ns = time.perf_counter_ns()
    try:
        output = denoiser(network_input, timestep)
        mx.eval(output)
        for _ in range(args.calls - 1):
            output = denoiser(network_input, timestep)
            mx.eval(output)
    finally:
        stopped_ns = time.perf_counter_ns()
        if args.capture is not None:
            stop_capture()
    if not bool(mx.all(mx.isfinite(output)).item()):
        raise RuntimeError("Captured denoiser produced non-finite output")
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    print(
        json.dumps(
            {
                "stage": args.stage,
                "revision": revision,
                "changed_files": subprocess.check_output(
                    ["git", "diff", "--name-only"], cwd=root, text=True
                ).splitlines(),
                "command": sys.argv,
                "python": sys.executable,
                "mlx_version": importlib.metadata.version("mlx"),
                "mlx_extension": native.__file__,
                "macos": platform.mac_ver()[0],
                "device": mx.device_info(),
                "inputs": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in inputs.items()
                },
                "nax_packed_weights": len(model.nax_weights),
                "warmup_calls": 2,
                "calls": args.calls,
                "started_ns": started_ns,
                "stopped_ns": stopped_ns,
                "cached_context": args.cached_context,
                "capture": str(args.capture) if args.capture is not None else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
