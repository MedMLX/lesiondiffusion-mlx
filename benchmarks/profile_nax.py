"""Compare NAX row tiles on one fixed, retained conditioned UNet input.

This experimental entrypoint changes only module-local launch geometry before
tracing each callable. The split-K threshold scales inversely with tile rows to
preserve the original reduction strategy on the power-of-two LINet grids.
Run under the shared GPU lock, separately from capture/profiling instrumentation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    import mlx.core as mx

    from lesiondiffusion_mlx._typing import Mlx


def main() -> None:
    """Time alternating warmed row-tile variants and compare their full outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("lmnet", "linet"), default="linet")
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")

    from medmlx_core.runtime import import_mlx

    from lesiondiffusion_mlx import _nax_conv
    from lesiondiffusion_mlx.medmlx import load_stage
    from lesiondiffusion_mlx.mlx_model import LINET_CONFIG, LMNET_CONFIG, LdmUnet

    runtime = cast("Mlx", import_mlx())
    import mlx.core as mx

    inputs = runtime.load(str(args.inputs), format="safetensors")
    x, timestep, context = inputs["network_input"], inputs["timestep"], inputs["context"]
    is_lmnet = args.stage == "lmnet"
    model = LdmUnet(
        load_stage(args.weights, args.stage, runtime, dtype=mx.float32 if is_lmnet else mx.float16),
        runtime,
        LMNET_CONFIG if is_lmnet else replace(LINET_CONFIG, skip_context=False),
        source_arithmetic=is_lmnet,
        fused_attention=True,
        native_kernels=True,
    )
    runtime.eval(x, timestep, context, *model.weights.values())
    settings = cast(dict[str, object], vars(_nax_conv))
    saved_rows = cast(int, settings["_SIMDGROUP_ROWS"])
    saved_threshold = cast(int, settings["_SPLIT_BELOW_THREADGROUPS"])
    variants: dict[int, Callable[[mx.array, mx.array, mx.array], mx.array]] = {}
    retained_functions: list[Callable[[mx.array, mx.array, mx.array], mx.array]] = []
    outputs: dict[int, mx.array] = {}
    first_call: dict[int, float] = {}
    samples: dict[int, list[float]] = {rows: [] for rows in (8, 4, 2)}
    peaks: dict[int, list[int]] = {rows: [] for rows in samples}
    try:
        for rows in samples:
            settings["_SIMDGROUP_ROWS"] = rows
            settings["_SPLIT_BELOW_THREADGROUPS"] = saved_threshold * saved_rows // rows

            def forward(a: mx.array, b: mx.array, c: mx.array) -> mx.array:
                """Trace a distinct graph with the selected NAX launch geometry."""
                return model(a, b, c)

            retained_functions.append(forward)
            compiled = mx.compile(forward)
            variants[rows] = compiled
            start = time.perf_counter()
            output = compiled(x, timestep, context)
            runtime.eval(output)
            first_call[rows] = time.perf_counter() - start
            runtime.eval(compiled(x, timestep, context))
            outputs[rows] = output

        reference = np.array(outputs[8], dtype=np.float64)
        parity: dict[int, dict[str, object]] = {}
        for rows, output in outputs.items():
            actual = np.array(output, dtype=np.float64)
            delta = actual - reference
            parity[rows] = {
                "finite": bool(np.all(np.isfinite(actual))),
                "exact": bool(np.array_equal(actual, reference)),
                "max_abs": float(np.max(np.abs(delta))),
                "mean_abs": float(np.mean(np.abs(delta))),
                "different_values": int(np.count_nonzero(delta)),
            }

        for repetition in range(args.repeats):
            order = (8, 4, 2) if repetition % 2 == 0 else (2, 4, 8)
            for rows in order:
                mx.synchronize()
                mx.reset_peak_memory()
                start = time.perf_counter()
                output = variants[rows](x, timestep, context)
                runtime.eval(output)
                samples[rows].append(time.perf_counter() - start)
                peaks[rows].append(mx.get_peak_memory())
    finally:
        settings["_SIMDGROUP_ROWS"] = saved_rows
        settings["_SPLIT_BELOW_THREADGROUPS"] = saved_threshold

    report: dict[str, object] = {
        "stage": args.stage,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_extension": mx.__file__,
        "device": mx.device_info(),
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "context_shape": list(context.shape),
        "warmups": 2,
        "timing": "compiled denoiser invocation through mx.eval(output), after mx.synchronize",
        "split_threshold_scaling": "96 * 8 / rows; preserve LINet split-K selection",
        "variants": {
            str(rows): {
                "samples_s": timings,
                "median_s": statistics.median(timings),
                "min_s": min(timings),
                "max_s": max(timings),
                "first_call_s": first_call[rows],
                "peak_mlx_bytes": peaks[rows],
                "parity": parity[rows],
            }
            for rows, timings in samples.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
