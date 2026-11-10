"""Record DPM-Solver++ 2M trajectories from Diffusers 0.42.1 on Torch CPU.

Run in a reference environment with Diffusers 0.42.1 and the project's Torch,
NumPy and safetensors dependencies. ``--weights`` supplies the converted native
bundle; only the two stored alpha schedules are read. The LINet schedule uses
the runtime's FP16 cast before promotion to FP32. The recorder uses
independent seeded noise predictions, including FP16 predictions for LINet
promoted to FP32 at the solver boundary,
and projects known regions before every step and after the final update.

The compact NPZ contains inputs, predictions, normalized sigma ratios and every
reference state for 4, 8 and 25 evaluations. Its adjacent JSON records the pinned
source and numerical contract. Diffusers is needed only to regenerate fixtures.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch
from safetensors import safe_open

from lesiondiffusion_mlx._arrays import HostArray

REFERENCE_VERSION = "0.42.1"
REFERENCE_COMMIT = "2a1433913c7044f34fa4767dd84b28bba0bb0ecf"
REFERENCE_SOURCE = (
    "https://github.com/huggingface/diffusers/blob/"
    + REFERENCE_COMMIT
    + "/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py"
)


class _Reader(Protocol):
    def get_tensor(self, name: str) -> HostArray:
        """Read one named NumPy tensor from the safetensors bundle."""
        ...


class _ArchiveWriter(Protocol):
    def __call__(self, file: Path, **arrays: HostArray) -> None:
        """Write the explicitly numeric array mapping to a NumPy archive."""
        ...


class _Scheduler(Protocol):
    alphas_cumprod: torch.Tensor
    timesteps: torch.Tensor
    sigmas: torch.Tensor

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Prepare the reference's rounded linspace schedule and sigma ratios."""
        ...

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        return_dict: bool,
    ) -> tuple[torch.Tensor]:
        """Apply the reference's next first-order or multistep update."""
        ...


class _SchedulerFactory(Protocol):
    def __call__(
        self,
        *,
        num_train_timesteps: int,
        algorithm_type: str,
        solver_order: int,
        solver_type: str,
        prediction_type: str,
        thresholding: bool,
        lower_order_final: bool,
        final_sigmas_type: str,
        timestep_spacing: str,
    ) -> _Scheduler:
        """Construct the pinned reference with explicit algorithm settings."""
        ...


def record(weights: Path, destination: Path) -> None:
    """Record six independent scheduler trajectories without executing a learned model."""
    version = importlib.metadata.version("diffusers")
    if version != REFERENCE_VERSION:
        raise ValueError(f"Expected diffusers {REFERENCE_VERSION}; found {version}")
    module = importlib.import_module("diffusers")
    factory = cast(_SchedulerFactory, module.DPMSolverMultistepScheduler)
    from_numpy = cast("Callable[[HostArray], torch.Tensor]", vars(torch)["from_numpy"])
    schedules: dict[str, HostArray] = {}
    with safe_open(str(weights / "model.safetensors"), framework="np") as handle:
        reader = cast(_Reader, handle)
        for stage in ("lmnet", "linet"):
            schedules[stage] = reader.get_tensor(stage + ".alphas_cumprod").astype(
                np.float16 if stage == "linet" else np.float32
            )
    recorded: dict[str, HostArray] = {}
    cases: list[dict[str, object]] = []
    for stage, stored_alphas in schedules.items():
        for steps in (4, 8, 25):
            name = f"{stage}_{steps}"
            seed = 701 + steps + (100 if stage == "linet" else 0)
            generator = np.random.default_rng(seed)
            scheduler = factory(
                num_train_timesteps=stored_alphas.size,
                algorithm_type="dpmsolver++",
                solver_order=2,
                solver_type="midpoint",
                prediction_type="epsilon",
                thresholding=False,
                lower_order_final=True,
                final_sigmas_type="zero",
                timestep_spacing="linspace",
            )
            # Preserve the stored curve exactly; reconstructing betas and taking
            # another cumulative product would introduce unrelated FP32 rounding.
            scheduler.alphas_cumprod = from_numpy(stored_alphas.astype(np.float32))
            scheduler.set_timesteps(steps)
            shape = (1, 4 if stage == "linet" else 1, 2, 3, 4)
            initial = generator.normal(size=shape).astype(np.float32)
            condition = generator.normal(size=shape).astype(np.float16).astype(np.float32)
            mask = (np.arange(24).reshape(1, 1, 2, 3, 4) % 3 != 0).astype(np.float32)
            prediction_dtype = np.float16 if stage == "linet" else np.float32
            predictions = (0.2 * generator.normal(size=(steps, *shape))).astype(prediction_dtype)
            x = from_numpy(initial.copy())
            known = from_numpy(condition)
            native_mask = from_numpy(mask)
            states = [initial.copy()]
            projected: list[HostArray] = []
            with torch.no_grad():
                for index, timestep in enumerate(scheduler.timesteps):
                    if stage == "linet":
                        x = x * native_mask + (1 - native_mask) * known
                    projected.append(x.numpy().copy())
                    x = scheduler.step(
                        from_numpy(predictions[index].astype(np.float32)),
                        timestep,
                        x,
                        return_dict=False,
                    )[0]
                    states.append(x.numpy().copy())
                if stage == "linet":
                    x = x * native_mask + (1 - native_mask) * known
            output = x.numpy().copy()
            if not np.isfinite(output).all():
                raise ValueError(f"The official reference produced non-finite output for {name}")
            recorded.update(
                {
                    name + ".alphas_cumprod": stored_alphas,
                    name + ".initial_noise": initial,
                    name + ".conditioning_image": condition,
                    name + ".predictions": predictions,
                    name + ".timesteps": scheduler.timesteps.numpy().copy(),
                    name + ".sigma_ratios": scheduler.sigmas.numpy().copy(),
                    name + ".states": np.stack(states),
                    name + ".projected_inputs": np.stack(projected),
                    name + ".output": output,
                }
            )
            if stage == "linet":
                recorded[name + ".mask"] = mask
            cases.append(
                {"name": name, "stage": stage, "steps": steps, "seed": seed, "shape": shape}
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_archive = cast(_ArchiveWriter, vars(np)["savez_compressed"])
    save_archive(destination, **recorded)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "reference": {
            "distribution": "diffusers",
            "version": version,
            "commit": REFERENCE_COMMIT,
            "source": REFERENCE_SOURCE,
        },
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "recording_platform": platform.system().lower() + "-" + platform.machine(),
        "backend": "torch-cpu",
        "algorithm": "dpmsolver++",
        "solver_order": 2,
        "solver_type": "midpoint",
        "prediction_type": "epsilon",
        "thresholding": False,
        "lower_order_final": True,
        "final_sigmas_type": "zero",
        "timestep_spacing": "linspace",
        "state_dtype": "float32",
        "prediction_boundary": "stored predictions promoted to float32 before scheduler.step",
        "linet_schedule": (
            "checkpoint cumulative alphas cast to runtime float16, then promoted to float32"
        ),
        "known_region": "clean conditioning latent before every solver step and after final update",
        "states": (
            "initial noise followed by each unprojected step result; "
            "output includes final projection"
        ),
        "cases": cases,
    }
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "cases": len(cases), "arrays": len(recorded)}))


def main() -> None:
    """Record an independent scheduler fixture from the supplied native checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record(args.weights, args.output)


if __name__ == "__main__":
    main()
