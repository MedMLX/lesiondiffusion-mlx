"""Train a radiomic context adapter against the frozen source LINet on CUDA.

This research entrypoint consumes Mac-exported real VQ latents and text contexts.
It uses the maintained source UNet and the released EMA parameters, without loading
patient pixels into the adapter. The source inpainting objective is masked epsilon
L1, normalized by the lesion bounding box, plus its auxiliary mask Dice loss.
Only the four MLP parameter tensors are optimized. Full 64-cubed latents are used.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import safetensors.torch as safetensors_torch
import torch
import torch.nn.functional as functional
import torch.utils.checkpoint as checkpoint_module
from ldm.modules import attention
from ldm.modules.diffusionmodules import openaimodel
from numpy.typing import NDArray
from torch import nn


class _TorchArrayBoundary(Protocol):
    def from_numpy(self, array: NDArray[np.float32]) -> torch.Tensor: ...
    def manual_seed(self, seed: int) -> torch.Generator: ...


class _CheckpointBoundary(Protocol):
    def checkpoint(
        self,
        function: Callable[..., torch.Tensor],
        *inputs: torch.Tensor | None,
        use_reentrant: bool,
    ) -> torch.Tensor: ...


class _SafetensorsBoundary(Protocol):
    def save_file(self, tensors: dict[str, torch.Tensor], filename: str | Path) -> None: ...


class _BackwardBoundary(Protocol):
    def backward(self) -> None: ...


class _OptimizerStepBoundary(Protocol):
    def step(self) -> None: ...


_torch_arrays = cast(_TorchArrayBoundary, torch)
_checkpoint = cast(_CheckpointBoundary, checkpoint_module).checkpoint
_save_file = cast(_SafetensorsBoundary, safetensors_torch).save_file


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    raw = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in raw):
        raise ValueError(f"{label} keys must be strings")
    return cast(dict[str, object], raw)


def _tensor(value: object, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    return value


def _array(value: object, length: int, label: str) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{label} requires {length} finite values")
    return array


def _strings(value: object, length: int, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    rows = cast(list[object], value)
    if len(rows) != length:
        raise ValueError(f"{label} requires {length} entries")
    return tuple(_text(row, label) for row in rows)


@dataclass(frozen=True)
class CachedCase:
    """One explicitly partitioned real lesion and its exported tensor cache."""

    patient_id: str
    lesion_id: str
    split: str
    file: Path


@dataclass(frozen=True)
class Bundle:
    """Frozen training-only normalization and explicit patient membership."""

    path: Path
    feature_names: tuple[str, ...]
    center: NDArray[np.float32]
    scale: NDArray[np.float32]
    context_fields: tuple[str, ...]
    target_space: str
    dataset_complete: bool
    cases: tuple[CachedCase, ...]
    alphas_cumprod: NDArray[np.float32]


def read_bundle(index: Path) -> Bundle:
    payload = _object(json.loads(index.read_text()), "training index")
    names = _strings(payload["feature_names"], 101, "feature_names")
    center = _array(payload["feature_center"], 101, "feature_center")
    scale = _array(payload["feature_scale"], 101, "feature_scale")
    fields = _strings(payload["context_fields"], 10, "context_fields")
    target_space = _text(payload["target_space"], "target_space")
    complete = payload["dataset_complete"]
    if not isinstance(complete, bool):
        raise ValueError("dataset_complete must be a boolean")
    if target_space not in {"absolute_hu", "liver_contrast"}:
        raise ValueError("Unsupported target_space")
    if np.any(scale <= 0):
        raise ValueError("feature_scale must be positive")
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list):
        raise ValueError("cases must be a list")
    cases: list[CachedCase] = []
    for raw in cast(list[object], raw_cases):
        row = _object(raw, "case")
        split = _text(row["split"], "split")
        if split not in {"train", "selection"}:
            raise ValueError("Only train/selection cases may enter this bundle")
        cases.append(
            CachedCase(
                _text(row["patient_id"], "patient_id"),
                _text(row["lesion_id"], "lesion_id"),
                split,
                index.parent / _text(row["file"], "file"),
            )
        )
    train = {case.patient_id for case in cases if case.split == "train"}
    selection = {case.patient_id for case in cases if case.split == "selection"}
    if not train or train & selection:
        raise ValueError("Training patients must be present and disjoint from selection")
    schedule = np.asarray(
        np.load(index.parent / _text(payload["schedule_file"], "schedule_file")),
        dtype=np.float32,
    )
    if schedule.shape != (1000,) or not np.isfinite(schedule).all():
        raise ValueError("Expected the released 1000-step alpha schedule")
    if np.any(schedule <= 0) or np.any(schedule > 1) or np.any(np.diff(schedule) >= 0):
        raise ValueError("Alpha schedule must decrease within (0, 1]")
    return Bundle(
        index, names, center, scale, fields, target_space, complete, tuple(cases), schedule
    )


class RadiomicAdapter(nn.Module):
    """Map train-normalized 101 features to ten distinct 512-dimensional residuals."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(101, 64)
        self.output = nn.Linear(64, 10 * 512)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = self.output(functional.silu(self.hidden(features)))
        return context + residual.reshape(features.shape[0], 10, 512)


def training_checkpoint(
    function: Callable[..., torch.Tensor],
    inputs: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.Tensor],
    enabled: bool,
) -> torch.Tensor:
    """Replace the CUDA inference helper with PyTorch's maintained gradient path.

    The deployed CUDA checkout deliberately implements its checkpoint boundary as
    no-grad inference. This isolated training process restores ordinary activation
    checkpointing without altering that checkout or implementing a custom VJP.
    """
    del parameters
    if enabled and torch.is_grad_enabled():
        return _checkpoint(function, *inputs, use_reentrant=False)
    return function(*inputs)


def load_frozen_unet(checkpoint: Path, device: torch.device) -> tuple[nn.Module, torch.Tensor]:
    """Use the installed source architecture and released EMA parameter naming."""
    attention.checkpoint = training_checkpoint
    openaimodel.checkpoint = training_checkpoint
    with torch.device("meta"):
        model = openaimodel.UNetModel(
            image_size=64,
            in_channels=5,
            out_channels=5,
            model_channels=128,
            attention_resolutions=(4, 8),
            num_res_blocks=2,
            channel_mult=(1, 2, 4),
            num_head_channels=32,
            use_checkpoint=True,
            use_fp16=False,
            use_spatial_transformer=True,
            skip_context=False,
            context_dim=512,
            dims=3,
        )
    raw: object = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = _object(_object(raw, "LINet checkpoint")["state_dict"], "state_dict")
    ema = {
        name: _tensor(state[f"model_ema.{name.replace('.', '')}"], name).float()
        for name in model.state_dict()
    }
    model.load_state_dict(ema, strict=True, assign=True)
    alphas = _tensor(state["alphas_cumprod"], "alphas_cumprod").float().to(device)
    model = model.eval().to(device=device, dtype=torch.float32)
    model.requires_grad_(False)
    return model, alphas


def _case_tensors(
    case: CachedCase, bundle: Bundle, device: torch.device
) -> dict[str, torch.Tensor]:
    expected = {
        "latent": (1, 4, 64, 64, 64),
        "fine_mask": (1, 1, 64, 64, 64),
        "coarse_mask": (1, 1, 64, 64, 64),
        "context": (1, 10, 512),
        "features": (101,),
    }
    tensors: dict[str, torch.Tensor] = {}
    with np.load(case.file, allow_pickle=False) as cache:
        for name, shape in expected.items():
            array = np.asarray(cache[name], dtype=np.float32)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{case.lesion_id}: invalid {name} shape or finite values")
            tensors[name] = _torch_arrays.from_numpy(array.copy()).to(device)
    for name in ("fine_mask", "coarse_mask"):
        mask = tensors[name]
        if not bool(torch.all((mask == 0) | (mask == 1)).item()) or mask.sum().item() == 0:
            raise ValueError(f"{case.lesion_id}: {name} must be a nonempty binary mask")
    if bool(torch.any(tensors["fine_mask"] > tensors["coarse_mask"]).item()):
        raise ValueError(f"{case.lesion_id}: fine mask extends outside bounding box")
    center = _torch_arrays.from_numpy(bundle.center).to(device)
    scale = _torch_arrays.from_numpy(bundle.scale).to(device)
    tensors["normalized_features"] = ((tensors["features"] - center) / scale).unsqueeze(0)
    return tensors


def objective(
    model: nn.Module,
    adapter: RadiomicAdapter,
    values: Mapping[str, torch.Tensor],
    alphas: torch.Tensor,
    timestep: torch.Tensor,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the pinned inpainting epsilon L1 and auxiliary mask Dice objective."""
    latent, mask = values["latent"], values["fine_mask"]
    alpha = alphas[timestep].reshape(-1, 1, 1, 1, 1)
    noised = alpha.sqrt() * latent + (1 - alpha).sqrt() * noise
    noised = latent * (1 - mask) + noised * mask
    context = adapter(values["normalized_features"], values["context"])
    prediction = _tensor(model(torch.cat((noised, mask), dim=1), timestep, context), "prediction")
    epsilon = torch.abs((prediction[:, :4] - noise) * mask).sum()
    epsilon = epsilon / (values["coarse_mask"].sum() + 1e-6)
    probability = prediction[:, 4:].sigmoid()
    dice = 1 - (2 * (probability * mask).sum() + 1e-6) / (probability.sum() + mask.sum() + 1e-6)
    return epsilon + dice, epsilon, dice


def _write_json(path: Path, values: object) -> None:
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


def evaluate_selection(
    model: nn.Module,
    adapter: RadiomicAdapter,
    bundle: Bundle,
    alphas: torch.Tensor,
    *,
    seed: int,
    step: int,
    output: Path,
) -> None:
    """Record fixed-t500 selection noise loss without consuming training RNG.

    Every lesion receives an independently seeded CUDA generator. Aggregation is
    equal within each patient, then equal across patients. This diagnostic does
    not select a checkpoint or replace decoded enhancement/texture evaluation.
    """
    device = alphas.device
    rows: list[dict[str, object]] = []
    by_patient: dict[str, list[float]] = defaultdict(list)
    selected = sorted(
        (case for case in bundle.cases if case.split == "selection"),
        key=lambda case: (case.patient_id, case.lesion_id),
    )
    with torch.no_grad():
        for index, case in enumerate(selected):
            values = _case_tensors(case, bundle, device)
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + 10_000 + index)
            noise = torch.randn(values["latent"].shape, generator=generator, device=device)
            timestep = torch.tensor([500], device=device, dtype=torch.long)
            loss, epsilon, dice = objective(model, adapter, values, alphas, timestep, noise)
            value = float(loss.item())
            if not math.isfinite(value):
                raise ValueError(f"Non-finite selection noise loss for {case.lesion_id}")
            by_patient[case.patient_id].append(value)
            rows.append(
                {
                    "patient_id": case.patient_id,
                    "lesion_id": case.lesion_id,
                    "loss": value,
                    "epsilon_l1": float(epsilon.item()),
                    "mask_dice_loss": float(dice.item()),
                }
            )
    means = {patient: float(np.mean(values)) for patient, values in by_patient.items()}
    result = {
        "step": step,
        "timestep": 500,
        "patient_mean_loss": float(np.mean(list(means.values()))) if means else None,
        "patients": means,
        "lesions": rows,
        "scope": "Fixed-noise diagnostic; checkpoint selection uses decoded appearance",
    }
    _write_json(output / f"selection-noise-{step:06d}.json", result)
    print(
        json.dumps({"selection_step": step, "patient_mean_loss": result["patient_mean_loss"]}),
        flush=True,
    )


def restore_training(
    path: Path,
    adapter: RadiomicAdapter,
    optimizer: torch.optim.AdamW,
    rng: np.random.Generator,
    bundle: Bundle,
    *,
    seed: int,
    learning_rate: float,
) -> int:
    """Restore optimizer and all training RNG states after validating the data contract."""
    metadata = _object(json.loads((path.parent / "metadata.json").read_text()), "metadata")
    expected: dict[str, object] = {
        "feature_names": list(bundle.feature_names),
        "feature_center": bundle.center.tolist(),
        "feature_scale": bundle.scale.tolist(),
        "context_fields": list(bundle.context_fields),
        "target_space": bundle.target_space,
        "seed": seed,
        "learning_rate": learning_rate,
        "train_patients": sorted({c.patient_id for c in bundle.cases if c.split == "train"}),
        "selection_patients": sorted(
            {c.patient_id for c in bundle.cases if c.split == "selection"}
        ),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Resume checkpoint differs from the training data/normalization/seed/lr contract"
        )
    raw: object = torch.load(path, map_location="cpu", weights_only=False)
    state = _object(raw, "training_state")
    weights = {
        name: _tensor(value, name) for name, value in _object(state["adapter"], "adapter").items()
    }
    adapter.load_state_dict(weights, strict=True)
    optimizer.load_state_dict(_object(state["optimizer"], "optimizer"))
    step = state["step"]
    if not isinstance(step, int) or isinstance(step, bool):
        raise ValueError("Resume step must be an integer")
    torch.set_rng_state(_tensor(state["torch_rng"], "torch_rng"))
    cuda_states = state["cuda_rng"]
    if not isinstance(cuda_states, list):
        raise ValueError("Resume CUDA RNG state must be a list")
    torch.cuda.set_rng_state_all(
        [_tensor(item, "cuda_rng") for item in cast(list[object], cuda_states)]
    )
    rng.bit_generator.state = _object(state["numpy_rng"], "numpy_rng")
    return step


def save_adapter(
    output: Path,
    adapter: RadiomicAdapter,
    bundle: Bundle,
    *,
    step: int,
    seed: int,
    learning_rate: float,
) -> None:
    """Write four portable FP32 tensors and the train-only feature contract."""
    output.mkdir(parents=True, exist_ok=False)
    weights = {
        name: _tensor(value, name).detach().cpu().float().contiguous()
        for name, value in adapter.state_dict().items()
    }
    _save_file(weights, output / "adapter.safetensors")
    _write_json(
        output / "metadata.json",
        {
            "schema": "lesioncontrol.radiomic_adapter.v1",
            "feature_names": bundle.feature_names,
            "feature_center": bundle.center.tolist(),
            "feature_scale": bundle.scale.tolist(),
            "target_space": bundle.target_space,
            "token_count": 10,
            "hidden_dim": 64,
            "conditioning_schema": "training",
            "context_fields": bundle.context_fields,
            "training_step": step,
            "seed": seed,
            "learning_rate": learning_rate,
            "train_patients": sorted({c.patient_id for c in bundle.cases if c.split == "train"}),
            "selection_patients": sorted(
                {c.patient_id for c in bundle.cases if c.split == "selection"}
            ),
            "training_objective": "Pinned masked epsilon L1 / coarse bbox volume + mask Dice",
            "base_weights": "Released frozen LINet EMA; context enabled",
            "latent_shape": [1, 4, 64, 64, 64],
            "dtype": "float32",
            "backend": "PyTorch CUDA reference training; MLX inference evaluated separately",
        },
    )


def export_parity(
    output: Path,
    model: nn.Module,
    adapter: RadiomicAdapter,
    values: Mapping[str, torch.Tensor],
) -> None:
    """Export a real central 8³ latent crop and a nonzero learned adapter residual."""
    bounds = (slice(None), slice(None), slice(28, 36), slice(28, 36), slice(28, 36))
    latent, mask = values["latent"][bounds], values["fine_mask"][bounds]
    inputs = torch.cat((latent, mask), dim=1)
    timestep = torch.tensor([500], device=latent.device, dtype=torch.long)
    with torch.no_grad():
        normalized = values["normalized_features"]
        context = values["context"]
        adapted = adapter(normalized, context)
        prediction = _tensor(model(inputs, timestep, adapted), "parity prediction")
        base_prediction = _tensor(model(inputs, timestep, context), "base prediction")
    tensors = {
        "raw_features": values["features"].unsqueeze(0),
        "normalized_features": normalized,
        "base_context": context,
        "adapted_context": adapted,
        "input": inputs,
        "timestep": timestep,
        "prediction": prediction,
        "base_prediction": base_prediction,
    }
    _save_file({name: value.detach().cpu().contiguous() for name, value in tensors.items()}, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--resume", type=Path, help="Exact optimizer/RNG checkpoint to continue")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU or Metal training fallback")
    if args.steps < 1 or args.save_every < 1 or not 0 < args.learning_rate < math.inf:
        raise ValueError("Positive steps, save interval and finite learning rate are required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    _torch_arrays.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    bundle = read_bundle(args.index)
    if args.mode == "train" and not bundle.dataset_complete:
        raise ValueError(
            "Training requires dataset_complete=true; partial caches are preflight-only"
        )
    if args.mode == "preflight" and args.resume is not None:
        raise ValueError("The preflight starts from zero initialization")
    patients: dict[str, list[CachedCase]] = defaultdict(list)
    for case in bundle.cases:
        if case.split == "train":
            patients[case.patient_id].append(case)
    patient_ids = sorted(patients)
    _write_json(
        args.output / "request.json",
        {
            "mode": args.mode,
            "index": str(args.index),
            "weights": str(args.weights),
            "steps": 1 if args.mode == "preflight" else args.steps,
            "learning_rate": args.learning_rate,
            "save_every": args.save_every,
            "seed": args.seed,
            "sampling": "Uniform patient, then uniform lesion within that patient",
            "optimizer": "AdamW, weight_decay=0.01, gradient clip norm1",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "safetensors": importlib.metadata.version("safetensors"),
            "cuda_device": torch.cuda.get_device_name(device),
            "full_latent_training": True,
            "TF32": False,
            "resume": str(args.resume) if args.resume is not None else None,
        },
    )
    model, alphas = load_frozen_unet(args.weights, device)
    if not np.array_equal(alphas.cpu().numpy(), bundle.alphas_cumprod):
        raise ValueError("Native checkpoint and exported MLX alpha schedules differ")
    adapter = RadiomicAdapter().to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    steps = 1 if args.mode == "preflight" else args.steps
    first_step = (
        restore_training(
            args.resume,
            adapter,
            optimizer,
            rng,
            bundle,
            seed=args.seed,
            learning_rate=args.learning_rate,
        )
        if args.resume is not None
        else 0
    )
    if steps <= first_step:
        raise ValueError("Target total steps must exceed the restored step")
    if args.mode == "train":
        evaluate_selection(
            model,
            adapter,
            bundle,
            alphas,
            seed=args.seed,
            step=first_step,
            output=args.output,
        )
    final_values: dict[str, torch.Tensor] | None = None
    for step in range(first_step + 1, steps + 1):
        patient = patient_ids[int(rng.integers(len(patient_ids)))]
        cases = patients[patient]
        case = cases[int(rng.integers(len(cases)))]
        values = _case_tensors(case, bundle, device)
        if step == 1 and not torch.equal(
            adapter(values["normalized_features"], values["context"]), values["context"]
        ):
            raise ValueError("Zero-initialized adapter changed the base context")
        timestep = torch.randint(0, 1000, (1,), device=device, dtype=torch.long)
        noise = torch.randn_like(values["latent"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, epsilon, dice = objective(model, adapter, values, alphas, timestep, noise)
        if not bool(torch.isfinite(loss).item()):
            raise ValueError(f"Non-finite loss at step{step}")
        cast(_BackwardBoundary, loss).backward()
        gradients = [p.grad for p in adapter.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all().item()) for g in gradients):
            raise ValueError(f"Missing or non-finite adapter gradients at step{step}")
        norm = nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
        if float(norm.item()) == 0:
            raise ValueError(f"Zero adapter gradient at step{step}")
        cast(_OptimizerStepBoundary, optimizer).step()
        torch.cuda.synchronize()
        record = {
            "step": step,
            "patient_id": patient,
            "lesion_id": case.lesion_id,
            "timestep": int(timestep.item()),
            "loss": float(loss.item()),
            "epsilon_l1": float(epsilon.item()),
            "mask_dice_loss": float(dice.item()),
            "gradient_norm_before_clip": float(norm.item()),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        print(json.dumps(record), flush=True)
        with (args.output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if step % args.save_every == 0 or step == steps:
            directory = args.output / f"step-{step:06d}"
            save_adapter(
                directory,
                adapter,
                bundle,
                step=step,
                seed=args.seed,
                learning_rate=args.learning_rate,
            )
            torch.save(
                {
                    "adapter": adapter.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "numpy_rng": rng.bit_generator.state,
                    "request": vars(args),
                },
                directory / "training_state.pt",
            )
            if args.mode == "train":
                evaluate_selection(
                    model,
                    adapter,
                    bundle,
                    alphas,
                    seed=args.seed,
                    step=step,
                    output=args.output,
                )
        final_values = values
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Frozen base weights unexpectedly received gradients")
    if args.mode == "preflight":
        assert final_values is not None
        export_parity(args.output / "parity.safetensors", model, adapter, final_values)
    _write_json(
        args.output / "completed.json",
        {"completed_steps": steps, "base_parameters_optimized": False, "finite": True},
    )


if __name__ == "__main__":
    main()
