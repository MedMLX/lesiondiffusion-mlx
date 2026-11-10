"""Run frozen, paired-seed lesion fidelity and attenuation-aligned diagnostics.

The adapter, text, sampler and phenotype targets are identical within a pair;
gain zero disables only the learned residual. Mean alignment changes an in-memory
copy used for feature measurement, never the saved synthetic CT or generation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal, TypedDict, cast

import nibabel as nib
import numpy as np

from lesioncontrol.backend import GenerationControls
from lesioncontrol.controller import sample_reference
from lesioncontrol.profile import (
    build_profile,
    compare_features,
    extract_features,
    load_host,
    load_profile,
    save_profile,
)

type Cohort = Literal["selection", "heldout", "previously_evaluated"]


class PreparedCase(TypedDict):
    id: str
    patient: str
    host_patient: str
    real_ct: str
    real_mask: str
    real_liver_mask: str
    clean_ct: str
    liver_mask: str
    placed_mask: str
    known_tumors_mask: str


class CaseRecord(PreparedCase):
    lesion: str
    cohort: Cohort
    target_profile: str


class DrawRecord(TypedDict):
    patient: str
    lesion: str
    cohort: Cohort
    seed: int
    gain: int
    comparison_path: str
    mean_aligned_comparison_path: str
    metrics_path: str
    output_dir: str


def write_json(path: Path, value: object) -> None:
    """Publish a complete JSON artifact on the same filesystem."""
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _file_record(path: Path) -> dict[str, object]:
    information = path.stat()
    return {
        "path": str(path.resolve()),
        "size": information.st_size,
        "mtime_ns": information.st_mtime_ns,
    }


def register(root: Path, source: Path, cohort: Cohort, adapter: Path, weights: Path) -> None:
    """Freeze all twenty seeds, cases, input identities and model controls."""
    source_data = cast(dict[str, object], json.loads(source.read_text()))
    prepared = cast(list[PreparedCase], source_data["cases"])
    if not prepared or len({case["patient"] for case in prepared}) != len(prepared):
        raise ValueError("This assay requires one explicit reference lesion per patient")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    adapter, weights = adapter.resolve(), weights.resolve()
    inputs = [
        _file_record(Path(case[key]))
        for case in prepared
        for key in (
            "real_ct",
            "real_mask",
            "real_liver_mask",
            "clean_ct",
            "liver_mask",
            "placed_mask",
            "known_tumors_mask",
        )
    ]
    cases: list[CaseRecord] = [
        {
            **case,
            "lesion": case["id"],
            "cohort": cohort,
            "target_profile": str(root / f"{case['id']}.profile.json"),
        }
        for case in prepared
    ]
    plan = {
        "schema": "lesioncontrol.reliability_plan.v1",
        "source_manifest": str(source.resolve()),
        "source_scope": source_data.get("scope"),
        "cohort": cohort,
        "cases": cases,
        "seeds": list(range(1000, 1020)),
        "adapter": str(adapter),
        "weights": str(weights),
        "adapter_files": [
            _file_record(adapter / name) for name in ("metadata.json", "adapter.safetensors")
        ],
        "input_files": inputs,
        "controls": asdict(
            GenerationControls(conditioning_schema="training", surface="well-defined margin")
        ),
        "gains": [0, 1],
        "expected_draws": len(cases) * 40,
        "frozen_model": "Selected150-update adapter; no training, gain search or seed selection",
        "analysis_unit": "Reference patient; seeds are paired technical repetitions",
        "primary_contrast": "Lesion mean minus unchanged clean-host3to10mm liver-ring mean",
        "additional_diagnostic": (
            "Shift liver-centered generated CT uniformly until its ROI mean matches "
            "reference; recompute identical features; do not alter synthetic image"
        ),
        "failure_policy": (
            "Stop on model/QC/feature/setup failure; preserve attempt events, traceback "
            "and partial artifacts; never replace a predetermined seed. Report planned, "
            "attempted and valid pair counts. Incomplete campaigns are not completed."
        ),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "medmlx-lesioncontrol",
                "lesiondiffusion-mlx",
                "mlxradiomics",
                "mlx",
                "numpy",
                "scipy",
            )
        },
    }
    write_json(root / "plan.json", plan)
    write_json(
        root / "index.json",
        {
            "schema": "lesioncontrol.reliability_runs.v1",
            "cases": cases,
            "seeds": plan["seeds"],
            "draws": [],
        },
    )
    print(
        f"REGISTERED {cohort}: {len(cases)} patients, 20 paired seeds, {len(cases) * 40} draws",
        flush=True,
    )


def run(root: Path) -> None:
    """Complete paired generation, retaining ordinary and mean-aligned features."""
    root = root.resolve()
    plan = cast(dict[str, object], json.loads((root / "plan.json").read_text()))
    if plan["schema"] != "lesioncontrol.reliability_plan.v1":
        raise ValueError("Unsupported reliability plan")
    for record in cast(list[dict[str, object]], plan["input_files"]) + cast(
        list[dict[str, object]], plan["adapter_files"]
    ):
        if _file_record(Path(cast(str, record["path"]))) != record:
            raise ValueError("A frozen input or adapter file changed after registration")
    cases = cast(list[CaseRecord], plan["cases"])
    seeds = cast(list[int], plan["seeds"])
    controls = GenerationControls(conditioning_schema="training", surface="well-defined margin")
    if asdict(controls) != plan["controls"]:
        raise ValueError("Registered generation controls differ from this runner")
    draws: list[DrawRecord] = []
    for case in cases:
        profile_path = Path(case["target_profile"])
        if not profile_path.exists():
            save_profile(
                build_profile(
                    Path(case["real_ct"]),
                    Path(case["real_mask"]),
                    Path(case["real_liver_mask"]),
                    target_space="liver_contrast",
                ),
                profile_path,
            )
        profile = load_profile(profile_path)
        host = load_host(
            Path(case["clean_ct"]),
            Path(case["liver_mask"]),
            Path(case["placed_mask"]),
            known_tumor_mask=Path(case["known_tumors_mask"]),
        )
        for seed in seeds:
            pair: list[DrawRecord] = []
            for gain in (0, 1):
                with (root / "attempts.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "patient": case["patient"],
                                "seed": seed,
                                "gain": gain,
                                "status": "started",
                            }
                        )
                        + "\n"
                    )
                output = root / "runs" / case["lesion"] / f"seed-{seed}" / f"gain-{gain}"
                comparison_path = output / "reference_comparison.json"
                if not comparison_path.exists():
                    sample_reference(
                        profile,
                        host,
                        controls,
                        weights=Path(cast(str, plan["weights"])),
                        output_dir=output,
                        seed=seed,
                        adapter=Path(cast(str, plan["adapter"])),
                        adapter_gain=gain,
                    )
                arguments = cast(
                    dict[str, object], json.loads((output / "arguments.json").read_text())
                )
                if arguments["seed"] != seed or arguments["controls"] != asdict(controls):
                    raise ValueError("Cached draw has different controls or seed")
                expected_paths = {
                    "image_path": str(Path(case["clean_ct"]).resolve()),
                    "organ_mask_path": str(Path(case["liver_mask"]).resolve()),
                    "lesion_mask_path": str(Path(case["placed_mask"]).resolve()),
                    "weights_path": str(Path(cast(str, plan["weights"])).resolve()),
                }
                if any(arguments.get(key) != value for key, value in expected_paths.items()):
                    raise ValueError("Cached draw has different image, mask or base weights")
                learned = arguments.get("learned_adapter")
                if not isinstance(learned, dict):
                    raise ValueError("Cached draw lacks learned adapter metadata")
                learned_fields = cast(dict[str, object], learned)
                cached_gain = learned_fields.get("gain")
                if (
                    isinstance(cached_gain, bool)
                    or not isinstance(cached_gain, int | float)
                    or cached_gain != gain
                    or learned_fields.get("checkpoint")
                    != str(Path(cast(str, plan["adapter"])).resolve())
                ):
                    raise ValueError("Cached draw has a different adapter checkpoint or gain")
                desired_mean_hu = profile.target_mean_hu + (
                    host.ring_mean_hu if profile.target_space == "liver_contrast" else 0.0
                )
                if (
                    arguments.get("conditioning_mean_hu") != desired_mean_hu
                    or arguments.get("target_mean_hu") != desired_mean_hu + controls.numeric_offset
                ):
                    raise ValueError("Cached draw has a different numeric HU condition")
                previous_profile = load_profile(output / "reference_profile.json")
                if (
                    previous_profile.features != profile.features
                    or previous_profile.target_space != profile.target_space
                ):
                    raise ValueError("Cached draw used another reference target")
                validation = cast(
                    dict[str, object], json.loads((output / "validation.json").read_text())
                )
                if validation["valid"] is not True or validation["actual_steps"] != 25:
                    raise ValueError("Generated draw did not qualify")
                result = cast(dict[str, object], json.loads((output / "result.json").read_text()))
                outputs = cast(dict[str, str], result["outputs"])
                aligned_path = output / "mean_aligned_comparison.json"
                metrics_path = output / "reliability_metrics.json"
                if not aligned_path.exists() or not metrics_path.exists():
                    image = np.asarray(
                        nib.Nifti1Image.from_filename(outputs["synthetic_image"]).dataobj,
                        dtype=np.float32,
                    )
                    relative = image - np.float32(host.ring_mean_hu)
                    actual_mean = float(np.mean(relative[host.placement_mask], dtype=np.float64))
                    shift = profile.target_mean_hu - actual_mean
                    aligned = relative + np.float32(shift)
                    aligned_features = extract_features(
                        aligned, host.placement_mask, profile.settings
                    )
                    comparison = compare_features(profile, aligned_features)
                    write_json(
                        aligned_path,
                        {
                            **comparison,
                            "diagnostic": {
                                "uniform_shift_hu": shift,
                                "synthetic_image_unchanged": True,
                                "purpose": (
                                    "Attenuation/bin-origin sensitivity; "
                                    "not pure spatial-texture isolation"
                                ),
                            },
                        },
                    )
                    generated_ring = float(np.mean(image[host.ring_mask], dtype=np.float64))
                    original_comparison = cast(
                        dict[str, object], json.loads(comparison_path.read_text())
                    )
                    deviations = cast(
                        dict[str, dict[str, float]], original_comparison["deviations"]
                    )
                    write_json(
                        metrics_path,
                        {
                            "patient": case["patient"],
                            "lesion": case["lesion"],
                            "seed": seed,
                            "gain": gain,
                            "target_contrast_hu": profile.target_mean_hu,
                            "fixed_generated_contrast_hu": deviations["original_firstorder_Mean"][
                                "generated"
                            ],
                            "actual_generated_contrast_hu": float(
                                np.mean(image[host.placement_mask], dtype=np.float64)
                            )
                            - generated_ring,
                            "generated_ring_shift_hu": generated_ring - host.ring_mean_hu,
                            "mean_alignment_shift_hu": shift,
                        },
                    )
                pair.append(
                    {
                        "patient": case["patient"],
                        "lesion": case["lesion"],
                        "cohort": case["cohort"],
                        "seed": seed,
                        "gain": gain,
                        "comparison_path": str(comparison_path),
                        "mean_aligned_comparison_path": str(aligned_path),
                        "metrics_path": str(metrics_path),
                        "output_dir": str(output),
                    }
                )
                with (root / "attempts.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "patient": case["patient"],
                                "seed": seed,
                                "gain": gain,
                                "status": "completed",
                            }
                        )
                        + "\n"
                    )
            draws.extend(pair)
            write_json(
                root / "index.json",
                {
                    "schema": "lesioncontrol.reliability_runs.v1",
                    "cases": cases,
                    "seeds": seeds,
                    "draws": draws,
                },
            )
            print(
                f"PAIRED {case['patient']} seed{seed}; {len(draws)}/{plan['expected_draws']} draws",
                flush=True,
            )
    if len(draws) != plan["expected_draws"]:
        raise ValueError("Reliability campaign has incomplete pairs")
    write_json(
        root / "completed.json",
        {
            "patients": len(cases),
            "seeds_per_patient": len(seeds),
            "draws": len(draws),
            "frozen_controls": True,
            "mean_alignment_diagnostic": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cohort", choices=("selection", "heldout", "previously_evaluated"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    if args.command == "register":
        if any(value is None for value in (args.source, args.cohort, args.adapter, args.weights)):
            parser.error("Registration requires source, cohort, adapter and weights")
        register(args.root, args.source, args.cohort, args.adapter, args.weights)
    else:
        run(args.root)


if __name__ == "__main__":
    main()
