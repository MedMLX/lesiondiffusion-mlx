"""Select learned HCC conditioning using decoded enhancement and MLX radiomics.

Reference targets belong to unseen selection patients. Their clean host anatomy
is shared with registered training patients, so this is conditional reference
generalization. A matched zero-gain comparator isolates the learned residual.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import TypedDict, cast

import numpy as np

from lesioncontrol.backend import GenerationControls
from lesioncontrol.controller import sample_reference
from lesioncontrol.profile import build_profile, load_host, load_profile, save_profile


class SelectionCase(TypedDict):
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


class Variant(TypedDict):
    name: str
    checkpoint: str
    gain: float
    retained_steps: int


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def register(root: Path, hosts: Path, bundle: Path, training: Path, weights: Path) -> None:
    """Freeze patient targets, matched controls and scalar selection criteria."""
    source = cast(dict[str, object], json.loads(hosts.read_text()))
    if source["schema"] != "lesioncontrol.conditional_reference_selection.v1":
        raise ValueError("Unsupported conditional selection manifest")
    cases = cast(list[SelectionCase], source["cases"])
    if not cases:
        raise ValueError("No prepared conditional selection cases")
    training_index = cast(dict[str, object], json.loads(bundle.read_text()))
    names = cast(list[str], training_index["feature_names"])
    scales = cast(list[float], training_index["feature_scale"])
    contrast_scale = scales[names.index("original_firstorder_Mean")]
    variants: list[Variant] = [
        {
            "name": "gain0-steps25",
            "checkpoint": str(training / "step-000025"),
            "gain": 0,
            "retained_steps": 25,
        }
    ]
    variants.extend(
        {
            "name": f"step{step}-gain1-steps25",
            "checkpoint": str(training / f"step-{step:06d}"),
            "gain": 1,
            "retained_steps": 25,
        }
        for step in (25, 50, 100)
    )
    root.mkdir(parents=True, exist_ok=False)
    _write(
        root / "plan.json",
        {
            "schema": "lesioncontrol.adapter_selection.v1",
            "source_hosts": str(hosts.resolve()),
            "training_bundle": str(bundle.resolve()),
            "weights": str(weights.resolve()),
            "training_root": str(training.resolve()),
            "cases": cases,
            "initial_variants": variants,
            "search_seed": 23,
            "confirmation_seed": 24,
            "contrast_scale_hu": contrast_scale,
            "objective": (
                "0.5*(absolute fixed-host liver-contrast error / training contrast SD "
                "+ mean Canberra distance over the five texture families)"
            ),
            "aggregation": "Mean draws within each reference patient, then equal patient mean",
            "target_space": "liver_contrast",
            "scope": source["scope"],
            "next_stage": (
                "Best initial learned checkpoint: gains 0.5,2 at 25 steps and "
                "gains 0,0.5,1,2 at 14 steps; freeze best before seed24 confirmation"
            ),
        },
    )
    print(f"REGISTERED {len(cases)} conditional selection references", flush=True)


def evaluate(root: Path, variants: list[Variant], seed: int) -> list[dict[str, object]]:
    """Generate equal-input comparisons and retain every feature residual."""
    plan = cast(dict[str, object], json.loads((root / "plan.json").read_text()))
    cases = cast(list[SelectionCase], plan["cases"])
    scale = cast(float, plan["contrast_scale_hu"])
    records: list[dict[str, object]] = []
    for case in cases:
        profile_path = root / f"{case['id']}.profile.json"
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
        for variant in variants:
            output = root / "runs" / case["id"] / variant["name"] / f"seed-{seed}"
            metric_path = output / "selection_metrics.json"
            if metric_path.exists():
                metric = cast(dict[str, object], json.loads(metric_path.read_text()))
                if metric["variant"] != variant or metric["seed"] != seed:
                    raise ValueError("Existing decoded result differs from the requested controls")
                records.append(metric)
                continue
            if not Path(variant["checkpoint"]).joinpath("metadata.json").exists():
                print(f"WAITING {variant['checkpoint']}", flush=True)
                continue
            controls = GenerationControls(
                base_steps=25,
                retained_steps=variant["retained_steps"],
                conditioning_schema="training",
                surface="well-defined margin",
            )
            result = sample_reference(
                profile,
                host,
                controls,
                weights=Path(cast(str, plan["weights"])),
                output_dir=output,
                seed=seed,
                adapter=Path(variant["checkpoint"]),
                adapter_gain=variant["gain"],
            )
            comparison = result["reference_comparison"]
            mean = comparison["deviations"]["original_firstorder_Mean"]
            contrast_error = abs(mean["difference"])
            texture = float(
                np.mean(
                    [
                        value
                        for name, value in comparison["family_scores"].items()
                        if name != "firstorder"
                    ]
                )
            )
            metric: dict[str, object] = {
                "patient": case["patient"],
                "lesion": case["id"],
                "host_patient": case["host_patient"],
                "variant": variant,
                "seed": seed,
                "enhancement_abs_error_hu": contrast_error,
                "target_contrast_hu": mean["target"],
                "generated_contrast_hu": mean["generated"],
                "texture_canberra": texture,
                "family_scores": comparison["family_scores"],
                "score": 0.5 * (contrast_error / scale + texture),
                "all_feature_score": comparison["score"],
                "outputs": result["outputs"],
            }
            _write(metric_path, metric)
            records.append(metric)
            print(
                f"{case['id']} {variant['name']} seed{seed} "
                f"enhancement={contrast_error:.3f}HU texture={texture:.5f} "
                f"score={metric['score']:.5f}",
                flush=True,
            )
    return records


def summarize(root: Path) -> list[dict[str, object]]:
    """Average repeated draws per patient before ranking control configurations."""
    metrics = [
        cast(dict[str, object], json.loads(path.read_text()))
        for path in root.glob("runs/*/*/seed-*/selection_metrics.json")
    ]
    by_variant: dict[str, list[dict[str, object]]] = defaultdict(list)
    for metric in metrics:
        variant = cast(Variant, metric["variant"])
        by_variant[variant["name"]].append(metric)
    rows: list[dict[str, object]] = []
    for name, items in by_variant.items():
        by_patient: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in items:
            by_patient[cast(str, item["patient"])].append(item)
        row: dict[str, object] = {
            "name": name,
            "patients": len(by_patient),
            "draws": len(items),
            "variant": items[0]["variant"],
        }
        for field in ("score", "enhancement_abs_error_hu", "texture_canberra", "all_feature_score"):
            row[field] = float(
                np.mean(
                    [
                        np.mean([cast(float, item[field]) for item in values])
                        for values in by_patient.values()
                    ]
                )
            )
        rows.append(row)
    rows.sort(key=lambda row: cast(float, row["score"]))
    _write(root / "summary.json", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "initial", "tune", "confirm", "summary"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hosts", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--seed", type=int, choices=(24, 25), default=24)
    args = parser.parse_args()
    if args.command == "register":
        if not all((args.hosts, args.bundle, args.training, args.weights)):
            parser.error("Registration requires hosts, bundle, training and weights")
        register(args.root, args.hosts, args.bundle, args.training, args.weights)
        return
    plan = cast(dict[str, object], json.loads((args.root / "plan.json").read_text()))
    initial = cast(list[Variant], plan["initial_variants"])
    if args.command == "initial":
        evaluate(args.root, initial, 23)
    elif args.command == "tune":
        if (args.root / "selected.json").exists():
            raise ValueError("Selection is frozen; use a new campaign for further tuning")
        rows = summarize(args.root)
        completed = {
            cast(str, row["name"])
            for row in rows
            if row["patients"] == len(cast(list[object], plan["cases"]))
        }
        if any(variant["name"] not in completed for variant in initial):
            raise ValueError("Complete every initial checkpoint comparison before tuning")
        initial_names = {variant["name"] for variant in initial}
        eligible = [
            row
            for row in rows
            if row["name"] in initial_names
            and row["name"] in completed
            and cast(Variant, row["variant"])["gain"] == 1
        ]
        if not eligible:
            raise ValueError("No initial learned configuration has completed")
        best = cast(Variant, eligible[0]["variant"])
        variants: list[Variant] = []
        for steps, gains in ((25, (0.5, 2.0)), (14, (0.0, 0.5, 1.0, 2.0))):
            for gain in gains:
                variants.append(
                    {
                        **best,
                        "name": f"tuned-gain{gain:g}-steps{steps}",
                        "gain": gain,
                        "retained_steps": steps,
                    }
                )
        _write(args.root / "tuning_plan.json", variants)
        evaluate(args.root, variants, 23)
    elif args.command == "confirm":
        chosen = cast(Variant, json.loads((args.root / "selected.json").read_text()))
        if chosen["retained_steps"] == 25:
            control = initial[0]
        else:
            tuning = cast(list[Variant], json.loads((args.root / "tuning_plan.json").read_text()))
            control = next(variant for variant in tuning if variant["gain"] == 0)
        paired = evaluate(args.root, [chosen, control], args.seed)
        pairs = {
            name: [row for row in paired if cast(Variant, row["variant"])["name"] == name]
            for name in (chosen["name"], control["name"])
        }
        expected = len(cast(list[object], plan["cases"]))
        if any(len(rows) != expected for rows in pairs.values()):
            raise ValueError("Confirmation requires every paired reference patient")
        _write(
            args.root / f"confirmation-seed{args.seed}.json",
            {
                "seed": args.seed,
                "reference_patients": expected,
                "scope": plan["scope"],
                "selected": chosen,
                "matched_control": control,
                "means": {
                    name: {
                        field: float(np.mean([cast(float, row[field]) for row in rows]))
                        for field in ("score", "enhancement_abs_error_hu", "texture_canberra")
                    }
                    for name, rows in pairs.items()
                },
            },
        )
    for row in summarize(args.root):
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "name",
                        "patients",
                        "draws",
                        "enhancement_abs_error_hu",
                        "texture_canberra",
                        "score",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
