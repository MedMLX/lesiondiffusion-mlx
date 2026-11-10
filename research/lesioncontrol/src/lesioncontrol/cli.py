"""Command-line reference profiling, inverse matching and fresh-seed sampling."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import cast


def _host_arguments(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--host", type=Path, required=required)
    parser.add_argument("--liver-mask", type=Path, required=required)
    parser.add_argument("--placement-mask", type=Path, required=required)
    parser.add_argument("--known-tumor-mask", type=Path)
    parser.add_argument("--declared-clean", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(prog="lesioncontrol", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser(
        "profile", help="Extract a reference's appearance feature targets"
    )
    profile.add_argument("--image", type=Path, required=True)
    profile.add_argument("--mask", type=Path, required=True)
    profile.add_argument("--liver-mask", type=Path)
    profile.add_argument("--qc-record", type=Path, help="Reference QC object in JSON")
    profile.add_argument("--target-space", choices=("absolute_hu", "liver_contrast"), required=True)
    profile.add_argument("--target", action="append", default=[], metavar="FEATURE=VALUE")
    profile.add_argument("--output", type=Path, required=True)
    match = commands.add_parser(
        "match", help="Fit bounded controls to a reference profile on a clean host"
    )
    match.add_argument("--profile", type=Path, required=True)
    _host_arguments(match, required=True)
    match.add_argument("--weights", type=Path, required=True)
    match.add_argument("--output", type=Path, required=True)
    match.add_argument("--budget", type=int, default=12)
    match.add_argument("--seed", type=int, default=23)
    match.add_argument("--base-steps", type=int, default=25)
    generate = commands.add_parser(
        "generate", help="Sample a saved policy with a fresh seed, without refitting"
    )
    generate.add_argument("--policy", type=Path, required=True)
    _host_arguments(generate, required=False)
    generate.add_argument("--weights", type=Path)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--output", type=Path, required=True)
    sample = commands.add_parser("sample", help="Generate from a profile with a trained adapter")
    sample.add_argument("--profile", type=Path, required=True)
    _host_arguments(sample, required=True)
    sample.add_argument("--weights", type=Path, required=True)
    sample.add_argument("--adapter", type=Path)
    sample.add_argument("--adapter-gain", type=float, default=1.0)
    sample.add_argument("--base-steps", type=int, default=25)
    sample.add_argument("--retained-steps", type=int, default=25)
    sample.add_argument("--seed", type=int, required=True)
    sample.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from lesioncontrol.profile import build_profile, load_host, load_profile, save_profile

    if args.command == "profile":
        qc_record: dict[str, object] | None = None
        if args.qc_record is not None:
            supplied: object = json.loads(args.qc_record.read_text())
            if not isinstance(supplied, dict):
                raise ValueError("Reference QC record must be an object with string keys")
            fields = cast(dict[object, object], supplied)
            if any(not isinstance(key, str) for key in fields):
                raise ValueError("Reference QC record must be an object with string keys")
            qc_record = cast(dict[str, object], supplied)
        reference = build_profile(
            args.image,
            args.mask,
            args.liver_mask,
            target_space=args.target_space,
            qc_record=qc_record,
        )
        targets = dict(reference.features)
        for specification in args.target:
            name, value = specification.split("=", 1)
            if name not in targets:
                raise ValueError(f"Unknown reference target feature: {name}")
            target = float(value)
            if not math.isfinite(target):
                raise ValueError(f"Reference target must be finite: {name}")
            targets[name] = target
        reference = replace(reference, features=targets)
        save_profile(reference, args.output)
        print(
            json.dumps(
                {
                    "profile": str(args.output),
                    "target_space": reference.target_space,
                    "target_mean_hu": reference.target_mean_hu,
                    "features": len(reference.features),
                },
                indent=2,
            )
        )
        return
    from lesioncontrol.controller import generate_saved, match_reference

    if args.command == "sample":
        from lesioncontrol.backend import GenerationControls
        from lesioncontrol.controller import sample_reference

        host = load_host(
            args.host,
            args.liver_mask,
            args.placement_mask,
            known_tumor_mask=args.known_tumor_mask,
            declared_clean=args.declared_clean,
        )
        result = sample_reference(
            load_profile(args.profile),
            host,
            GenerationControls(
                base_steps=args.base_steps,
                retained_steps=args.retained_steps,
                conditioning_schema="training" if args.adapter else "native",
                surface="well-defined margin",
            ),
            weights=args.weights,
            output_dir=args.output,
            seed=args.seed,
            adapter=args.adapter,
            adapter_gain=args.adapter_gain,
        )
        print(
            json.dumps(
                {
                    "outputs": result["outputs"],
                    "reference_score": result["reference_comparison"]["score"],
                    "family_scores": result["reference_comparison"]["family_scores"],
                },
                indent=2,
            )
        )
        return

    if args.command == "match":
        host = load_host(
            args.host,
            args.liver_mask,
            args.placement_mask,
            known_tumor_mask=args.known_tumor_mask,
            declared_clean=args.declared_clean,
        )
        policy = match_reference(
            load_profile(args.profile),
            host,
            weights=args.weights,
            output_dir=args.output,
            budget=args.budget,
            seed=args.seed,
            base_steps=args.base_steps,
        )
        print(
            json.dumps(
                {
                    "policy": str(args.output / "policy.json"),
                    "score": policy["score"],
                    "unique_trials": policy["unique_trials"],
                    "evidence": policy["evidence"],
                },
                indent=2,
            )
        )
        return
    provided = (args.host is not None, args.liver_mask is not None, args.placement_mask is not None)
    if any(provided) and not all(provided):
        parser.error("A new host requires --host, --liver-mask and --placement-mask together")
    host = (
        load_host(
            args.host,
            args.liver_mask,
            args.placement_mask,
            known_tumor_mask=args.known_tumor_mask,
            declared_clean=args.declared_clean,
        )
        if all(provided)
        else None
    )
    record = generate_saved(
        args.policy, output_dir=args.output, seed=args.seed, host=host, weights=args.weights
    )
    print(
        json.dumps(
            {
                "outputs": record["outputs"],
                "seed": args.seed,
                "reference_score": record["reference_comparison"]["score"],
                "reference_family_scores": record["reference_comparison"]["family_scores"],
                "evidence": record["evidence"],
            },
            indent=2,
        )
    )
