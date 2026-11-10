"""Reference-specific inverse control of lesion appearance on an independent host."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution

from lesioncontrol.backend import GenerationControls, GenerationRecord, generate
from lesioncontrol.profile import (
    FeatureComparison,
    PreparedHost,
    ReferenceProfile,
    compare_features,
    extract_generated_features,
    load_host,
    load_profile,
    save_profile,
)

DENSITIES = ("hypodense lesion", "isodense lesion", "hyperdense lesion")


class TrialRecord(TypedDict):
    """One completed inverse-control trial and its reference residuals."""

    trial: str
    controls: dict[str, object]
    score: float
    family_scores: dict[str, float]
    outputs: dict[str, str]
    comparison: FeatureComparison
    top_residual_features: list[dict[str, object]]


class SampleRecord(GenerationRecord):
    """A fresh generated sample with its unchanged reference objective evaluated."""

    reference_comparison: FeatureComparison
    evidence: str


def _controls(values: NDArray[np.float64], base_steps: int) -> GenerationControls:
    kept, density, variation, margin, schema, offset, feature = (round(float(v)) for v in values)
    return GenerationControls(
        base_steps=base_steps,
        retained_steps=kept,
        density=DENSITIES[density],
        density_variation=("homogeneous", "heterogeneous")[variation],
        surface=("well-defined margin", "ill-defined margin")[margin],
        conditioning_schema=("native", "training")[schema],
        numeric_offset=offset if schema else 0,
        specific_features=("", "hepatocellular carcinoma")[feature],
    )


def _vector(controls: GenerationControls) -> list[float]:
    return [
        controls.retained_steps,
        DENSITIES.index(controls.density),
        ("homogeneous", "heterogeneous").index(controls.density_variation),
        ("well-defined margin", "ill-defined margin").index(controls.surface),
        ("native", "training").index(controls.conditioning_schema),
        controls.numeric_offset,
        ("", "hepatocellular carcinoma").index(controls.specific_features),
    ]


def _condition(target_space: str, target_mean_hu: float, host: PreparedHost) -> float:
    """Translate a contrast target to the independent host's absolute HU level."""
    return (
        target_mean_hu + host.ring_mean_hu if target_space == "liver_contrast" else target_mean_hu
    )


def _host_record(host: PreparedHost) -> dict[str, object]:
    return {
        "image": str(host.image_path),
        "liver_mask": str(host.liver_mask_path),
        "placement_mask": str(host.placement_mask_path),
        "known_tumor_mask": str(host.known_tumor_mask_path) if host.known_tumor_mask_path else None,
        "declared_clean": host.declared_clean,
    }


def match_reference(
    profile: ReferenceProfile,
    host: PreparedHost,
    *,
    weights: Path,
    output_dir: Path,
    budget: int = 12,
    seed: int = 23,
    base_steps: int = 25,
    initial_controls: Sequence[GenerationControls] = (),
) -> dict[str, object]:
    """Fit controls to supplied reference features, keeping reference pixels out of generation.

    The finite budget counts objective calls; exact duplicate controls reuse the
    completed trial. The score measures conditional feature fidelity to this one
    reference. It does not estimate performance on independent tumors.
    """
    if budget < 5 or not 3 <= base_steps <= 999:
        raise ValueError("Matching requires budget >=5 and 3 <= base_steps <=999")
    output_dir, weights = output_dir.resolve(), weights.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    save_profile(profile, output_dir / "profile.json")
    population_size = min(12, max(5, budget // 2))
    generations = max(0, budget // population_size - 1)
    bounds = [(3, base_steps), (0, 2), (0, 1), (0, 1), (0, 1), (-25, 25), (0, 1)]
    rng = np.random.default_rng(seed)
    initial = np.column_stack(
        [rng.integers(low, high + 1, population_size) for low, high in bounds]
    ).astype(np.float64)
    starting = [
        GenerationControls(base_steps=base_steps, retained_steps=base_steps),
        *initial_controls,
        GenerationControls(
            base_steps=base_steps,
            retained_steps=min(14, base_steps),
            density="isodense lesion",
            density_variation="heterogeneous",
            surface="well-defined margin",
            conditioning_schema="training",
            numeric_offset=0,
        ),
    ]
    for index, controls in enumerate(starting[:population_size]):
        initial[index] = _vector(controls)
    request = {
        "schema": "lesioncontrol.match.v1",
        "target_space": profile.target_space,
        "target_mean_hu": profile.target_mean_hu,
        "host": _host_record(host),
        "weights": str(weights),
        "maximum_trials": budget,
        "planned_objective_calls": population_size * (generations + 1),
        "seed": seed,
        "base_steps": base_steps,
        "optimizer": "scipy.optimize.differential_evolution",
        "population": initial.tolist(),
        "generations": generations,
        "objective": "Single-reference family-balanced Canberra distance; supplied shape excluded",
        "reference_conditioning": (
            "Feature targets guide control selection; "
            "only the intended HU level conditions generation. "
            "The generator sees independent clean-host pixels."
        ),
    }
    (output_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    untouched = compare_features(
        profile, extract_generated_features(profile, host, host.image_path)
    )
    (output_dir / "baselines.json").write_text(
        json.dumps({"untouched_host": untouched, "untuned_full_dpm_trial": "trial-000"}, indent=2)
        + "\n"
    )
    trials: list[TrialRecord] = []
    cache: dict[str, float] = {}

    def evaluate(values: NDArray[np.float64]) -> float:
        controls = _controls(values, base_steps)
        key = json.dumps(asdict(controls), sort_keys=True)
        if key in cache:
            return cache[key]
        trial = f"trial-{len(trials):03d}"
        directory = output_dir / trial
        generated = generate(
            host,
            controls,
            weights=weights,
            output_dir=directory,
            seed=seed,
            conditioning_mean_hu=_condition(profile.target_space, profile.target_mean_hu, host),
        )
        features = extract_generated_features(
            profile, host, Path(generated["outputs"]["synthetic_image"])
        )
        comparison = compare_features(profile, features)
        score = float(comparison["score"])
        if not np.isfinite(score):
            raise ValueError(f"{trial}: reference feature comparison produced non-finite score")
        residuals: list[dict[str, object]] = [
            {"feature": name, **comparison["deviations"][name]}
            for name in sorted(
                comparison["deviations"],
                key=lambda name: comparison["deviations"][name]["canberra"],
                reverse=True,
            )[:10]
        ]
        record: TrialRecord = {
            "trial": trial,
            "controls": asdict(controls),
            "score": score,
            "family_scores": comparison["family_scores"],
            "outputs": generated["outputs"],
            "comparison": comparison,
            "top_residual_features": residuals,
        }
        (directory / "features.json").write_text(json.dumps(features, indent=2) + "\n")
        (directory / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        trials.append(record)
        cache[key] = score
        with (output_dir / "trials.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        best = min(trials, key=lambda item: float(item["score"]))
        (output_dir / "best.json").write_text(json.dumps(best, indent=2) + "\n")
        print(f"{trial} score={score:.6f} best={float(best['score']):.6f}", flush=True)
        return score

    result = differential_evolution(
        evaluate,
        bounds,
        init=initial,
        integrality=[True] * len(bounds),
        maxiter=generations,
        polish=False,
        workers=1,
        rng=seed,
        tol=0,
        atol=0,
    )
    best = min(trials, key=lambda item: float(item["score"]))
    policy: dict[str, object] = {
        "schema": "lesioncontrol.policy.v1",
        "policy": best["trial"],
        "controls": best["controls"],
        "target_space": profile.target_space,
        "target_mean_hu": profile.target_mean_hu,
        "host": request["host"],
        "weights": str(weights),
        "reference_profile": str(output_dir / "profile.json"),
        "score": best["score"],
        "baseline_scores": {
            "unchanged_host": untouched["score"],
            "untuned_full_dpm": trials[0]["score"],
        },
        "improved_over_untuned": best["score"] < trials[0]["score"],
        "improved_over_unchanged_host": best["score"] < untouched["score"],
        "top_residual_features": best["top_residual_features"],
        "fit_seed": seed,
        "objective_calls": int(result.nfev),
        "unique_trials": len(trials),
        "maximum_trials": budget,
        "optimizer_message": str(result.message),
        "evidence": "Conditional fidelity to supplied reference features on this host and fit seed",
    }
    (output_dir / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    return policy


def generate_saved(
    policy_path: Path,
    *,
    output_dir: Path,
    seed: int,
    host: PreparedHost | None = None,
    weights: Path | None = None,
) -> SampleRecord:
    """Sample a saved control policy with a new seed, without fitting again."""
    policy = json.loads(policy_path.read_text())
    if policy["schema"] != "lesioncontrol.policy.v1":
        raise ValueError("Unsupported lesioncontrol policy schema")
    profile = load_profile(Path(policy["reference_profile"]))
    if (
        policy["target_space"] != profile.target_space
        or policy["target_mean_hu"] != profile.target_mean_hu
    ):
        raise ValueError("Saved policy target space or mean differs from its reference profile")
    if host is None:
        source = policy["host"]
        host = load_host(
            Path(source["image"]),
            Path(source["liver_mask"]),
            Path(source["placement_mask"]),
            known_tumor_mask=Path(source["known_tumor_mask"])
            if source["known_tumor_mask"]
            else None,
            declared_clean=source["declared_clean"],
        )
    controls = GenerationControls(**policy["controls"])
    return sample_reference(
        profile,
        host,
        controls,
        weights=weights or Path(policy["weights"]),
        output_dir=output_dir,
        seed=seed,
    )


def sample_reference(
    profile: ReferenceProfile,
    host: PreparedHost,
    controls: GenerationControls,
    *,
    weights: Path,
    output_dir: Path,
    seed: int,
    adapter: Path | None = None,
    adapter_gain: float = 1.0,
) -> SampleRecord:
    """Generate one measured reference-conditioned lesion with an optional trained MLP.

    The reference provides feature targets only. Learned checkpoints must declare
    the same intensity space, and their ten-token schema is explicit in controls.
    A zero gain retains that schema while disabling the learned residual, making
    it a matched control for measuring the effect of adapter training.
    """
    from lesioncontrol.learned import load_adapter

    if adapter is None and adapter_gain != 1:
        raise ValueError("adapter_gain requires a learned adapter checkpoint")
    conditioner = None
    if adapter is not None:
        if controls.conditioning_schema != "training":
            raise ValueError("A learned adapter requires conditioning_schema='training'")
        conditioner = load_adapter(adapter).bind(
            profile.features, target_space=profile.target_space, gain=adapter_gain
        )
    generated = generate(
        host,
        controls,
        weights=weights,
        output_dir=output_dir,
        seed=seed,
        conditioning_mean_hu=_condition(profile.target_space, profile.target_mean_hu, host),
        conditioner=conditioner,
    )
    features = extract_generated_features(
        profile, host, Path(generated["outputs"]["synthetic_image"])
    )
    comparison = compare_features(profile, features)
    save_profile(profile, output_dir / "reference_profile.json")
    (output_dir / "features.json").write_text(json.dumps(features, indent=2) + "\n")
    (output_dir / "reference_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return {
        **generated,
        "reference_comparison": comparison,
        "evidence": (
            "Conditional fidelity to supplied reference features on this host and sampling seed"
        ),
    }
