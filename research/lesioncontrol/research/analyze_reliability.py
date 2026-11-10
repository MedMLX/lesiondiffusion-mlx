"""Describe paired radiomic fidelity across seeds without refitting a generator.

The reference patient is the analysis unit. Seeds are technical repetitions,
paired between gain zero and one. Cohort intervals resample patient means while
retaining the full observed seed panel; they describe conditional reference
fidelity in the registered host setting. Feature tables are descriptive and do
not perform 101 unadjusted significance tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FAMILIES = ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
MEAN = "original_firstorder_Mean"
type Cohort = Literal["selection", "heldout", "previously_evaluated"]
type Space = Literal["raw", "mean_aligned"]
type Row = dict[str, object]
type Floats = NDArray[np.float64]


@dataclass(frozen=True)
class Case:
    """One explicitly registered reference lesion per patient."""

    patient: str
    lesion: str
    cohort: Cohort
    target_profile: Path


@dataclass(frozen=True)
class Comparison:
    """Finite target and observed ROI features from public MLXRadiomics output."""

    target: dict[str, float]
    observed: dict[str, float]
    error: dict[str, float]

    def families(self) -> dict[str, float]:
        """Average the existing per-feature Canberra distances within each family."""
        return {
            family: float(
                np.mean(
                    [
                        value
                        for name, value in self.error.items()
                        if name.startswith(f"original_{family}_")
                    ]
                )
            )
            for family in FAMILIES
        }


@dataclass(frozen=True)
class Draw:
    case: Case
    seed: int
    gain: int
    comparisons: dict[Space, Comparison]
    diagnostics: Row


@dataclass(frozen=True)
class Pair:
    case: Case
    seed: int
    disabled: Draw
    trained: Draw


def _bad_constant(value: str) -> NoReturn:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _object(value: object, label: str) -> Row:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    fields = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in fields):
        raise ValueError(f"{label} requires text keys")
    return cast(Row, fields)


def _read(path: Path) -> Row:
    value: object = json.loads(path.read_text(), parse_constant=_bad_constant)
    return _object(value, str(path))


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} requires nonempty text")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{label} requires a finite number")
    return float(value)


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} requires an integer")
    return value


def _rows(value: object, label: str) -> list[Row]:
    if not isinstance(value, list):
        raise ValueError(f"{label} requires an array")
    return [_object(item, label) for item in cast(list[object], value)]


def _resolve(value: object, base: Path, label: str) -> Path:
    path = Path(_text(value, label))
    return path if path.is_absolute() else base / path


def load_comparison(path: Path) -> Comparison:
    """Read finite per-feature measurements without executing a model or extractor."""
    payload = _read(path)
    deviations = _object(payload.get("deviations"), "deviations")
    target: dict[str, float] = {}
    observed: dict[str, float] = {}
    errors: dict[str, float] = {}
    for name, item in deviations.items():
        if not any(name.startswith(f"original_{family}_") for family in FAMILIES):
            raise ValueError(f"Unexpected appearance feature: {name}")
        values = _object(item, name)
        target[name] = _number(values.get("target"), name)
        observed[name] = _number(values.get("generated"), name)
        errors[name] = _number(values.get("canberra"), name)
    if len(target) != 101 or MEAN not in target:
        raise ValueError("Reliability analysis requires all 101 appearance features")
    if any(value < 0 or value > 1 for value in errors.values()):
        raise ValueError("Per-feature Canberra distances must lie in [0, 1]")
    return Comparison(target, observed, errors)


def _stratum(target: float) -> str:
    return "positive" if target > 0 else "negative" if target < 0 else "zero"


def phenotype_metrics(target: float, disabled: float, trained: float) -> Row:
    """Separate movement direction, absolute fidelity and mathematical contrast sign.

    A movement can point toward the target and overshoot far enough to worsen
    absolute error. Correct-sign recovery is likewise distinct from lower error.
    Zero HU is the exact stratification boundary, not a clinical threshold.
    """
    before_error, after_error = disabled - target, trained - target
    before_correct = bool(np.sign(disabled) == np.sign(target))
    after_correct = bool(np.sign(trained) == np.sign(target))
    delta = abs(after_error) - abs(before_error)
    return {
        "target_contrast_hu": target,
        "disabled_contrast_hu": disabled,
        "trained_contrast_hu": trained,
        "disabled_signed_error_hu": before_error,
        "trained_signed_error_hu": after_error,
        "disabled_abs_error_hu": abs(before_error),
        "trained_abs_error_hu": abs(after_error),
        "delta_abs_error_hu": delta,
        "absolute_error_improved": delta < 0,
        "absolute_error_worsened": delta > 0,
        "absolute_error_tied": delta == 0,
        "moved_toward_target": (trained - disabled) * (target - disabled) > 0,
        "overshot_target": before_error * after_error < 0,
        "disabled_correct_sign": before_correct,
        "trained_correct_sign": after_correct,
        "recovered_correct_sign": not before_correct and after_correct,
        "lost_correct_sign": before_correct and not after_correct,
        "moved_toward_but_still_wrong_sign": (
            (trained - disabled) * (target - disabled) > 0 and not after_correct
        ),
        "moved_toward_but_absolute_error_worsened": (
            (trained - disabled) * (target - disabled) > 0 and delta > 0
        ),
    }


def attempt_coverage(path: Path, cases: Mapping[str, Case], seeds: Sequence[int]) -> Row:
    """Count unique attempts without interpreting unmatched starts as valid results.

    An unmatched start can mean an active or failed draw. Resume/cache events
    do not create additional experimental observations.
    """
    if not path.is_file():
        return {
            "attempt_log_available": False,
            "unique_attempted_draws": None,
            "unique_completed_draws": None,
            "uncompleted_attempts": [],
            "patient_attempt_counts": {},
        }
    started: set[tuple[str, int, int]] = set()
    completed: set[tuple[str, int, int]] = set()
    for line in path.read_text().splitlines():
        event = _object(json.loads(line, parse_constant=_bad_constant), "attempt event")
        patient = _text(event.get("patient"), "attempt patient")
        seed = _integer(event.get("seed"), "attempt seed")
        gain = _number(event.get("gain"), "attempt gain")
        if patient not in cases or seed not in seeds or gain not in (0, 1):
            raise ValueError("Attempt event is outside the registered assay")
        key = patient, seed, int(gain)
        status = event.get("status")
        if status == "started":
            started.add(key)
        elif status == "completed":
            completed.add(key)
        else:
            raise ValueError("Unsupported attempt event status")
    if completed - started:
        raise ValueError("Completed attempt event has no corresponding recorded start")
    return {
        "attempt_log_available": True,
        "attempt_log": str(path.resolve()),
        "unique_attempted_draws": len(started),
        "unique_completed_draws": len(completed),
        "patient_attempt_counts": {
            patient: {
                "unique_attempted_draws": sum(key[0] == patient for key in started),
                "unique_completed_draws": sum(key[0] == patient for key in completed),
                "uncompleted_attempts": sum(key[0] == patient for key in started - completed),
            }
            for patient in cases
        },
        "uncompleted_attempts": [
            {"patient": key[0], "seed": key[1], "gain": key[2]}
            for key in sorted(started - completed)
        ],
        "attempt_semantics": (
            "Unmatched starts may be active or failed; inspect runner status/traceback. "
            "Repeated resume/cache events are deduplicated."
        ),
    }


def load_pairs(index_path: Path, allow_partial: bool) -> tuple[list[Pair], Row]:
    """Load only complete gain pairs from an authoritative registered run index."""
    index = _read(index_path)
    if index.get("schema") != "lesioncontrol.reliability_runs.v1":
        raise ValueError("Unsupported reliability run index")
    base = index_path.resolve().parent
    raw_seeds = index.get("seeds")
    if not isinstance(raw_seeds, list):
        raise ValueError("The registered seed panel is required")
    seeds = [_integer(value, "seed") for value in cast(list[object], raw_seeds)]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must form one nonempty unique registered panel")
    cases: dict[str, Case] = {}
    for item in _rows(index.get("cases"), "cases"):
        patient = _text(item.get("patient"), "patient")
        cohort = item.get("cohort")
        if cohort not in ("selection", "heldout", "previously_evaluated"):
            raise ValueError("Every reference requires an explicit registered cohort")
        if patient in cases:
            raise ValueError("This assay supports one reference lesion per patient")
        cases[patient] = Case(
            patient,
            _text(item.get("lesion"), "lesion"),
            cohort,
            _resolve(item.get("target_profile"), base, "target_profile"),
        )
    if not cases:
        raise ValueError("The registered assay requires at least one reference patient")
    draws: dict[tuple[str, int, int], Draw] = {}
    for item in _rows(index.get("draws"), "draws"):
        patient = _text(item.get("patient"), "patient")
        if patient not in cases:
            raise ValueError("A draw refers to an unregistered patient")
        case = cases[patient]
        if item.get("lesion") != case.lesion or item.get("cohort") != case.cohort:
            raise ValueError("Draw lesion or cohort differs from its registered reference")
        seed = _integer(item.get("seed"), "seed")
        gain_value = _number(item.get("gain"), "gain")
        if seed not in seeds or gain_value not in (0, 1):
            raise ValueError("Draw seed or gain is outside the registered paired assay")
        gain = int(gain_value)
        key = patient, seed, gain
        if key in draws:
            raise ValueError(f"Duplicate indexed draw: {key}")
        paths: dict[Space, Path] = {
            "raw": _resolve(item.get("comparison_path"), base, "comparison_path")
        }
        aligned = item.get("mean_aligned_comparison_path")
        if aligned is not None:
            paths["mean_aligned"] = _resolve(aligned, base, "mean_aligned_comparison_path")
        if not all(path.is_file() for path in paths.values()):
            if allow_partial:
                continue
            raise FileNotFoundError(f"Indexed comparison is missing: {key}")
        comparisons: dict[Space, Comparison] = {
            space: load_comparison(path) for space, path in paths.items()
        }
        metrics_path = item.get("metrics_path")
        diagnostics = (
            {} if metrics_path is None else _read(_resolve(metrics_path, base, "metrics_path"))
        )
        draws[key] = Draw(case, seed, gain, comparisons, diagnostics)
    pairs: list[Pair] = []
    missing: list[Row] = []
    for patient, case in cases.items():
        for seed in seeds:
            if (patient, seed, 0) not in draws or (patient, seed, 1) not in draws:
                missing.append({"patient": patient, "seed": seed})
                continue
            before, after = draws[(patient, seed, 0)], draws[(patient, seed, 1)]
            if set(before.comparisons) != set(after.comparisons):
                raise ValueError("A gain pair has unequal raw/mean-aligned measurements")
            for space in before.comparisons:
                if before.comparisons[space].target != after.comparisons[space].target:
                    raise ValueError("A paired draw has different reference targets")
                if (
                    space == "mean_aligned"
                    and before.comparisons[space].target != before.comparisons["raw"].target
                ):
                    raise ValueError("Mean-aligned measurements changed the reference targets")
            pairs.append(Pair(case, seed, before, after))
    coverage = attempt_coverage(base / "attempts.jsonl", cases, seeds)
    if missing and not allow_partial:
        raise ValueError(
            f"Incomplete seed pairs: {len(missing)} of {len(cases) * len(seeds)}; "
            f"unique attempted draws: {coverage['unique_attempted_draws']}"
        )
    for patient in cases:
        targets = [
            pair.disabled.comparisons["raw"].target
            for pair in pairs
            if pair.case.patient == patient
        ]
        if targets and any(target != targets[0] for target in targets[1:]):
            raise ValueError("Reference feature targets changed across seeds")
        if targets:
            features = _object(
                _read(cases[patient].target_profile).get("features"), "profile features"
            )
            expected = {name: _number(value, name) for name, value in features.items()}
            if targets[0] != expected:
                raise ValueError("Comparison target differs from the registered reference profile")
    patient_attempts = _object(coverage["patient_attempt_counts"], "patient attempt counts")
    patient_coverage: list[Row] = []
    for patient, case in cases.items():
        counts = (
            _object(patient_attempts[patient], "patient attempts")
            if patient in patient_attempts
            else {}
        )
        patient_coverage.append(
            {
                "patient": patient,
                "lesion": case.lesion,
                "cohort": case.cohort,
                "planned_pairs": len(seeds),
                "planned_draws": 2 * len(seeds),
                "valid_pairs": sum(pair.case.patient == patient for pair in pairs),
                "unique_attempted_draws": counts.get("unique_attempted_draws"),
                "unique_completed_draws": counts.get("unique_completed_draws"),
                "uncompleted_attempts": counts.get("uncompleted_attempts"),
            }
        )
    return pairs, coverage | {
        "source_index": str(index_path.resolve()),
        "planned_seeds": seeds,
        "planned_reference_patients": len(cases),
        "planned_pairs": len(cases) * len(seeds),
        "planned_draws": 2 * len(cases) * len(seeds),
        "valid_indexed_draws": len(draws),
        "complete_pairs": len(pairs),
        "missing_pairs": missing,
        "complete": not missing,
        "allow_partial": allow_partial,
        "patient_coverage": patient_coverage,
    }


def _write_csv(path: Path, rows: Sequence[Row]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _interval(values: Floats, rng: np.random.Generator) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    result = stats.bootstrap(
        (values,),
        np.mean,
        method="percentile",
        confidence_level=0.95,
        n_resamples=9999,
        batch=256,
        rng=rng,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def _base(pair: Pair) -> Row:
    return {
        "cohort": pair.case.cohort,
        "patient": pair.case.patient,
        "lesion": pair.case.lesion,
        "seed": pair.seed,
        "contrast_stratum": _stratum(pair.disabled.comparisons["raw"].target[MEAN]),
    }


def metric_rows(pair: Pair) -> tuple[list[Row], list[Row], Row]:
    """Expose paired feature/family errors and explicit enhancement separately."""
    before_raw, after_raw = pair.disabled.comparisons["raw"], pair.trained.comparisons["raw"]
    phenotype = _base(pair) | phenotype_metrics(
        before_raw.target[MEAN], before_raw.observed[MEAN], after_raw.observed[MEAN]
    )
    for label, draw in (("disabled", pair.disabled), ("trained", pair.trained)):
        for name in ("actual_generated_contrast_hu", "generated_ring_shift_hu"):
            if name in draw.diagnostics:
                phenotype[label + "_" + name] = _number(draw.diagnostics[name], name)
        if "actual_generated_contrast_hu" in draw.diagnostics:
            actual = _number(draw.diagnostics["actual_generated_contrast_hu"], "actual contrast")
            phenotype[label + "_actual_ring_abs_error_hu"] = abs(actual - before_raw.target[MEAN])
    metrics: list[Row] = [
        _base(pair)
        | {
            "space": "raw",
            "metric": "contrast_abs_hu",
            "disabled_error": phenotype["disabled_abs_error_hu"],
            "trained_error": phenotype["trained_abs_error_hu"],
            "delta_error": phenotype["delta_abs_error_hu"],
        }
    ]
    features: list[Row] = []
    for space in pair.disabled.comparisons:
        before, after = pair.disabled.comparisons[space], pair.trained.comparisons[space]
        before_families, after_families = before.families(), after.families()
        for family in (*FAMILIES, "texture", "all_families"):
            if family in FAMILIES:
                a, b = before_families[family], after_families[family]
            else:
                wanted = FAMILIES if family == "all_families" else FAMILIES[1:]
                a, b = (
                    _mean([before_families[key] for key in wanted]),
                    _mean([after_families[key] for key in wanted]),
                )
            metrics.append(
                _base(pair)
                | {
                    "space": space,
                    "metric": family,
                    "disabled_error": a,
                    "trained_error": b,
                    "delta_error": b - a,
                }
            )
        for name in sorted(before.target):
            features.append(
                _base(pair)
                | {
                    "space": space,
                    "family": name.split("_", 2)[1],
                    "feature": name,
                    "target": before.target[name],
                    "disabled": before.observed[name],
                    "trained": after.observed[name],
                    "disabled_raw_abs_error": abs(before.observed[name] - before.target[name]),
                    "trained_raw_abs_error": abs(after.observed[name] - after.target[name]),
                    "disabled_error": before.error[name],
                    "trained_error": after.error[name],
                    "delta_error": after.error[name] - before.error[name],
                }
            )
    return metrics, features, phenotype


def _paired_summary(items: Sequence[Row]) -> Row:
    before = np.asarray([_number(item["disabled_error"], "disabled_error") for item in items])
    after = np.asarray([_number(item["trained_error"], "trained_error") for item in items])
    delta = after - before
    wins, losses, ties = int(np.sum(delta < 0)), int(np.sum(delta > 0)), int(np.sum(delta == 0))
    ci = stats.binomtest(wins, len(items)).proportion_ci(method="wilson")
    return {
        "paired_seeds": len(items),
        "disabled_mean_error": float(before.mean()),
        "trained_mean_error": float(after.mean()),
        "mean_delta_error": float(delta.mean()),
        "median_delta_error": float(np.median(delta)),
        "delta_p05": float(np.percentile(delta, 5)),
        "delta_p95": float(np.percentile(delta, 95)),
        "improved_seeds": wins,
        "worsened_seeds": losses,
        "tied_seeds": ties,
        "improvement_fraction": wins / len(items),
        "improvement_fraction_wilson_low": float(ci.low),
        "improvement_fraction_wilson_high": float(ci.high),
    }


def summarize_metrics(
    rows: Sequence[Row], feature: bool = False, planned_seeds: int | None = None
) -> tuple[list[Row], list[Row]]:
    """Keep all seed pairs within patient, then summarize patient means by cohort."""
    keys = (
        "cohort",
        "patient",
        "lesion",
        "contrast_stratum",
        "space",
        "feature" if feature else "metric",
    )
    groups: dict[tuple[str, ...], list[Row]] = defaultdict(list)
    for row in rows:
        groups[tuple(_text(row[key], key) for key in keys)].append(row)
    patients: list[Row] = []
    for key, values in sorted(groups.items()):
        row = dict(zip(keys, key, strict=True)) | _paired_summary(values)
        row["planned_seeds"] = planned_seeds
        row["confirmed_improvements_per_planned_seed"] = (
            _number(row["improved_seeds"], "improved seeds") / planned_seeds
            if planned_seeds is not None
            else None
        )
        if feature:
            name = _text(row["feature"], "feature")
            row["family"] = name.split("_", 2)[1]
            row["structural_redundancy"] = {
                "original_glcm_DifferenceAverage": "GLCM Dissimilarity alias",
                "original_glcm_SumVariance": "GLCM ClusterTendency alias",
                "original_firstorder_TotalEnergy": "Energy-equivalent at unit voxel volume",
                "original_gldm_DependencePercentage": "Constant one in this extraction contract",
            }.get(name, "")
            for condition in ("disabled", "trained"):
                row[condition + "_mean_raw_abs_error"] = _mean(
                    [
                        _number(value[condition + "_raw_abs_error"], "raw feature error")
                        for value in values
                    ]
                )
                row[condition + "_mean_signed_error"] = _mean(
                    [
                        _number(value[condition], "observed feature")
                        - _number(value["target"], "target feature")
                        for value in values
                    ]
                )
        patients.append(row)
    cohort_groups: dict[tuple[str, str, str, str], list[Row]] = defaultdict(list)
    metric_key = "feature" if feature else "metric"
    for row in patients:
        for stratum in ("all", _text(row["contrast_stratum"], "contrast_stratum")):
            cohort_groups[
                (
                    _text(row["cohort"], "cohort"),
                    stratum,
                    _text(row["space"], "space"),
                    _text(row[metric_key], metric_key),
                )
            ].append(row)
    cohorts: list[Row] = []
    rng = np.random.default_rng(20261109)
    for key, values in sorted(cohort_groups.items()):
        delta = np.asarray([_number(row["mean_delta_error"], "mean_delta_error") for row in values])
        low, high = (None, None) if feature else _interval(delta, rng)
        cohorts.append(
            {
                "cohort": key[0],
                "contrast_stratum": key[1],
                "space": key[2],
                metric_key: key[3],
                "reference_patients": len(values),
                "disabled_mean_error": _mean(
                    [_number(row["disabled_mean_error"], "before") for row in values]
                ),
                "trained_mean_error": _mean(
                    [_number(row["trained_mean_error"], "after") for row in values]
                ),
                "mean_delta_error": float(delta.mean()),
                "patient_mean_delta_ci_low": low,
                "patient_mean_delta_ci_high": high,
                "patients_improved": int(np.sum(delta < 0)),
                "patients_worsened": int(np.sum(delta > 0)),
                "patients_tied": int(np.sum(delta == 0)),
                "mean_within_patient_improvement_fraction": _mean(
                    [_number(row["improvement_fraction"], "win fraction") for row in values]
                ),
                "interval_scope": "Not computed for descriptive feature grid"
                if feature
                else (
                    "Patient-cluster percentile bootstrap, conditional on fixed seed panel "
                    "and host setting; no interval for fewer than two patients"
                ),
            }
        )
        if feature:
            row = cohorts[-1]
            row["family"] = key[3].split("_", 2)[1]
            row["structural_redundancy"] = values[0]["structural_redundancy"]
            for condition in ("disabled", "trained"):
                for field in ("mean_raw_abs_error", "mean_signed_error"):
                    row[condition + "_" + field] = _mean(
                        [_number(value[condition + "_" + field], field) for value in values]
                    )
    return patients, cohorts


def phenotype_summary(rows: Sequence[Row]) -> list[Row]:
    groups: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in rows:
        groups[(_text(row["cohort"], "cohort"), _text(row["patient"], "patient"))].append(row)
    result: list[Row] = []
    flags = (
        "absolute_error_improved",
        "absolute_error_worsened",
        "absolute_error_tied",
        "moved_toward_target",
        "overshot_target",
        "disabled_correct_sign",
        "trained_correct_sign",
        "recovered_correct_sign",
        "lost_correct_sign",
        "moved_toward_but_still_wrong_sign",
        "moved_toward_but_absolute_error_worsened",
    )
    for (cohort, patient), values in sorted(groups.items()):
        record: Row = {
            "cohort": cohort,
            "patient": patient,
            "contrast_stratum": values[0]["contrast_stratum"],
            "reference_contrast_hu": values[0]["target_contrast_hu"],
            "paired_seeds": len(values),
        }
        for flag in flags:
            count = sum(row[flag] is True for row in values)
            record[flag + "_count"] = count
            record[flag + "_fraction"] = count / len(values)
        for condition in ("disabled", "trained"):
            record[condition + "_mean_signed_error_hu"] = _mean(
                [_number(row[condition + "_signed_error_hu"], "signed error") for row in values]
            )
            for suffix in (
                "actual_generated_contrast_hu",
                "generated_ring_shift_hu",
                "actual_ring_abs_error_hu",
            ):
                field = condition + "_" + suffix
                present = [field in row for row in values]
                if any(present) and not all(present):
                    raise ValueError("Actual-ring diagnostics are incomplete within a patient")
                if all(present):
                    record[condition + "_mean_" + suffix] = _mean(
                        [_number(row[field], field) for row in values]
                    )
        result.append(record)
    return result


def cohort_phenotypes(patients: Sequence[Row]) -> list[Row]:
    """Average sign-recovery and direction fractions over reference patients."""
    groups: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in patients:
        for stratum in ("all", _text(row["contrast_stratum"], "contrast_stratum")):
            groups[(_text(row["cohort"], "cohort"), stratum)].append(row)
    result: list[Row] = []
    for (cohort, stratum), values in sorted(groups.items()):
        row: Row = {
            "cohort": cohort,
            "contrast_stratum": stratum,
            "reference_patients": len(values),
            "paired_seed_counts": [value["paired_seeds"] for value in values],
        }
        for field in values[0]:
            if field.endswith("_fraction") or field.endswith("_mean_signed_error_hu"):
                row[field] = _mean([_number(value[field], field) for value in values])
        result.append(row)
    return result


def joint_reliability(pairs: Sequence[Pair]) -> tuple[list[Row], list[Row], list[Row]]:
    """Measure simultaneous family improvement with strict and tie-tolerant definitions."""
    flags = (
        "contrast_and_all_six_strict",
        "contrast_and_all_six_nonworsening",
        "contrast_and_all_five_texture_strict",
        "contrast_and_all_five_texture_nonworsening",
    )
    observations: list[Row] = []
    for pair in pairs:
        a, b = pair.disabled.comparisons["raw"], pair.trained.comparisons["raw"]
        contrast_better = abs(b.observed[MEAN] - b.target[MEAN]) < abs(
            a.observed[MEAN] - a.target[MEAN]
        )
        for space in pair.disabled.comparisons:
            before, after = (
                pair.disabled.comparisons[space].families(),
                pair.trained.comparisons[space].families(),
            )
            delta = {name: after[name] - before[name] for name in FAMILIES}
            observations.append(
                _base(pair)
                | {
                    "space": space,
                    "contrast_and_all_six_strict": contrast_better
                    and all(value < 0 for value in delta.values()),
                    "contrast_and_all_six_nonworsening": contrast_better
                    and all(value <= 0 for value in delta.values()),
                    "contrast_and_all_five_texture_strict": contrast_better
                    and all(delta[name] < 0 for name in FAMILIES[1:]),
                    "contrast_and_all_five_texture_nonworsening": contrast_better
                    and all(delta[name] <= 0 for name in FAMILIES[1:]),
                    "improved_families": sum(value < 0 for value in delta.values()),
                    "worsened_families": sum(value > 0 for value in delta.values()),
                    "tied_families": sum(value == 0 for value in delta.values()),
                }
            )
    groups: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for row in observations:
        key = (
            _text(row["cohort"], "cohort"),
            _text(row["patient"], "patient"),
            _text(row["space"], "space"),
        )
        groups[key].append(row)
    patients: list[Row] = []
    for key, values in sorted(groups.items()):
        row: Row = {
            "cohort": key[0],
            "patient": key[1],
            "space": key[2],
            "contrast_stratum": values[0]["contrast_stratum"],
            "paired_seeds": len(values),
        }
        for flag in flags:
            count = sum(value[flag] is True for value in values)
            interval = stats.binomtest(count, len(values)).proportion_ci(method="wilson")
            row[flag + "_count"] = count
            row[flag + "_fraction"] = count / len(values)
            row[flag + "_wilson_low"] = float(interval.low)
            row[flag + "_wilson_high"] = float(interval.high)
        patients.append(row)
    cohort_groups: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for row in patients:
        for stratum in ("all", _text(row["contrast_stratum"], "contrast_stratum")):
            cohort_groups[
                (_text(row["cohort"], "cohort"), stratum, _text(row["space"], "space"))
            ].append(row)
    cohorts: list[Row] = []
    for key, values in sorted(cohort_groups.items()):
        row = {
            "cohort": key[0],
            "contrast_stratum": key[1],
            "space": key[2],
            "reference_patients": len(values),
        }
        for flag in flags:
            row[flag + "_mean_patient_fraction"] = _mean(
                [_number(value[flag + "_fraction"], flag) for value in values]
            )
        cohorts.append(row)
    return observations, patients, cohorts


def tradeoff_rows(metrics: Sequence[Row], phenotypes: Sequence[Row]) -> list[Row]:
    contrast = {
        (_text(row["patient"], "patient"), _integer(row["seed"], "seed")): _number(
            row["delta_abs_error_hu"], "contrast delta"
        )
        for row in phenotypes
    }
    groups: dict[tuple[str, str, str, str], list[Row]] = defaultdict(list)
    for row in metrics:
        if row["metric"] in (*FAMILIES, "texture"):
            key = (
                _text(row["cohort"], "cohort"),
                _text(row["patient"], "patient"),
                _text(row["space"], "space"),
                _text(row["metric"], "metric"),
            )
            groups[key].append(row)
    result: list[Row] = []
    for key, values in sorted(groups.items()):
        x = np.asarray(
            [
                contrast[(_text(row["patient"], "patient"), _integer(row["seed"], "seed"))]
                for row in values
            ]
        )
        y = np.asarray([_number(row["delta_error"], "family delta") for row in values])
        correlation: float | None = None
        if len(values) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
            correlation = float(stats.spearmanr(x, y).statistic)
        result.append(
            {
                "cohort": key[0],
                "patient": key[1],
                "space": key[2],
                "metric": key[3],
                "paired_seeds": len(values),
                "both_improved": int(np.sum((x < 0) & (y < 0))),
                "contrast_better_family_worse": int(np.sum((x < 0) & (y > 0))),
                "contrast_worse_family_better": int(np.sum((x > 0) & (y < 0))),
                "both_worsened": int(np.sum((x > 0) & (y > 0))),
                "ties_in_either": int(np.sum((x == 0) | (y == 0))),
                "within_patient_spearman_delta_association": correlation,
            }
        )
    return result


def analyze(index: Path, output: Path, allow_partial: bool = False) -> Row:
    """Write paired raw data and reference-level fidelity tables from existing outputs."""
    started = time.perf_counter()
    pairs, completeness = load_pairs(index, allow_partial)
    if not pairs:
        if allow_partial:
            output.mkdir(parents=True, exist_ok=True)
            _write_csv(
                output / "patient_coverage.csv", cast(list[Row], completeness["patient_coverage"])
            )
            summary = completeness | {
                "analysis_status": "No complete valid pairs; no effect estimated"
            }
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n"
            )
            return summary
        raise ValueError("No complete paired model outputs are available")
    metrics: list[Row] = []
    features: list[Row] = []
    phenotypes: list[Row] = []
    for pair in pairs:
        m, f, p = metric_rows(pair)
        metrics.extend(m)
        features.extend(f)
        phenotypes.append(p)
    planned_seeds = len(cast(list[int], completeness["planned_seeds"]))
    patient_metrics, cohort_metrics = summarize_metrics(metrics, planned_seeds=planned_seeds)
    patient_features, cohort_features = summarize_metrics(
        features, feature=True, planned_seeds=planned_seeds
    )
    phenotype_rows = phenotype_summary(phenotypes)
    paired_joint, patient_joint, cohort_joint = joint_reliability(pairs)
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("paired_metrics.csv", metrics),
        ("patient_coverage.csv", cast(list[Row], completeness["patient_coverage"])),
        ("paired_features.csv", features),
        ("paired_phenotypes.csv", phenotypes),
        ("patient_metrics.csv", patient_metrics),
        ("cohort_metrics.csv", cohort_metrics),
        ("patient_features.csv", patient_features),
        ("cohort_features.csv", cohort_features),
        ("patient_phenotypes.csv", phenotype_rows),
        ("cohort_phenotypes.csv", cohort_phenotypes(phenotype_rows)),
        ("patient_tradeoffs.csv", tradeoff_rows(metrics, phenotypes)),
        ("paired_joint_reliability.csv", paired_joint),
        ("patient_joint_reliability.csv", patient_joint),
        ("cohort_joint_reliability.csv", cohort_joint),
    ):
        _write_csv(output / name, values)
    summary = completeness | {
        "analysis_unit": "Reference patient; seeds are paired technical repetitions",
        "effect_direction": "Trained minus disabled error; negative is better",
        "contrast_boundary": "Exact zero HU mathematical sign; no clinical phenotype threshold",
        "cohort_intervals": (
            "SciPy patient-cluster percentile bootstrap; 9999 resamples, seed 20261109; "
            "conditional on observed seed panel and registered host setting; "
            "no interval when n<2"
        ),
        "seed_intervals": (
            "Wilson interval for conditional improvement fraction; empirical 5th–95th "
            "percentiles for paired error changes; not independent-patient uncertainty"
        ),
        "multiplicity": "No hypothesis tests or unadjusted per-feature significance claims",
        "mean_aligned_scope": (
            "Uniform generated-image shift to reference ROI mean diagnoses attenuation/"
            "bin-origin sensitivity; it does not isolate pure spatial texture or "
            "establish causality"
        ),
        "cohort_metrics": cohort_metrics,
        "patient_phenotypes": phenotype_rows,
        "cohort_phenotypes": cohort_phenotypes(phenotype_rows),
        "cohort_joint_reliability": cohort_joint,
        "elapsed_cpu_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("complete", "complete_pairs", "planned_pairs", "elapsed_cpu_seconds")
            }
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    analyze(args.index, args.output, args.allow_partial)


if __name__ == "__main__":
    main()
