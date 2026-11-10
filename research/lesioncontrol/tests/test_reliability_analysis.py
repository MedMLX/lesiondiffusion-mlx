"""Scientific direction/sign distinctions and fail-fast accounting contracts."""

import json
from pathlib import Path

import pytest

from research.analyze_reliability import Case, attempt_coverage, phenotype_metrics


@pytest.mark.parametrize(
    "target,disabled,trained,before_error,after_error,improved,recovered,overshot",
    [
        (10.0, -20.0, -5.0, 30.0, 15.0, True, False, False),
        (10.0, -20.0, 60.0, 30.0, 50.0, False, True, True),
        (-10.0, -8.0, -15.0, 2.0, 5.0, False, False, True),
    ],
)
def test_direction_does_not_imply_lower_error_or_correct_sign(
    target: float,
    disabled: float,
    trained: float,
    before_error: float,
    after_error: float,
    improved: bool,
    recovered: bool,
    overshot: bool,
) -> None:
    result = phenotype_metrics(target, disabled, trained)
    assert result["disabled_abs_error_hu"] == before_error
    assert result["trained_abs_error_hu"] == after_error
    assert result["moved_toward_target"] is True
    assert result["absolute_error_improved"] is improved
    assert result["recovered_correct_sign"] is recovered
    assert result["overshot_target"] is overshot


def test_resume_events_do_not_inflate_attempts_or_hide_an_unfinished_draw(tmp_path: Path) -> None:
    """One cached completed draw and one unmatched start remain two attempts."""
    case = Case("reference", "reference_lesion", "heldout", tmp_path / "reference.json")
    events = [
        {"patient": "reference", "seed": 1000, "gain": 0, "status": "started"},
        {"patient": "reference", "seed": 1000, "gain": 0, "status": "completed"},
        {"patient": "reference", "seed": 1000, "gain": 0, "status": "started"},
        {"patient": "reference", "seed": 1000, "gain": 0, "status": "completed"},
        {"patient": "reference", "seed": 1000, "gain": 1, "status": "started"},
    ]
    log = tmp_path / "attempts.jsonl"
    log.write_text("".join(json.dumps(event) + "\n" for event in events))
    result = attempt_coverage(log, {"reference": case}, [1000])
    assert result["unique_attempted_draws"] == 2
    assert result["unique_completed_draws"] == 1
    assert result["uncompleted_attempts"] == [{"patient": "reference", "seed": 1000, "gain": 1}]
