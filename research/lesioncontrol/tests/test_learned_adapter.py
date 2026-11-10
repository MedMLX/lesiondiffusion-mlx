"""Checkpoint admission protects radiomic space, scaling and per-token meaning."""

import json
from pathlib import Path

import pytest

from lesioncontrol.learned import CONTEXT_FIELDS, load_adapter
from lesioncontrol.profile import default_settings


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("target_space", "normalized_pixels", "declare absolute_hu or liver_contrast"),
        ("feature_scale", [0.0] * 101, "Feature scales must be positive"),
        ("context_fields", list(reversed(CONTEXT_FIELDS)), "pinned training order"),
    ],
)
def test_checkpoint_rejects_incompatible_feature_contract(
    tmp_path: Path, field: str, value: object, reason: str
) -> None:
    """Reject incompatible semantics before opening any learned weight file."""
    metadata: dict[str, object] = {
        "schema": "lesioncontrol.radiomic_adapter.v1",
        "target_space": "liver_contrast",
        "conditioning_schema": "training",
        "context_fields": list(CONTEXT_FIELDS),
        "token_count": 10,
        "hidden_dim": 64,
        "feature_names": [f"original_{name}" for name in default_settings().features],
        "feature_center": [0.0] * 101,
        "feature_scale": [1.0] * 101,
    }
    metadata[field] = value
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=reason):
        load_adapter(tmp_path)
