"""Typed source pins and label vocabulary from the packaged reference inventory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal, TypedDict, cast


class SourceReference(TypedDict):
    """Contents of ``reference.json``: pinned upstream repositories, revisions and file list.

    ``files`` are paths relative to the repository root, except ``open_clip/`` entries,
    which come from ``text_repository`` under its ``src/`` directory. ``organ_labels``
    maps source label ids (as strings) to organ names.
    """

    schema_version: Literal[1]
    repository: str
    revision: str
    text_repository: str
    text_revision: str
    files: list[str]
    organ_labels: dict[str, str]


def _string(value: object) -> str:
    """Require a string for a reference metadata field."""
    if not isinstance(value, str):
        raise ValueError("reference.json source names, revisions and labels must be strings")
    return value


def _load_reference() -> SourceReference:
    """Validate the packaged JSON boundary once; consumers receive concrete types."""
    raw: object = json.loads(Path(__file__).with_name("reference.json").read_text())
    if not isinstance(raw, dict):
        raise ValueError("reference.json must contain an object")
    values = cast(dict[str, object], raw)
    if values["schema_version"] != 1:
        raise ValueError("Unsupported reference.json schema version")
    files, labels = values["files"], values["organ_labels"]
    if not isinstance(files, list) or not isinstance(labels, dict):
        raise ValueError("reference.json requires a file list and an organ label mapping")
    return SourceReference(
        schema_version=1,
        repository=_string(values["repository"]),
        revision=_string(values["revision"]),
        text_repository=_string(values["text_repository"]),
        text_revision=_string(values["text_revision"]),
        files=[_string(value) for value in cast(list[object], files)],
        organ_labels={
            _string(key): _string(value) for key, value in cast(dict[str, object], labels).items()
        },
    )


REFERENCE: Final[SourceReference] = _load_reference()
