"""Fetch only the pinned upstream source files, outside this repository."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

from lesiondiffusion_mlx._reference import REFERENCE

REPO_ROOT = Path(__file__).resolve().parents[1]


def fetch(output: Path, *, verify_only: bool) -> None:
    """Download missing pinned files into ``output``; existing files are kept as they are.

    With ``verify_only`` nothing is downloaded and a missing file raises ``FileNotFoundError``.
    """
    output = output.resolve()
    if output.is_relative_to(REPO_ROOT):
        raise ValueError("Reference downloads must stay outside this repository")
    for name in REFERENCE["files"]:
        path = output / name
        if not path.exists() and not verify_only:
            text_source = name.startswith("open_clip/")
            repository = REFERENCE["text_repository" if text_source else "repository"]
            revision = REFERENCE["text_revision" if text_source else "revision"]
            remote_path = "src/" + name if text_source else name
            url = f"https://raw.githubusercontent.com/{repository}/{revision}/{remote_path}"
            with urlopen(url, timeout=30) as response:
                contents = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        if not path.is_file():
            raise FileNotFoundError(f"Missing reference source: {path}")
    print(f"Fetched {len(REFERENCE['files'])} pinned upstream source files")


def main() -> None:
    """Parse options and fetch the pinned upstream source checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    fetch(args.output_dir, verify_only=args.verify_only)


if __name__ == "__main__":
    main()
