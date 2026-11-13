"""Record pinned upstream box placement on HCC-TACE liver mask crops."""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure

from tests._runner_cases import CENTER_CASES, organ_shape

REVISION = "880736e2b8a4d9becdb6594da7f8af6db19d04d1"


def main() -> None:
    """Run the source placement and box functions without importing its services."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / ".cache/medmlx/test-data/lesiondiffusion-source",
    )
    parser.add_argument("--output", type=Path, default=Path("work/real-reference"))
    args = parser.parse_args()
    revision = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if revision != REVISION:
        raise ValueError(f"Expected upstream {REVISION}, got {revision}")
    subprocess.run(
        [
            "git",
            "-C",
            str(args.source),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "pipeline/preprocess.py",
        ],
        check=True,
        capture_output=True,
    )
    source = args.source / "pipeline/preprocess.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    names = {"generate_random_bbox_from_mask", "create_bbox_mask"}
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in definitions} != names:
        raise ValueError("Pinned upstream placement definitions are missing")
    scope = dict(
        np=np,
        nib=nib,
        os=os,
        binary_erosion=binary_erosion,
        generate_binary_structure=generate_binary_structure,
    )
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), scope)
    placement = scope["generate_random_bbox_from_mask"]
    make_box = scope["create_bbox_mask"]
    phrases = {
        "inside_parenchyma": "within the parenchymal organ",
        "protruding_from_parenchyma": "protruding from the parenchymal organ",
        "within_lumen": "within the lumen of a hollow organ",
        "wall_thickening": "wall thickening",
    }
    output = {
        "upstream_revision": np.asarray(revision),
        "source": np.asarray("HCC-TACE-Seg HCC_063 liver"),
    }
    with tempfile.TemporaryDirectory(prefix="lesion-placement-") as tmp:
        for name, (mode, region, size, requested, seed) in CENTER_CASES.items():
            if requested is not None:
                continue
            mask = organ_shape(region)
            np.random.seed(seed)
            center = placement(
                mask,
                1,
                np.asarray(size),
                phrases[mode],
                "colon" if mode in {"within_lumen", "wall_thickening"} else "liver",
            )
            output[f"center.{name}"] = center
            if all(width <= dim for width, dim in zip(size, mask.shape, strict=True)):
                path = make_box(nib.Nifti1Image(mask, np.eye(4)), center, np.asarray(size), tmp)
                output[f"box.{name}"] = np.asarray(nib.load(path).dataobj, dtype=np.uint8)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "runner_host.npz", **output)
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/darwin-arm64/runner_host.npz"
    with np.load(fixture, allow_pickle=False) as archive:
        retained = {
            key: archive[key] for key in archive.files if not key.startswith(("center.", "box."))
        }
    np.savez_compressed(fixture, **retained)


if __name__ == "__main__":
    main()
