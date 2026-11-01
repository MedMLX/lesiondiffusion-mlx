"""Probe native runtime in a child process so unavailable Metal cannot abort pytest."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Mlx
import subprocess
import sys

import pytest


@pytest.fixture(scope="session")
def mx() -> Mlx:
    """Provide the native runtime when the Metal subprocess probe succeeds."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mlx.core; from medmlx_core.runtime import import_mlx; import_mlx()",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        if "No Metal device available" in completed.stderr:
            pytest.skip("Native MLX runtime is unavailable: No Metal device available")
        pytest.fail("Native runtime probe failed unexpectedly:\n" + completed.stderr)
    from medmlx_core.runtime import import_mlx

    return cast("Mlx", import_mlx())
