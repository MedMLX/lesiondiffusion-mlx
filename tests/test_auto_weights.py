"""Every registered learned model supports the shared automatic-weight contract."""

import importlib
import inspect
import tomllib
from pathlib import Path
from typing import cast


def test_registered_models_download_weights_when_omitted() -> None:
    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    entries = cast(dict[str, str], project["project"]["entry-points"]["medmlx.models"])
    for reference in entries.values():
        module_name, attribute = reference.split(":")
        declaration = cast(dict[str, str], getattr(importlib.import_module(module_name), attribute))
        runner_module, runner_name = declaration["run"].split(":")
        runner = getattr(importlib.import_module(runner_module), runner_name)
        parameter = inspect.signature(runner).parameters["weights_path"]
        assert parameter.default is None, declaration["id"]
