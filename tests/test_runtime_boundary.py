"""Package discovery and native execution do not import reference frameworks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Mlx

import subprocess
import sys
import textwrap


def test_import_is_framework_free() -> None:
    """Keep package import free of native and reference frameworks."""
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys
        class BlockFrameworks(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'mlx', 'torch', 'monai'}:
                    raise AssertionError('Unexpected import: ' + fullname)
        sys.meta_path.insert(0, BlockFrameworks())
        import lesiondiffusion_mlx
        from lesiondiffusion_mlx import medmlx
        assert medmlx.MODEL['run'] == 'lesiondiffusion_mlx.medmlx:run'
        assert not {'mlx', 'torch', 'monai'} & sys.modules.keys()
        """
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_native_inference_has_no_reference_framework_imports(mx: Mlx) -> None:
    """Keep reference frameworks out of native inference imports."""
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys
        class BlockFrameworks(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'torch', 'monai'}:
                    raise AssertionError('Unexpected import: ' + fullname)
        sys.meta_path.insert(0, BlockFrameworks())
        from typing import cast
        from medmlx_core.runtime import import_mlx
        from lesiondiffusion_mlx import LdmUnet, UnetConfig
        from lesiondiffusion_mlx._typing import ArrayConstructor
        from tests._fixture_models import unet_weights
        from tests.test_equivalence import assert_numerical, reference
        mx = import_mlx()
        fixture = reference('unet_mask')
        model = LdmUnet(
            unet_weights(101, mx, mx.float32, skip_context=False),
            mx,
            UnetConfig(32, 1, (1, 2), (1, 2), 16),
        )
        actual = model(
            cast(ArrayConstructor, mx.array)(fixture['x']),
            cast(ArrayConstructor, mx.array)(fixture['timesteps']),
            cast(ArrayConstructor, mx.array)(fixture['context']),
        )
        assert_numerical(actual, fixture['output'])
        """
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
