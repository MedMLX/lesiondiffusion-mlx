"""Native MLX implementation of LesionDiffusion, with lazy public imports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lesiondiffusion_mlx._schedules import (
        lesion_diffusion_timesteps as lesion_diffusion_timesteps,
    )
    from lesiondiffusion_mlx.mlx_model import (
        LINET_CONFIG as LINET_CONFIG,
    )
    from lesiondiffusion_mlx.mlx_model import (
        LMNET_CONFIG as LMNET_CONFIG,
    )
    from lesiondiffusion_mlx.mlx_model import (
        BiomedTextEncoder as BiomedTextEncoder,
    )
    from lesiondiffusion_mlx.mlx_model import (
        LdmUnet as LdmUnet,
    )
    from lesiondiffusion_mlx.mlx_model import (
        SourceLayers as SourceLayers,
    )
    from lesiondiffusion_mlx.mlx_model import (
        UnetConfig as UnetConfig,
    )
    from lesiondiffusion_mlx.mlx_model import (
        VqAutoencoder as VqAutoencoder,
    )
    from lesiondiffusion_mlx.mlx_sampling import DdimSampler as DdimSampler

__version__ = "0.2.5"

__all__ = [
    "LINET_CONFIG",
    "LMNET_CONFIG",
    "BiomedTextEncoder",
    "DdimSampler",
    "LdmUnet",
    "SourceLayers",
    "UnetConfig",
    "VqAutoencoder",
    "lesion_diffusion_timesteps",
]


def __getattr__(name: str) -> object:
    """Load a public export from its module on first access."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = (
        "_schedules"
        if name == "lesion_diffusion_timesteps"
        else "mlx_sampling"
        if name == "DdimSampler"
        else "mlx_model"
    )
    value = getattr(import_module(f"lesiondiffusion_mlx.{module}"), name)
    globals()[name] = value
    return value
