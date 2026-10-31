"""Pinned upstream uniform DDIM timestep schedule, with explicit invalid-index errors."""

from __future__ import annotations


def _is_runtime_integer(value: object) -> bool:
    """Recognize integer inputs while excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def lesion_diffusion_timesteps(steps: int, total: int = 1000) -> tuple[int, ...]:
    """Return the pinned upstream uniform DDIM timestep indices, validated.

    The indices are ``value + 1`` for ``value`` in ``range(0, total, total // steps)``,
    in ascending order; the sampler walks them in reverse. The tuple has at least
    ``steps`` entries and has more when ``steps`` does not divide ``total`` (7 entries
    for ``steps=6``), so callers report ``len(...)`` as the actual step count. An
    out-of-range index is rejected rather than clipped, to keep the schedule upstream's.

    Args:
        steps: Requested number of DDIM steps, a positive integer (not a bool).
        total: Length of the training noise schedule (``alphas_cumprod``).

    Raises:
        ValueError: If ``steps`` or ``total`` is not an integer, ``steps < 1``,
            ``total < steps``, or the last index would reach ``total`` (for example
            ``steps=3`` or ``steps=1000`` with ``total=1000``).
    """
    if not _is_runtime_integer(steps) or steps < 1:
        raise ValueError("LesionDiffusion DDIM steps must be a positive integer")
    if not _is_runtime_integer(total) or total < steps:
        raise ValueError("LesionDiffusion DDIM schedule must contain at least as many timesteps")
    timesteps = tuple(value + 1 for value in range(0, total, total // steps))
    if timesteps[-1] >= total:
        raise ValueError(
            f"LesionDiffusion DDIM steps {steps} produce out-of-bounds indices "
            f"for a {total}-timestep schedule; choose another diffusion_steps value"
        )
    return timesteps
