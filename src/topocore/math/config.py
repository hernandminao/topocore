"""
topocore.math.config
====================

Mathematical configuration objects.

This module defines immutable configuration objects used by the
numerical utilities across TopoCore.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from topocore.core.exceptions import MathError
from topocore.math.constants import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_DECIMAL_PRECISION,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_RELATIVE_TOLERANCE,
)


@dataclass(frozen=True, slots=True)
class MathConfig:
    """
    Immutable mathematical configuration.

    Parameters
    ----------
    absolute_tolerance
        Absolute tolerance for floating-point comparisons.

    relative_tolerance
        Relative tolerance for floating-point comparisons.

    decimal_precision
        Number of decimal digits when exporting values.

    max_iterations
        Maximum iterations allowed by numerical algorithms.
    """

    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE

    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE

    decimal_precision: int = DEFAULT_DECIMAL_PRECISION

    max_iterations: int = DEFAULT_MAX_ITERATIONS

    def __post_init__(self) -> None:
        """
        Validate configuration values.
        """

        if self.absolute_tolerance <= 0.0:
            raise MathError("absolute_tolerance must be greater than zero.")

        if self.relative_tolerance <= 0.0:
            raise MathError("relative_tolerance must be greater than zero.")

        if self.decimal_precision < 0:
            raise MathError("decimal_precision cannot be negative.")

        if self.max_iterations <= 0:
            raise MathError("max_iterations must be greater than zero.")


DEFAULT_MATH_CONFIG = MathConfig()

__all__ = [
    "MathConfig",
    "DEFAULT_MATH_CONFIG",
]
