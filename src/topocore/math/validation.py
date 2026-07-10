"""
topocore.math.validation
========================

Validation helpers for numerical values.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from math import isfinite, isnan

from topocore.core.exceptions import MathError
from topocore.math.constants import MAX_COORDINATE_MAGNITUDE
from topocore.math.tolerance import is_positive


def validate_not_nan(value: float) -> None:
    """
    Validate that a value is not NaN.
    """
    if isnan(value):
        raise MathError("Value cannot be NaN.")


def validate_finite(value: float) -> None:
    """
    Validate that a value is finite.
    """
    if not isfinite(value):
        raise MathError("Value must be finite.")


def validate_coordinate(value: float) -> None:
    """
    Validate a coordinate value.
    """

    validate_not_nan(value)
    validate_finite(value)

    if abs(value) > MAX_COORDINATE_MAGNITUDE:
        raise MathError(f"Coordinate magnitude exceeds {MAX_COORDINATE_MAGNITUDE}.")


def validate_positive(value: float) -> None:
    """
    Validate a strictly positive value.
    """

    if not is_positive(value):
        raise MathError("Value must be greater than zero.")


__all__ = [
    "validate_coordinate",
    "validate_finite",
    "validate_not_nan",
    "validate_positive",
]
