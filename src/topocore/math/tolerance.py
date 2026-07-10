"""
topocore.math.tolerance
=======================

Floating-point comparison utilities.

This module centralizes every numerical comparison performed
by TopoCore.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from math import isclose

from topocore.math.config import DEFAULT_MATH_CONFIG


def is_close(
    a: float,
    b: float,
) -> bool:
    """
    Returns True if two floating-point values are numerically equal.
    """
    return isclose(
        a,
        b,
        rel_tol=DEFAULT_MATH_CONFIG.relative_tolerance,
        abs_tol=DEFAULT_MATH_CONFIG.absolute_tolerance,
    )


def is_zero(value: float) -> bool:
    """
    Returns True if value is numerically zero.
    """
    return is_close(value, 0.0)


def compare(
    a: float,
    b: float,
) -> int:
    """
    Robust comparison between two floating-point numbers.

    Returns
    -------
    int

        -1 if a < b

         0 if a ≈ b

         1 if a > b
    """

    if is_close(a, b):
        return 0

    return -1 if a < b else 1


def is_positive(value: float) -> bool:
    """
    Returns True if value is strictly positive.
    """
    return compare(value, 0.0) == 1


def is_negative(value: float) -> bool:
    """
    Returns True if value is strictly negative.
    """
    return compare(value, 0.0) == -1


__all__ = [
    "compare",
    "is_close",
    "is_negative",
    "is_positive",
    "is_zero",
]
