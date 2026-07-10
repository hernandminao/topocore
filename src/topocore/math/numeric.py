"""
topocore.math.numeric
=====================

General-purpose numerical utility functions used across TopoCore.

The functions in this module are pure and have no side effects.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Sequence

from topocore.core.exceptions import MathError
from topocore.math.tolerance import is_zero


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Clamp a value to the inclusive interval [minimum, maximum].

    Parameters
    ----------
    value
        Value to clamp.

    minimum
        Lower bound.

    maximum
        Upper bound.

    Returns
    -------
    float

    Raises
    ------
    MathError
        If minimum is greater than maximum.
    """
    if minimum > maximum:
        raise MathError("minimum cannot be greater than maximum.")

    return max(minimum, min(value, maximum))


def lerp(
    start: float,
    end: float,
    t: float,
) -> float:
    """
    Linear interpolation.

    Notes
    -----
    This function also supports extrapolation.
    Values of ``t`` outside the interval [0, 1] are valid.

    Parameters
    ----------
    start
        Initial value.

    end
        Final value.

    t
        Interpolation factor.

    Returns
    -------
    float
    """
    return start + (end - start) * t


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Divide two numbers safely.

    Parameters
    ----------
    numerator
        Dividend.

    denominator
        Divisor.

    Returns
    -------
    float

    Raises
    ------
    MathError
        If the denominator is numerically zero.
    """
    if is_zero(denominator):
        raise MathError("Division by zero.")

    return numerator / denominator


def mean(values: Sequence[float]) -> float:
    """
    Compute the arithmetic mean.

    Parameters
    ----------
    values
        Sequence of numeric values.

    Returns
    -------
    float

    Raises
    ------
    MathError
        If the sequence is empty.
    """
    if not values:
        raise MathError("Cannot compute the mean of an empty sequence.")

    return sum(values) / len(values)


def cube(x: float) -> float:
    return x * x * x


def square(x: float) -> float:
    return x * x


def sign(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


__all__ = [
    "clamp",
    "lerp",
    "mean",
    "safe_divide",
]
