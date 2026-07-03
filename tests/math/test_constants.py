"""
Tests for topocore.math.constants.
"""

from typing import Final

from topocore.math.constants import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_DECIMAL_PRECISION,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_RELATIVE_TOLERANCE,
    EPSILON,
    MAX_COORDINATE_MAGNITUDE,
)


def test_absolute_tolerance_is_positive() -> None:
    assert DEFAULT_ABSOLUTE_TOLERANCE > 0.0


def test_relative_tolerance_is_positive() -> None:
    assert DEFAULT_RELATIVE_TOLERANCE > 0.0


def test_epsilon_is_positive() -> None:
    assert EPSILON > 0.0


def test_decimal_precision_is_positive() -> None:
    assert DEFAULT_DECIMAL_PRECISION > 0


def test_max_coordinate_is_positive() -> None:
    assert MAX_COORDINATE_MAGNITUDE > 0.0


def test_default_iterations_is_positive() -> None:
    assert DEFAULT_MAX_ITERATIONS > 0


def test_constants_are_final() -> None:
    """
    This test exists only to verify the imported symbols
    have the expected runtime types.

    The Final typing annotation is enforced by static type
    checkers (mypy, pyright), not at runtime.
    """
    assert isinstance(DEFAULT_ABSOLUTE_TOLERANCE, float)
    assert isinstance(DEFAULT_RELATIVE_TOLERANCE, float)
    assert isinstance(EPSILON, float)
    assert isinstance(DEFAULT_DECIMAL_PRECISION, int)
    assert isinstance(MAX_COORDINATE_MAGNITUDE, float)
    assert isinstance(DEFAULT_MAX_ITERATIONS, int)

    _: Final[float] = DEFAULT_ABSOLUTE_TOLERANCE