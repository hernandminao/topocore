# mypy: ignore-errors
"""
Tests for topocore.math.validation.
"""

import math

import pytest

from topocore.core.exceptions import MathError
from topocore.math.validation import (
    validate_coordinate,
    validate_finite,
    validate_not_nan,
    validate_positive,
)


def test_validate_not_nan() -> None:
    validate_not_nan(1.0)


def test_validate_not_nan_nan() -> None:
    with pytest.raises(MathError):
        validate_not_nan(math.nan)


def test_validate_finite() -> None:
    validate_finite(10.0)


def test_validate_finite_inf() -> None:
    with pytest.raises(MathError):
        validate_finite(math.inf)


def test_validate_coordinate() -> None:
    validate_coordinate(100.0)


def test_validate_coordinate_large() -> None:
    with pytest.raises(MathError):
        validate_coordinate(1e20)


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        10.0,
        100.0,
    ],
)
def test_validate_positive(value: float) -> None:
    validate_positive(value)


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -1.0,
        -100.0,
    ],
)
def test_validate_positive_invalid(value: float) -> None:
    with pytest.raises(MathError):
        validate_positive(value)
