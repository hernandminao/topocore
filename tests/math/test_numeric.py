"""
Tests for topocore.math.numeric.
"""

import pytest

from topocore.core.exceptions import MathError
from topocore.math.numeric import (
    clamp,
    lerp,
    mean,
    safe_divide,
)


@pytest.mark.parametrize(
    ("value", "minimum", "maximum", "expected"),
    [
        (5.0, 0.0, 10.0, 5.0),
        (-5.0, 0.0, 10.0, 0.0),
        (20.0, 0.0, 10.0, 10.0),
    ],
)
def test_clamp(
    value: float,
    minimum: float,
    maximum: float,
    expected: float,
) -> None:
    assert clamp(value, minimum, maximum) == expected


def test_clamp_invalid_interval() -> None:
    with pytest.raises(MathError):
        clamp(5.0, 10.0, 0.0)


@pytest.mark.parametrize(
    ("start", "end", "t", "expected"),
    [
        (0.0, 10.0, 0.5, 5.0),
        (10.0, 20.0, 0.0, 10.0),
        (10.0, 20.0, 1.0, 20.0),
        (10.0, 20.0, 2.0, 30.0),
    ],
)
def test_lerp(
    start: float,
    end: float,
    t: float,
    expected: float,
) -> None:
    assert lerp(start, end, t) == expected


def test_safe_divide() -> None:
    assert safe_divide(20.0, 4.0) == 5.0


def test_safe_divide_zero() -> None:
    with pytest.raises(MathError):
        safe_divide(10.0, 0.0)


def test_mean() -> None:
    assert mean([2.0, 4.0, 6.0]) == 4.0


def test_mean_tuple() -> None:
    assert mean((1.0, 2.0, 3.0)) == 2.0


def test_mean_empty() -> None:
    with pytest.raises(MathError):
        mean([])
