"""
Tests for topocore.math.tolerance.
"""

import pytest

from topocore.math.tolerance import (
    compare,
    is_close,
    is_negative,
    is_positive,
    is_zero,
)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (1.0, 1.0, True),
        (0.0, 0.0, True),
        (1.0, 2.0, False),
    ],
)
def test_is_close(a: float, b: float, expected: bool) -> None:
    assert is_close(a, b) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, True),
        (1.0, False),
        (-1.0, False),
    ],
)
def test_is_zero(value: float, expected: bool) -> None:
    assert is_zero(value) is expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (1.0, 2.0, -1),
        (2.0, 1.0, 1),
        (1.0, 1.0, 0),
    ],
)
def test_compare(a: float, b: float, expected: int) -> None:
    assert compare(a, b) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (5.0, True),
        (-5.0, False),
        (0.0, False),
    ],
)
def test_is_positive(value: float, expected: bool) -> None:
    assert is_positive(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-5.0, True),
        (5.0, False),
        (0.0, False),
    ],
)
def test_is_negative(value: float, expected: bool) -> None:
    assert is_negative(value) is expected
