# mypy: ignore-errors
"""
Tests for topocore.math.config.
"""

from dataclasses import FrozenInstanceError

import pytest

from topocore.core.exceptions import MathError
from topocore.math.config import DEFAULT_MATH_CONFIG, MathConfig


def test_default_configuration() -> None:
    config = DEFAULT_MATH_CONFIG

    assert config.absolute_tolerance > 0.0
    assert config.relative_tolerance > 0.0
    assert config.decimal_precision >= 0
    assert config.max_iterations > 0


def test_custom_configuration() -> None:
    config = MathConfig(
        absolute_tolerance=1e-8,
        relative_tolerance=1e-10,
        decimal_precision=6,
        max_iterations=500,
    )

    assert config.absolute_tolerance == 1e-8
    assert config.relative_tolerance == 1e-10
    assert config.decimal_precision == 6
    assert config.max_iterations == 500


@pytest.mark.parametrize(
    "kwargs",
    [
        {"absolute_tolerance": 0.0},
        {"absolute_tolerance": -1.0},
        {"relative_tolerance": 0.0},
        {"relative_tolerance": -1.0},
        {"decimal_precision": -1},
        {"max_iterations": 0},
        {"max_iterations": -10},
    ],
)
def test_invalid_configuration(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(MathError):
        MathConfig(**kwargs)


def test_configuration_is_immutable() -> None:
    config = MathConfig()

    with pytest.raises(FrozenInstanceError):
        config.max_iterations = 2000
