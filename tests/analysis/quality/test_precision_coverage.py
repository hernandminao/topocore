"""
Targeted coverage suite for topocore.analysis.quality.precision.
PrecisionAnalysis -- PR20 coverage phase.

This module already had a real, documented PR19 bug fix (the
confidence-margin formula and the confidence_level hardcoding, both
noted in the source's own comments). This suite formalizes fresh
verification of that corrected behavior into permanent regressions:
the confidence margin is verified against a manually computed
t.ppf(1 - alpha/2, n-1) * s / sqrt(n) value (not just "some number
close to expected"), and confirms confidence_level genuinely changes
the result (0.99 gives a strictly wider margin than 0.95 for the
same data -- ruling out a silent hardcode regression). Also covers
the 2D deviation formula (sqrt(std_x^2 + std_y^2)), all properties,
and every validation path. No bugs found -- only test coverage was
added.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as scipy_t

from topocore.analysis.exceptions import QualityError
from topocore.analysis.quality.precision import PrecisionAnalysis


def test_precision_3d_std_dev_matches_known_value() -> None:
    analysis = PrecisionAnalysis(confidence_level=0.95)
    observations = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    result = analysis.compute(observations)

    expected_std = float(np.std([-1.0, 0.0, 1.0], ddof=1))
    assert result.std_dev == pytest.approx(expected_std)
    assert result.count == 3


def test_precision_confidence_margin_matches_manual_t_distribution_formula() -> None:
    """The exact PR19 fix, verified against an independently computed t.ppf value."""
    analysis = PrecisionAnalysis(confidence_level=0.95)
    observations = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    result = analysis.compute(observations)

    std = float(np.std([-1.0, 0.0, 1.0], ddof=1))
    n = 3
    t_value = float(scipy_t.ppf(1.0 - 0.05 / 2.0, n - 1))
    expected_margin = t_value * std / np.sqrt(n)

    assert result.confidence_interval[1] == pytest.approx(expected_margin)
    assert result.confidence_interval[0] == pytest.approx(-expected_margin)


def test_precision_2d_deviation_combines_x_and_y() -> None:
    analysis = PrecisionAnalysis()
    observations = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])

    result = analysis.compute(observations)

    std_x = float(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))
    std_y = float(np.std([2.0, 3.0, 4.0, 5.0], ddof=1))
    expected = float(np.sqrt(std_x**2 + std_y**2))
    assert result.std_dev == pytest.approx(expected)


def test_precision_confidence_level_genuinely_affects_margin_not_hardcoded() -> None:
    """
    Confirms the PR19 confidence_level-hardcoding bug stays fixed: a
    higher confidence level must give a strictly wider margin on the
    identical data.
    """
    observations = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    result_95 = PrecisionAnalysis(confidence_level=0.95).compute(observations)
    result_99 = PrecisionAnalysis(confidence_level=0.99).compute(observations)

    assert result_95.confidence_level == pytest.approx(0.95)
    assert result_99.confidence_level == pytest.approx(0.99)
    assert result_99.confidence_interval[1] > result_95.confidence_interval[1]


def test_precision_tolerance_properties() -> None:
    analysis = PrecisionAnalysis(tolerance_x=0.01, tolerance_y=0.02, tolerance_z=0.03)
    assert analysis.tolerance_x == pytest.approx(0.01)
    assert analysis.tolerance_y == pytest.approx(0.02)
    assert analysis.tolerance_z == pytest.approx(0.03)


def test_precision_call_matches_compute() -> None:
    analysis = PrecisionAnalysis()
    observations = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    assert analysis(observations).std_dev == analysis.compute(observations).std_dev


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tolerance_x": -1.0},
        {"tolerance_y": -1.0},
        {"tolerance_z": -1.0},
        {"confidence_level": 0.0},
        {"confidence_level": 1.0},
        {"confidence_level": 1.5},
    ],
)
def test_precision_rejects_invalid_parameters(kwargs: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(QualityError):
        PrecisionAnalysis(**kwargs)


def test_precision_rejects_fewer_than_three_observations() -> None:
    with pytest.raises(QualityError, match="At least 3"):
        PrecisionAnalysis().compute(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_precision_rejects_1d_array() -> None:
    with pytest.raises(QualityError, match="2D array"):
        PrecisionAnalysis().compute(np.array([1.0, 2.0, 3.0]))


def test_precision_rejects_unsupported_column_count() -> None:
    with pytest.raises(QualityError, match="shape \\(n,2\\) or \\(n,3\\)"):
        PrecisionAnalysis().compute(np.zeros((3, 4)))


def test_precision_rejects_nan() -> None:
    with pytest.raises(QualityError, match="NaN or infinite"):
        PrecisionAnalysis().compute(np.array([[1.0, np.nan, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
