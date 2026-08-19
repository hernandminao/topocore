"""
Regression suite for topocore.analysis.quality.rmse.RMSEAnalysis --
PR19. Verified with known residuals and the invalid-vs-low-quality
distinction: NaN/Inf residuals (genuinely corrupt/undefined data)
are rejected outright, while valid-but-large residuals (real, if
poor-quality, measurements) are accepted and correctly reflected in
the RMSE value. No bugs found.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis.exceptions import QualityError
from topocore.analysis.quality.rmse import RMSEAnalysis


def test_1d_known_residuals() -> None:
    result = RMSEAnalysis.compute(np.array([10.0, 20.0]), np.array([7.0, 24.0]))
    assert result.vertical == pytest.approx(np.sqrt((3**2 + 4**2) / 2))


def test_3d_known_components() -> None:
    reference = np.zeros((2, 3))
    observed = np.array([[3.0, 4.0, 0.0], [3.0, 4.0, 0.0]])
    result = RMSEAnalysis.compute(reference, observed)

    assert result.horizontal == pytest.approx(5.0)
    assert result.vertical == pytest.approx(0.0)
    assert result.total == pytest.approx(5.0)


def test_nan_residuals_rejected_as_invalid_not_low_quality() -> None:
    with pytest.raises(QualityError):
        RMSEAnalysis.compute(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_valid_but_poor_quality_data_is_accepted() -> None:
    """
    Large but finite residuals represent real, valid (if imprecise)
    measurements -- must NOT be rejected the way NaN/Inf are.
    """
    result = RMSEAnalysis.compute(np.array([1.0, 1.0]), np.array([100.0, 100.0]))
    assert result.vertical == pytest.approx(99.0)


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(QualityError):
        RMSEAnalysis.compute(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_rejects_empty_arrays() -> None:
    with pytest.raises(QualityError):
        RMSEAnalysis.compute(np.array([]), np.array([]))


def test_componentwise_matches_stacked_computation() -> None:
    rx, ry, rz = np.zeros(2), np.zeros(2), np.zeros(2)
    ox, oy, oz = np.array([3.0, 3.0]), np.array([4.0, 4.0]), np.zeros(2)

    result = RMSEAnalysis.componentwise(rx, ry, rz, ox, oy, oz)
    assert result.horizontal == pytest.approx(5.0)
