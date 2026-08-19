"""
Regression suite for topocore.analysis.statistics.elevation.
ElevationStatistics -- PR19. Verified with known values and NaN
exclusion (the DTM-convex-hull pattern). No bugs found.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis.exceptions import StatisticsError
from topocore.analysis.statistics.elevation import ElevationStatistics


def test_known_values() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ElevationStatistics.compute(values)

    assert result.minimum == pytest.approx(1.0)
    assert result.maximum == pytest.approx(5.0)
    assert result.mean == pytest.approx(3.0)
    assert result.median == pytest.approx(3.0)
    assert result.range == pytest.approx(4.0)
    assert result.count == 5


def test_nan_values_excluded() -> None:
    values = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
    result = ElevationStatistics.compute(values)

    assert result.count == 3
    assert result.mean == pytest.approx((1.0 + 2.0 + 4.0) / 3.0)


def test_rejects_all_nan() -> None:
    with pytest.raises(StatisticsError):
        ElevationStatistics.compute(np.array([np.nan, np.nan]))


def test_infinite_values_excluded() -> None:
    values = np.array([1.0, 2.0, np.inf, -np.inf, 3.0])
    result = ElevationStatistics.compute(values)
    assert result.count == 3
