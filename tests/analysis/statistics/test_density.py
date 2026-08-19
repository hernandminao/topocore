"""
Regression suite for topocore.analysis.statistics.density.
DensityStatistics -- PR19. Verified with known point/cell-area
ratios and the degenerate coincident-points case. No bugs found.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis.exceptions import StatisticsError
from topocore.analysis.statistics.density import DensityStatistics


def test_known_density_single_cell() -> None:
    points = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
    result = DensityStatistics(resolution=1.0).compute(points)
    assert result.mean_density == pytest.approx(4.0)


def test_two_cells_equal_density() -> None:
    points = np.array([[0.1, 0.1], [0.2, 0.2], [5.1, 5.1], [5.2, 5.2]])
    result = DensityStatistics(resolution=1.0).compute(points)
    assert result.mean_density == pytest.approx(2.0)
    assert result.minimum_density == pytest.approx(2.0)
    assert result.maximum_density == pytest.approx(2.0)


def test_coincident_points() -> None:
    points = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    result = DensityStatistics(resolution=1.0).compute(points)
    assert result.mean_density == pytest.approx(3.0)


def test_rejects_empty_input() -> None:
    with pytest.raises(StatisticsError):
        DensityStatistics().compute(np.empty((0, 2)))


def test_rejects_nonpositive_resolution() -> None:
    with pytest.raises(StatisticsError):
        DensityStatistics(resolution=0.0)


def test_accepts_list_of_tuples() -> None:
    result = DensityStatistics(resolution=1.0).compute([(0.1, 0.1), (0.2, 0.2)])
    assert result.mean_density == pytest.approx(2.0)
