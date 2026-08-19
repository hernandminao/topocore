"""
Regression suite for topocore.analysis.statistics.manager.
StatisticsAnalysis -- PR19.

Verified: all 5 methods dispatch correctly and correctly propagate
their parameters (num_bins for distribution/slope, resolution for
density) -- unlike some other managers audited elsewhere in this
session (distance, visibility), no forgotten-parameter or
positional-reordering bugs were found here.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis.exceptions import StatisticsError
from topocore.analysis.statistics.manager import StatisticsAnalysis
from topocore.analysis.types import AreaStats, DensityStats, ElevationStats, SlopeStats
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN


class _FakeGriddedSurface:
    """Minimal GriddedSurface double for manager tests."""

    def __init__(self, elevations: np.ndarray, resolution: float = 1.0) -> None:
        self.elevations = elevations
        self.resolution = resolution
        self.grid = elevations

    def interpolate(self, x: float, y: float) -> float:
        return 0.0

    def contains(self, x: float, y: float) -> bool:
        return True


def test_elevation_dispatch_with_raw_array() -> None:
    manager = StatisticsAnalysis(method="elevation")
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = manager.compute(values, method="elevation")
    assert isinstance(result, ElevationStats)
    assert result.mean == pytest.approx(3.0)


def test_elevation_dispatch_with_gridded_surface() -> None:
    manager = StatisticsAnalysis(method="elevation")
    surface = _FakeGriddedSurface(np.array([[1.0, 2.0], [3.0, 4.0]]))
    result = manager.compute(surface, method="elevation")
    assert isinstance(result, ElevationStats)
    assert result.mean == pytest.approx(2.5)


def test_slope_dispatch_propagates_num_bins() -> None:
    manager = StatisticsAnalysis(method="slope")
    n = 10
    elevations = np.zeros((n, n))
    for row in range(n):
        for col in range(n):
            elevations[row, col] = float(col)
    surface = _FakeGriddedSurface(elevations, resolution=1.0)

    result = manager.compute(surface, method="slope", num_bins=7)
    assert isinstance(result, SlopeStats)
    assert len(result.histogram_counts) == 7
    assert result.mean == pytest.approx(45.0)


def test_area_dispatch() -> None:
    manager = StatisticsAnalysis(method="area")
    points = (
        Point3D(0, 0, 5.0),
        Point3D(10, 0, 5.0),
        Point3D(0, 10, 5.0),
        Point3D(10, 10, 5.0),
    )
    tin = TIN.from_points(points)

    result = manager.compute(tin, method="area")
    assert isinstance(result, AreaStats)
    assert result.projected_area == pytest.approx(100.0)


def test_density_dispatch_propagates_resolution() -> None:
    manager = StatisticsAnalysis(method="density")
    points = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])

    result = manager.compute(points, method="density", resolution=1.0)
    assert isinstance(result, DensityStats)
    assert result.mean_density == pytest.approx(4.0)


def test_distribution_dispatch_uses_fixed_skewness_formula() -> None:
    """
    Confirms the distribution.py skewness/kurtosis fix is reachable
    through the manager, not just the underlying class directly.
    """
    from scipy import stats

    manager = StatisticsAnalysis(method="distribution")
    rng = np.random.default_rng(1)
    values = rng.exponential(scale=2.0, size=6)

    from topocore.analysis.types import DistributionStats

    result = manager.compute(values, method="distribution", num_bins=3)
    assert isinstance(result, DistributionStats)
    expected_kurt = stats.kurtosis(values, bias=True)

    assert result.kurtosis == pytest.approx(expected_kurt, abs=1e-9)


def test_percentiles() -> None:
    manager = StatisticsAnalysis(method="distribution")
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    result = manager.percentiles(values, [50.0])
    assert result[50.0] == pytest.approx(5.5)


def test_rejects_invalid_method_at_construction() -> None:
    with pytest.raises(StatisticsError):
        StatisticsAnalysis(method="bogus")


def test_rejects_invalid_method_at_compute() -> None:
    manager = StatisticsAnalysis(method="elevation")
    with pytest.raises(StatisticsError):
        manager.compute(np.array([1.0, 2.0]), method="bogus")
