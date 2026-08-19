"""
Regression suite for topocore.analysis.statistics.slope.
SlopeStatistics -- PR19. Verified against a known 45-degree plane
(z=x), a flat surface (slope=0), and NaN propagation (DTM
convex-hull pattern). No bugs found -- confirmed degrees (not
radians) throughout, and confirmed the dx/dy variable naming (if
ever swapped) would not affect the result, since the final magnitude
uses hypot(dx, dy), which is symmetric in its two arguments.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis.exceptions import StatisticsError
from topocore.analysis.statistics.slope import SlopeStatistics


def test_45_degree_plane_gives_45_degrees_everywhere() -> None:
    n = 10
    elevations = np.zeros((n, n))
    for row in range(n):
        for col in range(n):
            elevations[row, col] = float(col)  # z = x

    slope_arr = SlopeStatistics().slope_array(elevations, resolution=1.0)

    assert slope_arr.min() == pytest.approx(45.0)
    assert slope_arr.max() == pytest.approx(45.0)


def test_flat_surface_gives_zero_slope() -> None:
    flat = np.full((10, 10), 5.0)
    slope_arr = SlopeStatistics().slope_array(flat, resolution=1.0)
    assert slope_arr.max() == pytest.approx(0.0)


def test_nan_propagates_to_slope_without_breaking_computation() -> None:
    elevations = np.zeros((10, 10))
    for row in range(10):
        for col in range(10):
            elevations[row, col] = float(col)
    elevations[3, 3] = np.nan

    slope_arr = SlopeStatistics().slope_array(elevations, resolution=1.0)

    assert np.isnan(slope_arr[3, 3])
    assert np.isfinite(slope_arr).sum() > 0
    assert np.isfinite(slope_arr).sum() < slope_arr.size


def test_compute_from_array_statistics() -> None:
    n = 10
    elevations = np.zeros((n, n))
    for row in range(n):
        for col in range(n):
            elevations[row, col] = float(col)

    result = SlopeStatistics(num_bins=5).compute_from_array(elevations, resolution=1.0)
    assert result.mean == pytest.approx(45.0)


def test_rejects_nonpositive_resolution() -> None:
    with pytest.raises(StatisticsError):
        SlopeStatistics().slope_array(np.zeros((5, 5)), resolution=0.0)


def test_rejects_non_2d_array() -> None:
    with pytest.raises(StatisticsError):
        SlopeStatistics().slope_array(np.zeros(25), resolution=1.0)


def test_rejects_grid_smaller_than_3x3() -> None:
    with pytest.raises(StatisticsError):
        SlopeStatistics().slope_array(np.zeros((2, 2)), resolution=1.0)
