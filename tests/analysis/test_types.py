"""
Regression suite for topocore.analysis.types -- PR19.

Focused on the computed properties/methods that have real logic
(not plain dataclass field access), since the plain result
dataclasses are already exercised throughout this session's broader
test suites. Includes a real, minor bug found and fixed here:
QualitySummary was a legitimate, defined class missing from
__all__.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis import types
from topocore.analysis.types import (
    ChamferResult,
    ElevationStats,
    HausdorffResult,
    IntervisibilityResult,
    QualitySummary,
    ViewshedResult,
)


def test_qualitysummary_now_in_all() -> None:
    """
    The exact regression: QualitySummary is a real class defined in
    this module, but was missing from __all__.
    """
    assert "QualitySummary" in types.__all__
    assert hasattr(types, "QualitySummary")


def test_qualitysummary_constructs() -> None:
    summary = QualitySummary(metric="rmse", value=0.05, units="m", threshold=0.10, passed=True)
    assert summary.passed is True


def test_viewshed_visibility_ratio_guards_against_zero_division() -> None:
    result = ViewshedResult(visibility_map=np.empty((0, 0), dtype=np.bool_), visible_count=0, total_count=0)
    assert result.visibility_ratio == 0.0


def test_viewshed_visibility_ratio_known_value() -> None:
    result = ViewshedResult(visible_count=3, total_count=4)
    assert result.visibility_ratio == pytest.approx(0.75)


def test_intervisibility_ratio_guards_against_zero_division() -> None:
    result = IntervisibilityResult(visible_pairs=0, total_pairs=0)
    assert result.visibility_ratio == 0.0


def test_intervisibility_ratio_known_value() -> None:
    result = IntervisibilityResult(visible_pairs=1, total_pairs=3)
    assert result.visibility_ratio == pytest.approx(1 / 3)


def test_elevation_stats_percentiles_known_values() -> None:
    stats = ElevationStats(minimum=1.0, maximum=10.0, mean=5.5, std=2.0, median=5.5, range=9.0, count=10)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    result = stats.percentiles(values, [0.0, 50.0, 100.0])

    assert result[0.0] == pytest.approx(1.0)
    assert result[50.0] == pytest.approx(5.5)
    assert result[100.0] == pytest.approx(10.0)


def test_elevation_stats_percentiles_excludes_nan() -> None:
    stats = ElevationStats(minimum=1.0, maximum=3.0, mean=2.0, std=1.0, median=2.0, range=2.0, count=2)
    values = np.array([1.0, np.nan, 3.0])

    result = stats.percentiles(values, [50.0])
    assert result[50.0] == pytest.approx(2.0)


def test_elevation_stats_percentiles_all_nan_returns_empty() -> None:
    stats = ElevationStats(minimum=0.0, maximum=0.0, mean=0.0, std=0.0, median=0.0, range=0.0, count=0)
    result = stats.percentiles(np.array([np.nan, np.nan]), [50.0])
    assert result == {}


def test_hausdorff_symmetric_matches_hausdorff_field() -> None:
    result = HausdorffResult(directed_forward=3.0, directed_backward=7.0, hausdorff=7.0)
    assert result.symmetric == result.hausdorff == 7.0


def test_chamfer_symmetric_matches_chamfer_field() -> None:
    result = ChamferResult(directed_forward=3.0, directed_backward=7.0, chamfer=5.0)
    assert result.symmetric == result.chamfer == 5.0
