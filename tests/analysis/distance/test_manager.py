"""
Regression suite for topocore.analysis.distance.manager.
DistanceAnalysis -- PR19.

Includes a real, severe bug found and fixed in this session: the
"euclidean" dispatch branch for 6-argument calls forwarded args
straight through to EuclideanDistance.compute() (`*args`), silently
assuming the natural point-grouped order (x1,y1,z1,x2,y2,z2) matched
EuclideanDistance's own actual order (x1,y1,x2,y2,z1,z2) -- which it
does NOT. Confirmed directly: compute(0,0,0,3,4,12) (expected the
3-4-12-13 right-triangle distance, 13.0) silently returned 8.544
instead, with no error raised. Confirmed the correct reordering
pattern by cross-referencing SlopeDistance's own internal delegation
to the same EuclideanDistance engine, which already explicitly
reorders arguments before calling it -- the manager's dispatcher now
matches that same, already-correct pattern.
"""

from __future__ import annotations

import math

import pytest

from topocore.analysis.distance.manager import DistanceAnalysis
from topocore.analysis.exceptions import DistanceError
from topocore.geodesy.crs import CRS

# ----------------------------------------------------------------------
# The real bug: 6-arg euclidean dispatch parameter order.
# ----------------------------------------------------------------------


def test_euclidean_3d_dispatch_known_3_4_12_13_triangle() -> None:
    """
    The exact reproduction: before the fix, this silently returned
    8.544 instead of 13.0.
    """
    manager = DistanceAnalysis(method="euclidean")
    result = manager.compute(0, 0, 0, 3, 4, 12)
    assert result.value == pytest.approx(13.0)


def test_euclidean_3d_dispatch_asymmetric_case() -> None:
    manager = DistanceAnalysis(method="euclidean")
    result = manager.compute(1, 2, 3, 5, 7, 15)  # dx=4, dy=5, dz=12
    assert result.value == pytest.approx(math.sqrt(4**2 + 5**2 + 12**2))


def test_euclidean_2d_dispatch_unaffected_by_the_fix() -> None:
    manager = DistanceAnalysis(method="euclidean")
    result = manager.compute(0, 0, 3, 4)
    assert result.value == pytest.approx(5.0)


# ----------------------------------------------------------------------
# Other method dispatch, unaffected by the fix.
# ----------------------------------------------------------------------


def test_horizontal_dispatch() -> None:
    manager = DistanceAnalysis(method="euclidean")
    result = manager.compute(0, 0, 3, 4, method="horizontal")
    assert result.value == pytest.approx(5.0)


def test_vertical_dispatch() -> None:
    manager = DistanceAnalysis(method="euclidean")
    result = manager.compute(5.0, 8.0, method="vertical")
    assert result.value == pytest.approx(3.0)


def test_slope_dispatch() -> None:
    manager = DistanceAnalysis(method="euclidean")
    result = manager.compute(0, 0, 0, 3, 4, 12, method="slope")
    assert result.value == pytest.approx(13.0)


def test_gradient_and_angle_passthrough() -> None:
    manager = DistanceAnalysis(method="euclidean")
    assert manager.gradient(0, 0, 0, 10, 0, 10) == pytest.approx(100.0)
    assert manager.angle(0, 0, 0, 10, 0, 10) == pytest.approx(45.0)


# ----------------------------------------------------------------------
# Geodesic -- requires CRS at construction.
# ----------------------------------------------------------------------


def test_geodesic_requires_crs_at_construction() -> None:
    with pytest.raises(DistanceError):
        DistanceAnalysis(method="geodesic")  # no crs given


def test_geodesic_dispatch_with_crs() -> None:
    manager = DistanceAnalysis(method="geodesic", crs=CRS.from_epsg(4326))
    result = manager.compute(0.0, 0.0, 1.0, 0.0)
    assert result.value == pytest.approx(111319.49, abs=1.0)


def test_geodesic_requested_per_call_without_manager_crs_raises() -> None:
    manager = DistanceAnalysis(method="euclidean")
    with pytest.raises(DistanceError):
        manager.compute(0.0, 0.0, 1.0, 0.0, method="geodesic")


# ----------------------------------------------------------------------
# Validation.
# ----------------------------------------------------------------------


def test_rejects_invalid_method_at_construction() -> None:
    with pytest.raises(DistanceError):
        DistanceAnalysis(method="bogus")


def test_rejects_invalid_method_at_compute() -> None:
    manager = DistanceAnalysis(method="euclidean")
    with pytest.raises(DistanceError):
        manager.compute(0, 0, 1, 1, method="bogus")


def test_rejects_wrong_arg_count_for_euclidean() -> None:
    manager = DistanceAnalysis(method="euclidean")
    with pytest.raises(DistanceError):
        manager.compute(0, 0, 1)


def test_available_methods() -> None:
    methods = DistanceAnalysis.available_methods()
    assert set(methods) == {"euclidean", "geodesic", "horizontal", "vertical", "slope"}
