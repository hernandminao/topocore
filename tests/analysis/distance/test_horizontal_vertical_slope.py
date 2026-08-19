"""
Regression suite for topocore.analysis.distance.horizontal,
.vertical, and .slope -- PR19. No bugs found.
"""

from __future__ import annotations

import pytest

from topocore.analysis.distance.horizontal import HorizontalDistance
from topocore.analysis.distance.slope import SlopeDistance
from topocore.analysis.distance.vertical import VerticalDistance
from topocore.analysis.exceptions import DistanceError


def test_horizontal_ignores_elevation() -> None:
    result = HorizontalDistance.compute(0, 0, 3, 4)
    assert result.value == pytest.approx(5.0)


def test_vertical_absolute_difference() -> None:
    result = VerticalDistance.compute(5.0, 8.0)
    assert result.value == pytest.approx(3.0)


def test_vertical_signed_difference() -> None:
    assert VerticalDistance.elevation_difference(5.0, 8.0) == pytest.approx(3.0)
    assert VerticalDistance.elevation_difference(8.0, 5.0) == pytest.approx(-3.0)


def test_slope_distance_known_3_4_12_13() -> None:
    # SlopeDistance's own order IS point-grouped: x1,y1,z1,x2,y2,z2.
    result = SlopeDistance.compute(0, 0, 0, 3, 4, 12)
    assert result.value == pytest.approx(13.0)


def test_slope_gradient_45_degrees() -> None:
    gradient = SlopeDistance.slope_gradient(0, 0, 0, 10, 0, 10)
    assert gradient == pytest.approx(100.0)


def test_slope_angle_45_degrees() -> None:
    angle = SlopeDistance.slope_angle(0, 0, 0, 10, 0, 10)
    assert angle == pytest.approx(45.0)


def test_slope_gradient_rejects_zero_horizontal() -> None:
    with pytest.raises(DistanceError):
        SlopeDistance.slope_gradient(0, 0, 0, 0, 0, 10)


def test_slope_engine_correctly_reorders_for_euclidean() -> None:
    """
    Confirms SlopeDistance's internal delegation to EuclideanDistance
    (whose parameter order differs) is correct -- this is the exact
    pattern the manager's dispatcher bug (see test_manager.py) failed
    to replicate before being fixed.
    """
    result = SlopeDistance.compute(1, 2, 3, 5, 7, 15)  # dx=4, dy=5, dz=12
    assert result.value == pytest.approx((4**2 + 5**2 + 12**2) ** 0.5)
