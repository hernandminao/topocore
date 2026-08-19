"""
Regression suite for topocore.analysis.statistics.area.AreaStatistics
-- PR19. Verified with a horizontal triangle (surface == projected
exactly) and a uniformly 45-degree tilted plane (known ratio
surface/projected = sqrt(2)). No bugs found.
"""

from __future__ import annotations

import math

import pytest

from topocore.analysis.exceptions import StatisticsError
from topocore.analysis.statistics.area import AreaStatistics
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN


def test_horizontal_triangle_surface_equals_projected() -> None:
    p1, p2, p3 = Point3D(0, 0, 5), Point3D(4, 0, 5), Point3D(0, 3, 5)
    projected = AreaStatistics.projected_triangle_area(p1, p2, p3)
    surface = AreaStatistics.triangle_area(p1, p2, p3)

    assert projected == pytest.approx(6.0)
    assert surface == pytest.approx(6.0)


def test_45_degree_tilted_plane_known_ratio() -> None:
    p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 0, 1), Point3D(0, 1, 0)
    projected = AreaStatistics.projected_triangle_area(p1, p2, p3)
    surface = AreaStatistics.triangle_area(p1, p2, p3)

    assert projected == pytest.approx(0.5)
    assert surface == pytest.approx(0.5 * math.sqrt(2))
    assert surface / projected == pytest.approx(math.sqrt(2))


def test_compute_over_tin() -> None:
    points = (
        Point3D(0, 0, 5.0),
        Point3D(10, 0, 5.0),
        Point3D(0, 10, 5.0),
        Point3D(10, 10, 5.0),
    )
    tin = TIN.from_points(points)

    result = AreaStatistics.compute(tin)

    assert result.projected_area == pytest.approx(100.0)
    assert result.surface_area == pytest.approx(100.0)  # flat, so surface == projected
    assert result.count == tin.triangle_count


def test_rejects_empty_tin() -> None:
    class _EmptyTIN:
        triangle_count = 0

    with pytest.raises(StatisticsError):
        AreaStatistics.compute(_EmptyTIN())  # type: ignore[arg-type]
