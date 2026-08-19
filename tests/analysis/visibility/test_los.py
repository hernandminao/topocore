"""
Regression suite for topocore.analysis.visibility.los.LineOfSight --
PR19.

Includes a real, severe bug found and fixed in this session:
_curvature_correction() used the formula d1**2 / (2R), where d1 is
the distance from the OBSERVER only -- this grows WITHOUT BOUND as
the sample point approaches the TARGET, instead of correctly
returning to zero at both path endpoints (the observer's and
target's own given elevations are already correct/complete and need
no further correction at their own position). The standard, correct
"earth bulge" formula (radio/microwave path engineering, surveying
visibility studies) is the PRODUCT form d1*d2/(2R) -- zero at either
endpoint (d1=0 or d2=0), maximal at the midpoint.

Confirmed directly against the well-known horizon-distance formula
(d = sqrt(2*R*h), ~4.65 km for a 1.7 m eye height looking at a
ground-level target over flat terrain): the OLD formula reported the
target as already invisible at just 1 km. After the fix, the target
remains visible up to ~4.65 km and becomes invisible just beyond,
matching the classic formula closely.
"""

from __future__ import annotations

import math

import pytest

from topocore.analysis.exceptions import VisibilityError
from topocore.analysis.visibility.los import LineOfSight
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN

from ._helpers import SurfaceAdapter

_EARTH_RADIUS_M = 6371000.0


@pytest.fixture
def huge_flat_surface() -> SurfaceAdapter:
    points = (
        Point3D(-100000, -100000, 0.0),
        Point3D(100000, -100000, 0.0),
        Point3D(-100000, 100000, 0.0),
        Point3D(100000, 100000, 0.0),
    )
    return SurfaceAdapter(TIN.from_points(points))


# ----------------------------------------------------------------------
# The curvature formula itself -- direct, decisive evidence.
# ----------------------------------------------------------------------


def test_curvature_correction_is_zero_at_observer_endpoint() -> None:
    assert LineOfSight._curvature_correction(0.0, 1000.0) == pytest.approx(0.0)


def test_curvature_correction_is_zero_at_target_endpoint() -> None:
    assert LineOfSight._curvature_correction(1000.0, 0.0) == pytest.approx(0.0)


def test_curvature_correction_is_symmetric() -> None:
    near_observer = LineOfSight._curvature_correction(10.0, 990.0)
    near_target = LineOfSight._curvature_correction(990.0, 10.0)
    assert near_observer == pytest.approx(near_target)


def test_curvature_correction_is_maximal_at_midpoint() -> None:
    midpoint = LineOfSight._curvature_correction(500.0, 500.0)
    near_observer = LineOfSight._curvature_correction(10.0, 990.0)
    near_target = LineOfSight._curvature_correction(990.0, 10.0)
    assert midpoint > near_observer
    assert midpoint > near_target


# ----------------------------------------------------------------------
# End-to-end: the classic horizon-distance reproduction.
# ----------------------------------------------------------------------


def test_ground_level_target_visible_up_to_classic_horizon_distance(
    huge_flat_surface: SurfaceAdapter,
) -> None:
    """
    The exact reproduction: before the fix, this reported the target
    invisible at just 1 km.
    """
    eye_height = 1.7
    expected_horizon = math.sqrt(2 * _EARTH_RADIUS_M * eye_height)

    los = LineOfSight(
        observer_height=eye_height,
        target_height=0.0,
        earth_curvature=True,
        num_samples=500,
    )

    just_inside = los.compute((0.0, 0.0), (expected_horizon * 0.9, 0.0), huge_flat_surface)
    just_outside = los.compute((0.0, 0.0), (expected_horizon * 1.3, 0.0), huge_flat_surface)

    assert just_inside.visible is True
    assert just_outside.visible is False


def test_curvature_has_no_effect_at_short_distances(
    huge_flat_surface: SurfaceAdapter,
) -> None:
    los_with = LineOfSight(observer_height=10.0, target_height=10.0, earth_curvature=True, num_samples=50)
    los_without = LineOfSight(observer_height=10.0, target_height=10.0, earth_curvature=False, num_samples=50)

    result_with = los_with.compute((0.0, 0.0), (100.0, 0.0), huge_flat_surface)
    result_without = los_without.compute((0.0, 0.0), (100.0, 0.0), huge_flat_surface)

    assert result_with.visible == result_without.visible is True


# ----------------------------------------------------------------------
# Obstacle detection, geometry -- unaffected by the fix.
# ----------------------------------------------------------------------


def test_wall_blocks_line_of_sight() -> None:
    points = (
        Point3D(0, -10, 0.0),
        Point3D(0, 10, 0.0),
        Point3D(50, -10, 0.0),
        Point3D(50, 10, 0.0),
        Point3D(50.1, -10, 20.0),
        Point3D(50.1, 10, 20.0),
        Point3D(100, -10, 0.0),
        Point3D(100, 10, 0.0),
    )
    surface = SurfaceAdapter(TIN.from_points(points))

    los = LineOfSight(observer_height=1.7, target_height=1.7, earth_curvature=False, num_samples=200)
    result = los.compute((0.0, 0.0), (100.0, 0.0), surface)

    assert result.visible is False
    assert len(result.obstacles) > 0


def test_flat_terrain_no_obstacle_is_visible(huge_flat_surface: SurfaceAdapter) -> None:
    los = LineOfSight(observer_height=1.7, target_height=1.7, earth_curvature=False, num_samples=100)
    result = los.compute((0.0, 0.0), (500.0, 0.0), huge_flat_surface)
    assert result.visible is True


def test_coincident_points_are_trivially_visible(
    huge_flat_surface: SurfaceAdapter,
) -> None:
    los = LineOfSight()
    result = los.compute((5.0, 5.0), (5.0, 5.0), huge_flat_surface)
    assert result.visible is True
    assert result.distance == 0.0


def test_result_is_deterministic(huge_flat_surface: SurfaceAdapter) -> None:
    los = LineOfSight(observer_height=1.7, target_height=1.7, earth_curvature=True, num_samples=100)
    result1 = los.compute((0.0, 0.0), (500.0, 0.0), huge_flat_surface)
    result2 = los.compute((0.0, 0.0), (500.0, 0.0), huge_flat_surface)
    assert result1.visible == result2.visible
    assert result1.clearance == result2.clearance


# ----------------------------------------------------------------------
# Validation.
# ----------------------------------------------------------------------


def test_rejects_observer_outside_tin(huge_flat_surface: SurfaceAdapter) -> None:
    los = LineOfSight()
    with pytest.raises(VisibilityError):
        los.compute((-999999.0, -999999.0), (0.0, 0.0), huge_flat_surface)


def test_rejects_negative_observer_height() -> None:
    with pytest.raises(VisibilityError):
        LineOfSight(observer_height=-1.0)


def test_rejects_too_few_samples() -> None:
    with pytest.raises(VisibilityError):
        LineOfSight(num_samples=1)
