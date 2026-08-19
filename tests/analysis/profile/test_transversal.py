"""
Regression suite for topocore.analysis.profile.transversal.
TransversalProfile -- PR19.

Includes a real bug found and fixed in this session:
_generate_offsets() stepped from -width in increments of interval,
which only hit exactly offset=0 (the axis/centerline -- the most
important reference point of any cross-section) when width happened
to be an exact multiple of interval. Confirmed directly:
width=10, interval=3 generated [-10,-7,-4,-1,2,5,8,10], with the
points nearest zero being -1 and +2 -- straddling but never
touching the axis. This broke the guarantee LongitudinalProfile
already provides for its own equivalent point (station=0 is always
present by construction).

Fixed by building offsets outward from 0 on the positive side, then
mirroring to negative -- 0 is included by construction, exactly
once, regardless of whether width is an exact multiple of interval.
This also fixed the interval >= width special case, which
previously returned only [-width, width], missing 0 as well.
"""

from __future__ import annotations

import pytest

from topocore.analysis.exceptions import ProfileError
from topocore.analysis.profile.transversal import TransversalProfile
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN

from ._helpers import SurfaceAdapter


@pytest.fixture
def flat_surface() -> SurfaceAdapter:
    points = (
        Point3D(-100, -100, 5.0),
        Point3D(100, -100, 5.0),
        Point3D(-100, 100, 5.0),
        Point3D(100, 100, 5.0),
    )
    return SurfaceAdapter(TIN.from_points(points))


# ----------------------------------------------------------------------
# The real bug: offset=0 (the axis) must always be present.
# ----------------------------------------------------------------------


def test_axis_offset_present_for_non_exact_division() -> None:
    """
    The exact reproduction: before the fix, width=10, interval=3
    never included offset=0.0 at all.
    """
    profile = TransversalProfile(interval=3.0, width=10.0)
    offsets = profile._generate_offsets()

    assert offsets.count(0.0) == 1
    assert -10.0 in offsets
    assert 10.0 in offsets
    assert offsets == sorted(offsets)
    assert len(offsets) == len(set(offsets))


def test_axis_offset_appears_exactly_once_for_exact_division() -> None:
    profile = TransversalProfile(interval=5.0, width=10.0)
    offsets = profile._generate_offsets()

    assert offsets.count(0.0) == 1
    assert offsets == [-10.0, -5.0, 0.0, 5.0, 10.0]


def test_existing_behavior_preserved_for_exact_multiple() -> None:
    """
    width=10, interval=2 (exact multiple) must produce the same
    result as before the fix -- confirms no regression for the
    already-correct case.
    """
    profile = TransversalProfile(interval=2.0, width=10.0)
    offsets = profile._generate_offsets()

    assert offsets == [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def test_interval_exceeding_width_still_includes_axis() -> None:
    """
    The old "interval >= width" special case returned only
    [-width, width], missing 0 -- now handled by the same general
    logic, which naturally includes it.
    """
    profile = TransversalProfile(interval=100.0, width=10.0)
    offsets = profile._generate_offsets()

    assert offsets == [-10.0, 0.0, 10.0]


def test_axis_point_z_matches_surface_at_station(
    flat_surface: SurfaceAdapter,
) -> None:
    profile = TransversalProfile(interval=3.0, width=10.0)
    result = profile.generate((0.0, 0.0), (10.0, 0.0), 5.0, flat_surface)

    axis_point = next(p for p in result.points if p.offset == 0.0)
    assert axis_point.x == pytest.approx(5.0)
    assert axis_point.y == pytest.approx(0.0)
    assert axis_point.z == pytest.approx(5.0)  # flat surface at z=5


# ----------------------------------------------------------------------
# Perpendicularity / geometry, unaffected by the fix.
# ----------------------------------------------------------------------


def test_offsets_are_perpendicular_to_axis(flat_surface: SurfaceAdapter) -> None:
    # Axis along X -- transversal must extend along Y.
    profile = TransversalProfile(interval=5.0, width=10.0)
    result = profile.generate((0.0, 0.0), (10.0, 0.0), 5.0, flat_surface)

    for point in result.points:
        assert point.x == pytest.approx(5.0)  # constant along the axis-aligned coordinate
        assert point.y == pytest.approx(point.offset)  # varies exactly with offset


# ----------------------------------------------------------------------
# Validation.
# ----------------------------------------------------------------------


def test_rejects_nonpositive_interval() -> None:
    with pytest.raises(ProfileError):
        TransversalProfile(interval=0.0, width=10.0)


def test_rejects_nonpositive_width() -> None:
    with pytest.raises(ProfileError):
        TransversalProfile(interval=1.0, width=0.0)


def test_rejects_zero_length_axis(flat_surface: SurfaceAdapter) -> None:
    profile = TransversalProfile()
    with pytest.raises(ProfileError):
        profile.generate((5.0, 5.0), (5.0, 5.0), 0.0, flat_surface)
