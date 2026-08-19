"""
Regression suite for topocore.analysis.profile.cross_section and
.multi_profile -- PR19.

CrossSectionProfile confirmed to inherit the axis-offset fix (see
test_transversal.py) since it delegates to TransversalProfile at
every alignment vertex. MultiProfile confirmed NOT to share the bug
-- it uses explicit, caller-supplied offsets (defaulting to (0.0,))
rather than auto-generating a grid, so it was not modified.
"""

from __future__ import annotations

import pytest

from topocore.analysis.exceptions import ProfileError
from topocore.analysis.profile.cross_section import CrossSectionProfile
from topocore.analysis.profile.multi_profile import MultiProfile
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
# CrossSectionProfile -- inherits the TransversalProfile fix.
# ----------------------------------------------------------------------


def test_cross_section_includes_axis_offset_at_every_vertex(
    flat_surface: SurfaceAdapter,
) -> None:
    """
    The exact reproduction, at the CrossSectionProfile level: before
    the fix, none of the generated sections (using the non-exact
    width=10/interval=3 combination) included offset=0.0.
    """
    axis = [(0.0, 0.0), (10.0, 0.0), (20.0, 5.0)]
    sections = CrossSectionProfile(interval=3.0, width=10.0).generate(axis, flat_surface)

    assert len(sections) == 3
    for section in sections:
        offsets = [p.offset for p in section.points]
        assert offsets.count(0.0) == 1


def test_cross_section_cumulative_station_increases_along_axis(
    flat_surface: SurfaceAdapter,
) -> None:
    axis = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    sections = CrossSectionProfile(interval=5.0, width=5.0).generate(axis, flat_surface)

    stations = [section.points[0].station for section in sections]
    assert stations == [
        0.0,
        10.0,
        20.0,
    ]  # matches cumulative distance along a straight axis


def test_cross_section_rejects_fewer_than_two_vertices(
    flat_surface: SurfaceAdapter,
) -> None:
    with pytest.raises(ProfileError):
        CrossSectionProfile().generate([(0.0, 0.0)], flat_surface)


# ----------------------------------------------------------------------
# MultiProfile -- confirmed NOT to share the bug (explicit offsets).
# ----------------------------------------------------------------------


def test_multi_profile_default_offset_is_zero(flat_surface: SurfaceAdapter) -> None:
    results = MultiProfile(interval=2.0).generate((0.0, 0.0), (10.0, 0.0), flat_surface)
    assert len(results) == 1
    assert all(p.offset == 0.0 for p in results[0].points)


def test_multi_profile_explicit_offsets_used_exactly() -> None:
    profile = MultiProfile(interval=2.0, offsets=[-3.0, 0.0, 5.0])
    assert profile.offsets == (-3.0, 0.0, 5.0)


def test_multi_profile_parallel_offset_geometry(
    flat_surface: SurfaceAdapter,
) -> None:
    # Axis along X -- an offset profile should run parallel, shifted in Y.
    results = MultiProfile(interval=5.0, offsets=[3.0]).generate((0.0, 0.0), (10.0, 0.0), flat_surface)
    for point in results[0].points:
        assert point.y == pytest.approx(3.0)


def test_multi_profile_rejects_empty_offsets() -> None:
    with pytest.raises(ProfileError):
        MultiProfile(offsets=[])
