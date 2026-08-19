"""
Regression suite for topocore.analysis.profile.manager.
ProfileAnalysis -- PR19.

Verified: all 4 methods dispatch correctly, and the
TransversalProfile axis-offset fix is reachable through the manager
(not just the underlying class directly).

Includes a second real bug found and fixed in this session (found by
writing this test suite itself): compute()'s method resolution built
the ProfileMethod enum directly (`ProfileMethod(method or ...)`),
letting a raw, undocumented ValueError escape for an invalid method
string instead of the module's own ProfileError -- unlike __init__(),
which correctly validates before constructing the enum, and unlike
the equivalent dispatchers in DistanceAnalysis/VolumeAnalysis
(already audited and confirmed correct elsewhere in this session).
"""

from __future__ import annotations

import pytest

from topocore.analysis.exceptions import ProfileError
from topocore.analysis.profile.manager import ProfileAnalysis
from topocore.analysis.types import ProfileResult
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


def test_longitudinal_dispatch(flat_surface: SurfaceAdapter) -> None:
    manager = ProfileAnalysis(method="longitudinal")
    result = manager.compute((0.0, 0.0), (10.0, 0.0), flat_surface, method="longitudinal")
    assert isinstance(result, ProfileResult)
    assert result.axis_length == pytest.approx(10.0)


def test_transversal_fix_reachable_through_manager(
    flat_surface: SurfaceAdapter,
) -> None:
    manager = ProfileAnalysis(method="longitudinal")
    result = manager.compute(
        (0.0, 0.0),
        (10.0, 0.0),
        0.0,
        flat_surface,
        method="transversal",
        interval=3.0,
        width=10.0,
    )
    assert isinstance(result, ProfileResult)
    offsets = [p.offset for p in result.points]
    assert offsets.count(0.0) == 1


def test_cross_section_dispatch(flat_surface: SurfaceAdapter) -> None:
    manager = ProfileAnalysis(method="longitudinal")
    axis = [(0.0, 0.0), (10.0, 0.0)]
    results = manager.compute(axis, flat_surface, method="cross_section", interval=3.0, width=10.0)
    assert isinstance(results, list)
    assert len(results) == 2


def test_multi_dispatch(flat_surface: SurfaceAdapter) -> None:
    manager = ProfileAnalysis(method="longitudinal")
    results = manager.compute((0.0, 0.0), (10.0, 0.0), flat_surface, method="multi", offsets=[-2.0, 0.0, 2.0])
    assert isinstance(results, list)
    assert len(results) == 3


def test_rejects_invalid_method_at_construction() -> None:
    with pytest.raises(ProfileError):
        ProfileAnalysis(method="bogus")


def test_rejects_invalid_method_at_compute(flat_surface: SurfaceAdapter) -> None:
    """
    The exact regression: before the fix, this raised a raw
    ValueError instead of ProfileError.
    """
    manager = ProfileAnalysis(method="longitudinal")
    with pytest.raises(ProfileError):
        manager.compute((0.0, 0.0), (10.0, 0.0), flat_surface, method="bogus")
