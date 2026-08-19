"""
Regression suite for topocore.analysis.visibility.manager.
VisibilityAnalysis -- PR19.

Includes a real bug found and fixed in this session:
VisibilityAnalysis.viewshed() never passed earth_curvature to the
underlying Viewshed instance, silently always using Viewshed's own
default (True) regardless of what VisibilityConfig.
earth_curvature_correction was actually set to -- unlike
line_of_sight() and intervisibility(), which both correctly pass it
through. Confirmed directly: configuring
earth_curvature_correction=False and calling manager.viewshed(...)
over a huge flat surface at a 6 km max_distance gave 69/113 visible
cells (curvature incorrectly still applied) instead of the correct
113/113 (matching a directly-constructed Viewshed(earth_curvature=
False)).
"""

from __future__ import annotations

import pytest

from topocore.analysis.config import VisibilityConfig
from topocore.analysis.exceptions import VisibilityError
from topocore.analysis.types import IntervisibilityResult, LOSResult, ViewshedResult
from topocore.analysis.visibility.manager import VisibilityAnalysis
from topocore.analysis.visibility.viewshed import Viewshed
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN

from ._helpers import SurfaceAdapter


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
# The real bug: earth_curvature_correction not propagated to viewshed().
# ----------------------------------------------------------------------


def test_viewshed_respects_disabled_curvature_config(
    huge_flat_surface: SurfaceAdapter,
) -> None:
    """
    The exact reproduction: before the fix, this gave 69/113 instead
    of 113/113.
    """
    config = VisibilityConfig(earth_curvature_correction=False, observer_height=1.7)
    manager = VisibilityAnalysis(config=config, method="viewshed")

    result = manager.viewshed((0.0, 0.0), huge_flat_surface, resolution=1000.0, max_distance=6000.0)
    expected = Viewshed(
        observer_height=1.7,
        resolution=1000.0,
        max_distance=6000.0,
        earth_curvature=False,
    ).compute((0.0, 0.0), huge_flat_surface)

    assert result.visible_count == expected.visible_count == result.total_count


def test_viewshed_respects_enabled_curvature_config(
    huge_flat_surface: SurfaceAdapter,
) -> None:
    config = VisibilityConfig(earth_curvature_correction=True, observer_height=1.7)
    manager = VisibilityAnalysis(config=config, method="viewshed")

    result = manager.viewshed((0.0, 0.0), huge_flat_surface, resolution=1000.0, max_distance=6000.0)
    expected = Viewshed(
        observer_height=1.7,
        resolution=1000.0,
        max_distance=6000.0,
        earth_curvature=True,
    ).compute((0.0, 0.0), huge_flat_surface)

    assert result.visible_count == expected.visible_count


def test_line_of_sight_already_respected_curvature_config(
    huge_flat_surface: SurfaceAdapter,
) -> None:
    """
    Confirms line_of_sight() (unaffected by the bug, already correct)
    still works -- a control case.
    """
    config = VisibilityConfig(earth_curvature_correction=False, observer_height=1.7, target_height=0.0)
    manager = VisibilityAnalysis(config=config, method="los")

    result = manager.line_of_sight((0.0, 0.0), (500.0, 0.0), huge_flat_surface)
    assert result.visible is True


# ----------------------------------------------------------------------
# Dispatch.
# ----------------------------------------------------------------------


def test_compute_dispatches_los(huge_flat_surface: SurfaceAdapter) -> None:
    manager = VisibilityAnalysis(method="los")
    result = manager.compute((0.0, 0.0), (500.0, 0.0), huge_flat_surface, method="los")
    assert isinstance(result, LOSResult)
    assert result.visible is True


def test_compute_dispatches_viewshed(huge_flat_surface: SurfaceAdapter) -> None:
    manager = VisibilityAnalysis(method="viewshed")
    result = manager.compute((0.0, 0.0), huge_flat_surface, method="viewshed", resolution=1000.0)
    assert isinstance(result, ViewshedResult)
    assert result.total_count > 0


def test_compute_dispatches_intervisibility(huge_flat_surface: SurfaceAdapter) -> None:
    manager = VisibilityAnalysis(method="intervisibility")
    pts = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)]
    result = manager.compute(pts, huge_flat_surface, method="intervisibility")
    assert isinstance(result, IntervisibilityResult)
    assert result.total_pairs == 3


def test_rejects_invalid_method_at_construction() -> None:
    with pytest.raises(VisibilityError):
        VisibilityAnalysis(method="bogus")


def test_rejects_invalid_method_at_compute(huge_flat_surface: SurfaceAdapter) -> None:
    manager = VisibilityAnalysis(method="los")
    with pytest.raises(VisibilityError):
        manager.compute((0.0, 0.0), (1.0, 1.0), huge_flat_surface, method="bogus")
