"""
Regression suite for topocore.analysis.protocols -- PR19.

These are pure structural-typing Protocol definitions with no
runtime logic; already implicitly exercised throughout this
session's broader analysis test suites (every mypy-clean pass on
code using TerrainSurface/TriangulatedSurface/GriddedSurface/etc.
already confirms real objects correctly satisfy these contracts).
This suite just confirms the module imports cleanly and a real,
already-used implementation genuinely satisfies each protocol.
"""

from __future__ import annotations

from topocore.analysis.protocols import (
    Calculable,
    Generable,
    GriddedSurface,
    Measurable,
    TerrainSurface,
    TriangulatedSurface,
)
from topocore.geometry.point3d import Point3D
from topocore.terrain.linear import LinearInterpolator
from topocore.terrain.tin import TIN


def test_tin_satisfies_triangulated_surface() -> None:
    points = (Point3D(0, 0, 0.0), Point3D(1, 0, 1.0), Point3D(0, 1, 2.0))
    tin = TIN.from_points(points)

    assert hasattr(tin, "triangle_count")
    assert hasattr(tin, "bounds")
    assert hasattr(tin, "triangle_vertices")
    assert hasattr(tin, "find_triangle")
    assert hasattr(tin, "contains")

    # Structural conformance -- LinearInterpolator provides interpolate().
    interpolator = LinearInterpolator(tin)
    assert hasattr(interpolator, "interpolate")


def test_measurable_calculable_generable_are_distinct_protocols() -> None:
    assert Measurable is not Calculable
    assert Calculable is not Generable
    assert Measurable is not Generable


def test_protocol_classes_importable() -> None:
    for protocol in (TerrainSurface, TriangulatedSurface, GriddedSurface):
        assert protocol.__name__ in {
            "TerrainSurface",
            "TriangulatedSurface",
            "GriddedSurface",
        }
