"""Shared test helper -- see analysis.profile._helpers for the same
TerrainSurface-conformance rationale, applied to TriangulatedSurface
here (also requires .interpolate/.contains, plus .find_triangle and
.triangle_vertices/.bounds used by visibility)."""

from __future__ import annotations

from topocore.geometry.point3d import Point3D
from topocore.terrain.linear import LinearInterpolator
from topocore.terrain.tin import TIN


class SurfaceAdapter:
    __slots__ = ("_interpolator", "_tin")

    def __init__(self, tin: TIN) -> None:
        self._tin = tin
        self._interpolator = LinearInterpolator(tin)

    def interpolate(self, x: float, y: float) -> float:
        return self._interpolator.interpolate(x, y)

    def contains(self, x: float, y: float) -> bool:
        return self._tin.contains(x, y)

    def find_triangle(self, x: float, y: float) -> int:
        return self._tin.find_triangle(x, y)

    def triangle_vertices(self, index: int) -> tuple[Point3D, Point3D, Point3D]:
        return self._tin.triangle_vertices(index)

    @property
    def triangle_count(self) -> int:
        return self._tin.triangle_count

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self._tin.bounds
