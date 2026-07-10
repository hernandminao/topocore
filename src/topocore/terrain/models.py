"""
topocore.terrain.models
=======================

Immutable terrain geometry models.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from topocore.geometry.point3d import Point3D

from .constants import DEFAULT_GRID_ROTATION
from .enums import BreaklineType


@dataclass(frozen=True, slots=True)
class Edge:
    """
    Represents an edge of a triangulation.
    """

    start: Point3D
    end: Point3D
    is_breakline: bool = False

    @property
    def length(self) -> float:
        """
        Return the 3D edge length.
        """
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        dz = self.end.z - self.start.z

        return sqrt(dx * dx + dy * dy + dz * dz)


@dataclass(frozen=True, slots=True)
class Triangle:
    """
    Represents a triangle of a TIN.
    """

    p1: Point3D
    p2: Point3D
    p3: Point3D

    @property
    def centroid(self) -> Point3D:
        """
        Triangle centroid.
        """
        return Point3D(
            (self.p1.x + self.p2.x + self.p3.x) / 3.0,
            (self.p1.y + self.p2.y + self.p3.y) / 3.0,
            (self.p1.z + self.p2.z + self.p3.z) / 3.0,
        )

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """
        XY bounding box.
        """
        return (
            min(self.p1.x, self.p2.x, self.p3.x),
            min(self.p1.y, self.p2.y, self.p3.y),
            max(self.p1.x, self.p2.x, self.p3.x),
            max(self.p1.y, self.p2.y, self.p3.y),
        )

    @property
    def area(self) -> float:
        """
        Projected XY area.
        """
        return abs(
            (
                self.p1.x * (self.p2.y - self.p3.y)
                + self.p2.x * (self.p3.y - self.p1.y)
                + self.p3.x * (self.p1.y - self.p2.y)
            )
            * 0.5
        )


@dataclass(frozen=True, slots=True)
class Breakline:
    """
    Represents a breakline.
    """

    id: str
    name: str
    points: tuple[Point3D, ...]
    type: BreaklineType = BreaklineType.HARD

    @property
    def is_closed(self) -> bool:
        """
        Whether the breakline is closed.
        """
        return len(self.points) > 2 and self.points[0] == self.points[-1]

    @property
    def vertex_count(self) -> int:
        """
        Number of vertices.
        """
        return len(self.points)


@dataclass(frozen=True, slots=True)
class GridDefinition:
    """
    Defines a raster grid.
    """

    origin_x: float
    origin_y: float

    min_x: float
    min_y: float

    max_x: float
    max_y: float

    resolution: float

    rotation: float = DEFAULT_GRID_ROTATION

    @property
    def width(self) -> int:
        """
        Number of columns.
        """
        return int(round((self.max_x - self.min_x) / self.resolution)) + 1

    @property
    def height(self) -> int:
        """
        Number of rows.
        """
        return int(round((self.max_y - self.min_y) / self.resolution)) + 1


@dataclass(frozen=True, slots=True)
class ContourLine:
    """
    Represents a contour line.
    """

    elevation: float

    points: tuple[Point3D, ...]

    closed: bool = False

    @property
    def vertex_count(self) -> int:
        """
        Number of vertices.
        """
        return len(self.points)


__all__ = [
    "Edge",
    "Triangle",
    "Breakline",
    "GridDefinition",
    "ContourLine",
]
