"""
topocore.geometry.bbox3d
========================

Immutable three-dimensional axis-aligned bounding box.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, final, override

from topocore.geometry.base import Geometry
from topocore.geometry.point3d import Point3D
from topocore.math.validation import validate_coordinate


@final
@dataclass(frozen=True, slots=True)
class BBox3D(Geometry):
    """
    Immutable axis-aligned bounding box (AABB).

    Parameters
    ----------
    min_x
        Minimum X coordinate.

    min_y
        Minimum Y coordinate.

    min_z
        Minimum Z coordinate.

    max_x
        Maximum X coordinate.

    max_y
        Maximum Y coordinate.

    max_z
        Maximum Z coordinate.
    """

    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    def __post_init__(self) -> None:
        """
        Validate coordinates.
        """
        validate_coordinate(self.min_x)
        validate_coordinate(self.min_y)
        validate_coordinate(self.min_z)
        validate_coordinate(self.max_x)
        validate_coordinate(self.max_y)
        validate_coordinate(self.max_z)

        if self.min_x > self.max_x:
            raise ValueError("min_x must be less than or equal to max_x.")

        if self.min_y > self.max_y:
            raise ValueError("min_y must be less than or equal to max_y.")

        if self.min_z > self.max_z:
            raise ValueError("min_z must be less than or equal to max_z.")

    # ==========================================================
    # Properties
    # ==========================================================

    @property
    def width(self) -> float:
        """
        Width of the bounding box.
        """
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        """
        Height of the bounding box.
        """
        return self.max_y - self.min_y

    @property
    def depth(self) -> float:
        """
        Depth of the bounding box.
        """
        return self.max_z - self.min_z

    @property
    def volume(self) -> float:
        """
        Volume of the bounding box.
        """
        return self.width * self.height * self.depth

    @property
    def center(self) -> Point3D:
        """
        Center point.
        """
        return Point3D(
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
            (self.min_z + self.max_z) / 2.0,
        )

    @property
    def min_point(self) -> Point3D:
        """
        Minimum corner.
        """
        return Point3D(
            self.min_x,
            self.min_y,
            self.min_z,
        )

    @property
    def max_point(self) -> Point3D:
        """
        Maximum corner.
        """
        return Point3D(
            self.max_x,
            self.max_y,
            self.max_z,
        )

    # ==========================================================
    # Spatial predicates
    # ==========================================================

    def contains(
        self,
        point: Point3D,
    ) -> bool:
        """
        Return True if the point is inside the bounding box.
        """
        return (
            self.min_x <= point.x <= self.max_x
            and self.min_y <= point.y <= self.max_y
            and self.min_z <= point.z <= self.max_z
        )

    def intersects(
        self,
        other: Self,
    ) -> bool:
        """
        Return True if two bounding boxes intersect.
        """
        return not (
            self.max_x < other.min_x
            or self.min_x > other.max_x
            or self.max_y < other.min_y
            or self.min_y > other.max_y
            or self.max_z < other.min_z
            or self.min_z > other.max_z
        )

    # ==========================================================
    # Operations
    # ==========================================================

    def expand(
        self,
        margin: float,
    ) -> Self:
        """
        Expand the bounding box by a constant margin.
        """
        if margin < 0.0:
            raise ValueError("Margin must be non-negative.")

        return BBox3D(
            self.min_x - margin,
            self.min_y - margin,
            self.min_z - margin,
            self.max_x + margin,
            self.max_y + margin,
            self.max_z + margin,
        )

    def union(
        self,
        other: Self,
    ) -> Self:
        """
        Return the union of two bounding boxes.
        """
        return BBox3D(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            min(self.min_z, other.min_z),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y),
            max(self.max_z, other.max_z),
        )

    @override
    def to_dict(
        self,
    ) -> dict[str, float]:
        """
        Convert the bounding box to a dictionary.
        """
        return {
            "min_x": self.min_x,
            "min_y": self.min_y,
            "min_z": self.min_z,
            "max_x": self.max_x,
            "max_y": self.max_y,
            "max_z": self.max_z,
        }


__all__ = [
    "BBox3D",
]
