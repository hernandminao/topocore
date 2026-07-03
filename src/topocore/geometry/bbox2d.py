"""
topocore.geometry.bbox2d
========================

Immutable two-dimensional axis-aligned bounding box.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self
from typing import final
from typing import override

from topocore.geometry.base import Geometry
from topocore.geometry.point2d import Point2D
from topocore.math.validation import validate_coordinate


@final
@dataclass(frozen=True, slots=True)
class BBox2D(Geometry):
    """
    Immutable axis-aligned bounding box (AABB).

    Parameters
    ----------
    min_x
        Minimum X coordinate.

    min_y
        Minimum Y coordinate.

    max_x
        Maximum X coordinate.

    max_y
        Maximum Y coordinate.
    """

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        """
        Validate the bounding box coordinates.
        """
        validate_coordinate(self.min_x)
        validate_coordinate(self.min_y)
        validate_coordinate(self.max_x)
        validate_coordinate(self.max_y)

        if self.min_x > self.max_x:
            raise ValueError(
                "min_x must be less than or equal to max_x."
            )

        if self.min_y > self.max_y:
            raise ValueError(
                "min_y must be less than or equal to max_y."
            )

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
    def area(self) -> float:
        """
        Area of the bounding box.
        """
        return self.width * self.height

    @property
    def center(self) -> Point2D:
        """
        Center point.
        """
        return Point2D(
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )

    @property
    def min_point(self) -> Point2D:
        """
        Lower-left corner.
        """
        return Point2D(
            self.min_x,
            self.min_y,
        )

    @property
    def max_point(self) -> Point2D:
        """
        Upper-right corner.
        """
        return Point2D(
            self.max_x,
            self.max_y,
        )

    # ==========================================================
    # Spatial predicates
    # ==========================================================

    def contains(
        self,
        point: Point2D,
    ) -> bool:
        """
        Return True if the point is inside the bounding box.
        """
        return (
            self.min_x <= point.x <= self.max_x
            and self.min_y <= point.y <= self.max_y
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
        if margin < 0:
            raise ValueError(
                "Margin must be non-negative."
            )

        return BBox2D(
            self.min_x - margin,
            self.min_y - margin,
            self.max_x + margin,
            self.max_y + margin,
        )

    def union(
        self,
        other: Self,
    ) -> Self:
        """
        Compute the union of two bounding boxes.
        """
        return BBox2D(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y),
        )

    @override
    def to_dict(
        self,
    ) -> dict[str, float]:
        """
        Convert to dictionary.
        """
        return {
            "min_x": self.min_x,
            "min_y": self.min_y,
            "max_x": self.max_x,
            "max_y": self.max_y,
        }


__all__ = [
    "BBox2D",
]