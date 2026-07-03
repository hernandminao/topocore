"""
topocore.geometry.point2d
=========================

Immutable two-dimensional point.

This module defines the Point2D class used throughout TopoCore.

A point represents a position in the Cartesian plane and differs
conceptually from a vector.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Self
from typing import final
from typing import override

from topocore.geometry.base import Geometry
from topocore.linalg.vector2d import Vector2D
from topocore.math.tolerance import is_close
from topocore.math.validation import validate_coordinate


@final
@dataclass(frozen=True, slots=True)
class Point2D(Geometry):
    """
    Immutable point in a two-dimensional Cartesian coordinate system.

    Parameters
    ----------
    x
        X coordinate.

    y
        Y coordinate.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        """
        Validate coordinates.
        """
        validate_coordinate(self.x)
        validate_coordinate(self.y)

    # ==========================================================
    # Geometry
    # ==========================================================

    def distance_to(
        self,
        other: Self,
    ) -> float:
        """
        Compute the Euclidean distance to another point.

        Parameters
        ----------
        other
            Destination point.

        Returns
        -------
        float
        """
        return hypot(
            other.x - self.x,
            other.y - self.y,
        )

    def squared_distance_to(
        self,
        other: Self,
    ) -> float:
        """
        Compute the squared Euclidean distance.

        Parameters
        ----------
        other
            Destination point.

        Returns
        -------
        float
        """
        dx = other.x - self.x
        dy = other.y - self.y

        return dx * dx + dy * dy

    def midpoint(
        self,
        other: Self,
    ) -> Self:
        """
        Compute the midpoint between two points.

        Parameters
        ----------
        other
            Other point.

        Returns
        -------
        Point2D
        """
        return Point2D(
            (self.x + other.x) / 2.0,
            (self.y + other.y) / 2.0,
        )

    def vector_to(
        self,
        other: Self,
    ) -> Vector2D:
        """
        Compute the vector from this point to another.

        Parameters
        ----------
        other
            Destination point.

        Returns
        -------
        Vector2D
        """
        return Vector2D(
            other.x - self.x,
            other.y - self.y,
        )

    def translate(
        self,
        vector: Vector2D,
    ) -> Self:
        """
        Translate the point by a vector.

        Parameters
        ----------
        vector
            Translation vector.

        Returns
        -------
        Point2D
        """
        return Point2D(
            self.x + vector.x,
            self.y + vector.y,
        )

    def almost_equals(
        self,
        other: Self,
    ) -> bool:
        """
        Compare two points using floating-point tolerance.

        Parameters
        ----------
        other
            Other point.

        Returns
        -------
        bool
        """
        return (
            is_close(self.x, other.x)
            and is_close(self.y, other.y)
        )

    # ==========================================================
    # Conversion
    # ==========================================================

    def to_tuple(
        self,
    ) -> tuple[float, float]:
        """
        Convert to tuple.

        Returns
        -------
        tuple[float, float]
        """
        return (
            self.x,
            self.y,
        )

    @override
    def to_dict(
        self,
    ) -> dict[str, float]:
        """
        Convert to dictionary.

        Returns
        -------
        dict[str, float]
        """
        return {
            "x": self.x,
            "y": self.y,
        }

    # ==========================================================
    # Operators
    # ==========================================================

    def __add__(
        self,
        other: object,
    ) -> Self:
        """
        Point + Vector -> Point
        """
        if not isinstance(other, Vector2D):
            return NotImplemented

        return Point2D(
            self.x + other.x,
            self.y + other.y,
        )

    def __sub__(
        self,
        other: object,
    ) -> Self | Vector2D:
        """
        Supported operations.

        Point - Point -> Vector
        Point - Vector -> Point
        """
        if isinstance(other, Point2D):
            return Vector2D(
                self.x - other.x,
                self.y - other.y,
            )

        if isinstance(other, Vector2D):
            return Point2D(
                self.x - other.x,
                self.y - other.y,
            )

        return NotImplemented


__all__ = [
    "Point2D",
]