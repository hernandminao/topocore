"""
topocore.geometry.point3d
=========================

Immutable three-dimensional point.

This module defines the Point3D class used throughout TopoCore.

A point represents a position in a three-dimensional Cartesian coordinate
system and differs conceptually from a vector.

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
from topocore.linalg.vector3d import Vector3D
from topocore.math.tolerance import is_close
from topocore.math.validation import validate_coordinate


@final
@dataclass(frozen=True, slots=True)
class Point3D(Geometry):
    """
    Immutable point in a three-dimensional Cartesian coordinate system.

    Parameters
    ----------
    x
        X coordinate.

    y
        Y coordinate.

    z
        Z coordinate.
    """

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        """
        Validate coordinates.
        """
        validate_coordinate(self.x)
        validate_coordinate(self.y)
        validate_coordinate(self.z)

    # ==========================================================
    # Geometry
    # ==========================================================

    def distance_to(
        self,
        other: Self,
    ) -> float:
        """
        Compute the Euclidean distance to another point.
        """
        return hypot(
            other.x - self.x,
            other.y - self.y,
            other.z - self.z,
        )

    def squared_distance_to(
        self,
        other: Self,
    ) -> float:
        """
        Compute the squared Euclidean distance.
        """
        dx = other.x - self.x
        dy = other.y - self.y
        dz = other.z - self.z

        return (
            dx * dx
            + dy * dy
            + dz * dz
        )

    def midpoint(
        self,
        other: Self,
    ) -> Self:
        """
        Compute the midpoint between two points.
        """
        return Point3D(
            (self.x + other.x) / 2.0,
            (self.y + other.y) / 2.0,
            (self.z + other.z) / 2.0,
        )

    def vector_to(
        self,
        other: Self,
    ) -> Vector3D:
        """
        Compute the vector from this point to another.
        """
        return Vector3D(
            other.x - self.x,
            other.y - self.y,
            other.z - self.z,
        )

    def translate(
        self,
        vector: Vector3D,
    ) -> Self:
        """
        Translate the point by a vector.
        """
        return Point3D(
            self.x + vector.x,
            self.y + vector.y,
            self.z + vector.z,
        )

    def almost_equals(
        self,
        other: Self,
    ) -> bool:
        """
        Compare two points using floating-point tolerance.
        """
        return (
            is_close(self.x, other.x)
            and is_close(self.y, other.y)
            and is_close(self.z, other.z)
        )

    # ==========================================================
    # Conversion
    # ==========================================================

    def to_tuple(
        self,
    ) -> tuple[float, float, float]:
        """
        Convert to tuple.
        """
        return (
            self.x,
            self.y,
            self.z,
        )

    @override
    def to_dict(
        self,
    ) -> dict[str, float]:
        """
        Convert to dictionary.
        """
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
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
        if not isinstance(other, Vector3D):
            return NotImplemented

        return Point3D(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(
        self,
        other: object,
    ) -> Self | Vector3D:
        """
        Supported operations.

        Point - Point -> Vector3D
        Point - Vector3D -> Point3D
        """
        if isinstance(other, Point3D):
            return Vector3D(
                self.x - other.x,
                self.y - other.y,
                self.z - other.z,
            )

        if isinstance(other, Vector3D):
            return Point3D(
                self.x - other.x,
                self.y - other.y,
                self.z - other.z,
            )

        return NotImplemented


__all__ = [
    "Point3D",
]