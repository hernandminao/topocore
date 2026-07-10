"""
topocore.linalg.vector3d
========================

Immutable three-dimensional vector.

This module defines the Vector3D class, which represents a vector in a
three-dimensional Cartesian coordinate system.

The class is immutable and designed as a value object.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, hypot
from typing import Self, final

from topocore.core.exceptions import MathError
from topocore.math.tolerance import is_close, is_zero
from topocore.math.validation import validate_coordinate


@final
@dataclass(frozen=True, slots=True)
class Vector3D:
    """
    Immutable three-dimensional vector.
    """

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        validate_coordinate(self.x)
        validate_coordinate(self.y)
        validate_coordinate(self.z)

    # ==========================================================
    # Properties
    # ==========================================================

    @property
    def length(self) -> float:
        """
        Vector magnitude.
        """
        return hypot(
            self.x,
            self.y,
            self.z,
        )

    @property
    def length_squared(self) -> float:
        """
        Squared vector magnitude.
        """
        return self.x * self.x + self.y * self.y + self.z * self.z

    @property
    def is_zero(self) -> bool:
        """
        True if the vector magnitude is numerically zero.
        """
        return is_zero(
            self.length,
        )

    # ==========================================================
    # Geometry
    # ==========================================================

    def normalized(self) -> Self:
        """
        Return the normalized vector.
        """
        if self.is_zero:
            raise MathError("Cannot normalize a zero vector.")

        return self / self.length

    def dot(
        self,
        other: Self,
    ) -> float:
        """
        Compute the dot product.
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(
        self,
        other: Self,
    ) -> Self:
        """
        Compute the cross product.
        """
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def angle_to(
        self,
        other: Self,
    ) -> float:
        """
        Compute the angle between vectors.
        """
        denominator = self.length * other.length

        if is_zero(
            denominator,
        ):
            raise MathError("Cannot compute the angle of a zero vector.")

        cosine = self.dot(other) / denominator

        cosine = max(
            -1.0,
            min(
                1.0,
                cosine,
            ),
        )

        return acos(cosine)

    def almost_equals(
        self,
        other: Self,
    ) -> bool:
        """
        Floating-point comparison.
        """
        return is_close(self.x, other.x) and is_close(self.y, other.y) and is_close(self.z, other.z)

    # ==========================================================
    # Conversion
    # ==========================================================

    def to_tuple(
        self,
    ) -> tuple[float, float, float]:
        return (
            self.x,
            self.y,
            self.z,
        )

    def to_dict(
        self,
    ) -> dict[str, float]:
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
        if not isinstance(other, Vector3D):
            return NotImplemented

        return Vector3D(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(
        self,
        other: object,
    ) -> Self:
        if not isinstance(other, Vector3D):
            return NotImplemented

        return Vector3D(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __mul__(
        self,
        scalar: object,
    ) -> Self:
        if not isinstance(
            scalar,
            (int, float),
        ):
            return NotImplemented

        value = float(scalar)

        return Vector3D(
            self.x * value,
            self.y * value,
            self.z * value,
        )

    def __rmul__(
        self,
        scalar: object,
    ) -> Self:
        return self.__mul__(scalar)

    def __truediv__(
        self,
        scalar: object,
    ) -> Self:
        if not isinstance(
            scalar,
            (int, float),
        ):
            return NotImplemented

        value = float(scalar)

        if is_zero(value):
            raise MathError("Division by zero.")

        return Vector3D(
            self.x / value,
            self.y / value,
            self.z / value,
        )

    def __neg__(
        self,
    ) -> Self:
        return Vector3D(
            -self.x,
            -self.y,
            -self.z,
        )

    def __abs__(
        self,
    ) -> float:
        return self.length


__all__ = [
    "Vector3D",
]
