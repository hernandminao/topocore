"""
topocore.linalg.vector2d
========================

Immutable two-dimensional vector.

This module defines the Vector2D class, which represents a vector in a
two-dimensional Cartesian coordinate system.

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
class Vector2D:
    """
    Immutable two-dimensional vector.

    Parameters
    ----------
    x
        X component.

    y
        Y component.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        """
        Validate vector components.
        """
        validate_coordinate(self.x)
        validate_coordinate(self.y)

    # ==========================================================
    # Properties
    # ==========================================================

    @property
    def length(self) -> float:
        """
        Return the vector magnitude.

        Returns
        -------
        float
        """
        return hypot(
            self.x,
            self.y,
        )

    @property
    def length_squared(self) -> float:
        """
        Return the squared vector magnitude.

        Returns
        -------
        float
        """
        return self.x * self.x + self.y * self.y

    @property
    def is_zero(self) -> bool:
        """
        Return True if the vector magnitude is numerically zero.

        Returns
        -------
        bool
        """
        return is_zero(self.length)

    # ==========================================================
    # Basic operations
    # ==========================================================

    def normalized(self) -> Self:
        """
        Return a normalized copy of the vector.

        Returns
        -------
        Vector2D

        Raises
        ------
        MathError
            If the vector magnitude is zero.
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

        Parameters
        ----------
        other
            Other vector.

        Returns
        -------
        float
        """
        return self.x * other.x + self.y * other.y

    def cross(
        self,
        other: Self,
    ) -> float:
        """
        Compute the 2D cross product.

        Parameters
        ----------
        other
            Other vector.

        Returns
        -------
        float
        """
        return self.x * other.y - self.y * other.x

    def angle_to(
        self,
        other: Self,
    ) -> float:
        """
        Compute the angle to another vector.

        Parameters
        ----------
        other
            Other vector.

        Returns
        -------
        float
            Angle in radians.

        Raises
        ------
        MathError
            If either vector has zero magnitude.
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
        Compare vectors using floating-point tolerance.

        Parameters
        ----------
        other
            Other vector.

        Returns
        -------
        bool
        """
        return is_close(
            self.x,
            other.x,
        ) and is_close(
            self.y,
            other.y,
        )

    # ==========================================================
    # Conversion
    # ==========================================================

    def to_tuple(
        self,
    ) -> tuple[
        float,
        float,
    ]:
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
        Add two vectors.

        Parameters
        ----------
        other
            Vector to add.

        Returns
        -------
        Vector2D
        """
        if not isinstance(other, Vector2D):
            return NotImplemented

        return Vector2D(
            self.x + other.x,
            self.y + other.y,
        )

    def __sub__(
        self,
        other: object,
    ) -> Self:
        """
        Subtract two vectors.

        Parameters
        ----------
        other
            Vector to subtract.

        Returns
        -------
        Vector2D
        """
        if not isinstance(other, Vector2D):
            return NotImplemented

        return Vector2D(
            self.x - other.x,
            self.y - other.y,
        )

    def __mul__(
        self,
        scalar: object,
    ) -> Self:
        """
        Multiply by a scalar.

        Parameters
        ----------
        scalar
            Scalar value.

        Returns
        -------
        Vector2D
        """
        if not isinstance(scalar, (int, float)):
            return NotImplemented

        return Vector2D(
            self.x * float(scalar),
            self.y * float(scalar),
        )

    def __rmul__(
        self,
        scalar: object,
    ) -> Self:
        """
        Right scalar multiplication.
        """
        return self.__mul__(scalar)

    def __truediv__(
        self,
        scalar: object,
    ) -> Self:
        """
        Divide by a scalar.

        Parameters
        ----------
        scalar
            Scalar value.

        Returns
        -------
        Vector2D

        Raises
        ------
        MathError
            If the divisor is numerically zero.
        """
        if not isinstance(scalar, (int, float)):
            return NotImplemented

        value = float(scalar)

        if is_zero(value):
            raise MathError("Division by zero.")

        return Vector2D(
            self.x / value,
            self.y / value,
        )

    def __neg__(
        self,
    ) -> Self:
        """
        Return the opposite vector.

        Returns
        -------
        Vector2D
        """
        return Vector2D(
            -self.x,
            -self.y,
        )

    def __abs__(
        self,
    ) -> float:
        """
        Return the vector magnitude.

        Returns
        -------
        float
        """
        return self.length


__all__ = [
    "Vector2D",
]
