"""
topocore.terrain.weights
========================

Mathematical utilities used by terrain interpolation algorithms.

The functions implemented here are intentionally stateless and reusable
across multiple interpolation methods.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

from topocore.geometry.point3d import Point3D

_EPSILON: float = 1e-12


def triangle_area2(
    p1: Point3D,
    p2: Point3D,
    p3: Point3D,
) -> float:
    """
    Compute twice the signed area of a triangle.

    Parameters
    ----------
    p1
        First vertex.
    p2
        Second vertex.
    p3
        Third vertex.

    Returns
    -------
    float
        Twice the signed area.
    """
    return (
        (p2.x - p1.x) * (p3.y - p1.y)
        - (p3.x - p1.x) * (p2.y - p1.y)
    )


def triangle_area(
    p1: Point3D,
    p2: Point3D,
    p3: Point3D,
) -> float:
    """
    Compute the triangle area.

    Parameters
    ----------
    p1
        First vertex.
    p2
        Second vertex.
    p3
        Third vertex.

    Returns
    -------
    float
        Triangle area.
    """
    return abs(
        triangle_area2(
            p1,
            p2,
            p3,
        )
    ) * 0.5


def barycentric_weights(
    x: float,
    y: float,
    p1: Point3D,
    p2: Point3D,
    p3: Point3D,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Compute barycentric coordinates.

    Parameters
    ----------
    x
        X coordinate.
    y
        Y coordinate.
    p1
        Triangle vertex.
    p2
        Triangle vertex.
    p3
        Triangle vertex.

    Returns
    -------
    tuple[float, float, float]

    Raises
    ------
    ValueError
        If the triangle is degenerate.
    """
    denominator = triangle_area2(
        p1,
        p2,
        p3,
    )

    if abs(denominator) < _EPSILON:
        raise ValueError(
            "Degenerate triangle."
        )

    w1 = (
        (
            (p2.y - p3.y) * (x - p3.x)
            + (p3.x - p2.x) * (y - p3.y)
        )
        / denominator
    )

    w2 = (
        (
            (p3.y - p1.y) * (x - p3.x)
            + (p1.x - p3.x) * (y - p3.y)
        )
        / denominator
    )

    w3 = 1.0 - w1 - w2

    return (
        w1,
        w2,
        w3,
    )


def point_distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """
    Euclidean distance.

    Returns
    -------
    float
    """
    return math.hypot(
        x2 - x1,
        y2 - y1,
    )


def inverse_distance_weight(
    distance: float,
    power: float,
) -> float:
    """
    Compute an inverse-distance weight.

    Parameters
    ----------
    distance
        Distance.
    power
        IDW exponent.

    Returns
    -------
    float
    """
    if distance <= _EPSILON:
        return math.inf

    return 1.0 / (distance**power)


def inside_triangle(
    weights: tuple[
        float,
        float,
        float,
    ],
    tolerance: float = 1e-9,
) -> bool:
    """
    Determine whether barycentric coordinates are inside a triangle.

    Parameters
    ----------
    weights
        Barycentric coordinates.
    tolerance
        Numerical tolerance.

    Returns
    -------
    bool
    """
    w1, w2, w3 = weights

    return (
        w1 >= -tolerance
        and w2 >= -tolerance
        and w3 >= -tolerance
    )


__all__ = [
    "barycentric_weights",
    "inside_triangle",
    "inverse_distance_weight",
    "point_distance",
    "triangle_area",
    "triangle_area2",
]