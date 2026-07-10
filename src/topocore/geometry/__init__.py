"""
topocore.geometry
=================

Geometry primitives used throughout TopoCore.

This package contains immutable geometric objects for two-dimensional
and three-dimensional computations.

Classes
-------
Geometry
    Abstract base class for all geometry objects.

Point2D
    Immutable point in two-dimensional Cartesian space.

Point3D
    Immutable point in three-dimensional Cartesian space.

BBox2D
    Axis-aligned bounding box in 2D.

BBox3D
    Axis-aligned bounding box in 3D.
"""

from .base import Geometry
from .bbox2d import BBox2D
from .bbox3d import BBox3D
from .point2d import Point2D
from .point3d import Point3D

__all__ = [
    "Geometry",
    "Point2D",
    "Point3D",
    "BBox2D",
    "BBox3D",
]
