"""
topocore.geometry.protocols
==============================

Structural typing protocols shared across geometry primitives.

Author
------
Hernan Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Protocol

from topocore.geometry.bbox2d import BBox2D
from topocore.geometry.bbox3d import BBox3D
from topocore.geometry.point2d import Point2D
from topocore.geometry.point3d import Point3D


class HasArea(Protocol):
    @property
    def area(self) -> float: ...


class HasLength(Protocol):
    @property
    def length(self) -> float: ...


class HasVolume(Protocol):
    @property
    def volume(self) -> float: ...


class HasCentroid(Protocol):
    @property
    def centroid(self) -> Point2D | Point3D: ...


class Bounded(Protocol):
    def bounding_box(self) -> BBox2D | BBox3D: ...


class Serializable(Protocol):
    def to_wkt(self) -> str: ...


__all__ = [
    "HasArea",
    "HasLength",
    "HasVolume",
    "HasCentroid",
    "Bounded",
    "Serializable",
]
