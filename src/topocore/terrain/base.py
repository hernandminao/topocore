"""
topocore.terrain.base
=====================

Base abstractions for terrain models.

These abstract classes define the public contracts implemented by
TIN, DTM and interpolation algorithms throughout the Terrain module.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np

from topocore.geometry.point3d import Point3D
from topocore.terrain.models import Triangle


class BaseTIN(ABC):
    """
    Base interface implemented by every TIN model.
    """

    @property
    @abstractmethod
    def triangles(self) -> tuple[Triangle, ...]:
        """
        Return the triangles composing the TIN.
        """

    @property
    @abstractmethod
    def points(self) -> tuple[Point3D, ...]:
        """
        Return the vertices used by the TIN.
        """

    @property
    @abstractmethod
    def triangle_count(self) -> int:
        """
        Number of triangles.
        """

    @property
    @abstractmethod
    def point_count(self) -> int:
        """
        Number of vertices.
        """

    @abstractmethod
    def bounds(self) -> tuple[float, float, float, float]:
        """
        Return the XY bounding box.
        """


class BaseDTM(ABC):
    """
    Base interface implemented by every raster terrain model.

    Notes
    -----
    ``grid`` (``topocore.terrain.grid.Grid``) is part of this
    contract but is intentionally *not* declared as an
    ``@abstractmethod`` property here. ``DTM`` implements it as a
    dataclass field of the same name; a same-named
    ``@property @abstractmethod`` on the base class makes Python's
    dataclass machinery treat that inherited property object as a
    default value for the field, which breaks positional field
    ordering in any subclass -- the class fails to even be defined,
    not just to instantiate. This previously also referenced
    ``GridDefinition`` (``topocore.terrain.models.GridDefinition``),
    an unrelated, older grid model that ``DTM`` never actually used.
    Every concrete ``BaseDTM`` implementation is still expected to
    expose a ``grid: Grid`` attribute; it just cannot be enforced
    through the abstract-property mechanism without breaking
    dataclass subclasses.
    """

    @property
    @abstractmethod
    def elevations(self) -> np.ndarray:
        """
        Elevation matrix.
        """

    @property
    @abstractmethod
    def width(self) -> int:
        """
        Grid width, in columns.
        """

    @property
    @abstractmethod
    def height(self) -> int:
        """
        Grid height, in rows.
        """


class BaseInterpolator(ABC):
    """
    Base interface for terrain interpolation algorithms.
    """

    @abstractmethod
    def interpolate(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Interpolate elevation.
        """

    @abstractmethod
    def interpolate_many(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized interpolation.
        """


__all__ = [
    "BaseTIN",
    "BaseDTM",
    "BaseInterpolator",
]
