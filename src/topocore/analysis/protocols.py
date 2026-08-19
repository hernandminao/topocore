"""
topocore.analysis.protocols
============================

Protocols (structural typing) for the analysis subsystem.

These protocols define the expected interfaces for analysis components,
enabling flexible and decoupled implementations. They are used to enforce
contracts and facilitate dependency injection.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from topocore.geometry.point3d import Point3D

# ============================================================================
# Generic types
# ============================================================================

T_co = TypeVar("T_co", covariant=True)
R_co = TypeVar("R_co", covariant=True)


# ============================================================================
# Terrain protocols
# ============================================================================


class TerrainSurface(Protocol):
    """
    Protocol for terrain surfaces.

    Implementations may represent TIN, DTM, Grid or Raster surfaces.
    """

    def interpolate(
        self,
        x: float,
        y: float,
    ) -> float:
        """Interpolate terrain elevation."""
        ...

    def contains(
        self,
        x: float,
        y: float,
    ) -> bool:
        """Check whether coordinates are covered by the surface."""
        ...


class TriangulatedSurface(TerrainSurface, Protocol):
    """
    Protocol for triangulated terrain surfaces.
    """

    @property
    def triangle_count(self) -> int:
        """Number of triangles."""
        ...

    @property
    def bounds(
        self,
    ) -> tuple[float, float, float, float]:
        """Surface bounding box."""
        ...

    def triangle_vertices(
        self,
        index: int,
    ) -> tuple[
        Point3D,
        Point3D,
        Point3D,
    ]:
        """
        Return vertices of a triangle.

        Parameters
        ----------
        index:
            Triangle index.
        """
        ...

    def find_triangle(
        self,
        x: float,
        y: float,
    ) -> int:
        """Find triangle containing XY coordinates."""
        ...


class GriddedSurface(TerrainSurface, Protocol):
    """
    Protocol for gridded terrain surfaces.
    """

    @property
    def grid(self) -> NDArray[np.float64]:
        """Elevation grid."""
        ...

    @property
    def resolution(self) -> float:
        """Grid resolution."""
        ...

    @property
    def elevations(self) -> NDArray[np.float64]:
        """Elevation values."""
        ...


# ============================================================================
# Point cloud protocols
# ============================================================================


class PointCloudData(Protocol):
    """
    Protocol for point cloud data.

    Provides access to common coordinate arrays used by analysis
    algorithms.
    """

    @property
    def xy_array(self) -> NDArray[np.float64]:
        """Return XY coordinates."""
        ...

    @property
    def elevation_array(self) -> NDArray[np.float64]:
        """Return elevation values."""
        ...

    @property
    def array(self) -> NDArray[np.float64]:
        """
        Return underlying point data array.

        The concrete layout depends on the PointCloud implementation.
        """
        ...


# ============================================================================
# Coordinate protocols
# ============================================================================


class CoordinateTransformer(Protocol):
    """
    Protocol for coordinate transformations.
    """

    def transform_point(
        self,
        x: float,
        y: float,
        z: float | None = None,
    ) -> tuple[float, float, float | None]:
        """Transform a single coordinate."""
        ...


class CRSType(Protocol):
    """
    Protocol for Coordinate Reference Systems.
    """

    @property
    def ellipsoid(self) -> Any:
        """Reference ellipsoid."""
        ...

    @property
    def epsg(self) -> int | None:
        """EPSG authority code."""
        ...


# ============================================================================
# Generic analysis protocols
# ============================================================================


class Measurable(Protocol[T_co]):
    """
    Protocol for components computing scalar measurements.
    """

    def compute(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> T_co:
        """Compute a measurement."""
        ...

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> T_co: ...


class Calculable(Protocol[R_co]):
    """
    Protocol for components computing structured results.
    """

    def compute(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> R_co:
        """Compute a structured result."""
        ...

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> R_co: ...


class Generable(Protocol[R_co]):
    """
    Protocol for components generating results.
    """

    def generate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> R_co:
        """Generate results."""
        ...

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> R_co: ...


__all__ = [
    "CRSType",
    "Calculable",
    "CoordinateTransformer",
    "Generable",
    "GriddedSurface",
    "Measurable",
    "PointCloudData",
    "TerrainSurface",
    "TriangulatedSurface",
]
