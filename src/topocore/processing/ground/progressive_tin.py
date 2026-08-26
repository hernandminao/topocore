"""
topocore.processing.ground.progressive_tin
==========================================

Progressive TIN (PTD) ground classification.

This module implements the Progressive TIN Densification (PTD)
algorithm for ground classification, originally proposed by
Axelsson (2000). The algorithm iteratively builds a TIN from
seed ground points and adds points that are close enough to
the TIN surface.

This is one of the most widely used algorithms for ground
classification in LiDAR point clouds.

References
----------
Axelsson, P. (2000). DEM generation from laser scanner data
using adaptive TIN models. International Archives of
Photogrammetry and Remote Sensing, 33(B4/1; PART 4), 110-117.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import NDArray

from topocore.geometry.point3d import Point3D
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import GroundError
from topocore.processing.types import BoolArray1D, FloatArray1D
from topocore.terrain.tin import TIN

from .base import GroundClassifier, GroundExtractor
from .grid import _build_ground_cloud_from_mask, _extract_xyz


class ProgressiveTINGroundClassifier(GroundClassifier):
    """
    Progressive TIN ground classifier.

    This algorithm iteratively builds a TIN from seed ground points
    and adds points that are within a distance threshold of the TIN
    surface.

    Parameters
    ----------
    cell_size
        Size of grid cells for initial seed selection.
    max_distance
        Maximum distance from TIN surface for a point to be accepted.
    max_angle
        Maximum angle between point and TIN surface normal.
    max_iterations
        Maximum number of iterations.
    """

    __slots__ = (
        "_cell_size",
        "_cos_max_angle",
        "_max_angle",
        "_max_distance",
        "_max_iterations",
    )

    def __init__(
        self,
        cell_size: float = 1.0,
        max_distance: float = 0.5,
        max_angle: float = 15.0,
        max_iterations: int = 10,
    ) -> None:
        if cell_size <= 0:
            raise GroundError(f"cell_size must be positive, got {cell_size}.")

        if max_distance <= 0:
            raise GroundError(f"max_distance must be positive, got {max_distance}.")

        if max_angle < 0:
            raise GroundError(f"max_angle must be non-negative, got {max_angle}.")

        if max_iterations < 1:
            raise GroundError(f"max_iterations must be at least 1, got {max_iterations}.")

        self._cell_size = cell_size
        self._max_distance = max_distance

        self._max_angle = np.radians(max_angle)
        self._cos_max_angle = float(np.cos(self._max_angle))

        self._max_iterations = max_iterations

    def _densify_tin(
        self,
        tin: TIN,
        ground_mask: BoolArray1D,
        point_indices: NDArray[np.int64],
        x: FloatArray1D,
        y: FloatArray1D,
        z: FloatArray1D,
    ) -> bool:
        """
        Perform one Progressive TIN densification iteration.

        Parameters
        ----------
        tin
            Current TIN.
        ground_mask
            Ground classification mask.
        point_indices
            Indices of all points.
        x
            X coordinates.
        y
            Y coordinates.
        z
            Elevations.

        Returns
        -------
        bool
            True if at least one new ground point was added.
        """
        added = False

        for idx in point_indices[~ground_mask]:
            triangle = tin.find_triangle(
                x[idx],
                y[idx],
            )

            if triangle < 0:
                continue

            p1, p2, p3 = tin.triangle_vertices(triangle)

            z_surface = self._interpolate_tin(
                p1,
                p2,
                p3,
                x[idx],
                y[idx],
            )

            if abs(z[idx] - z_surface) > self._max_distance:
                continue

            normal = self._compute_normal(
                p1,
                p2,
                p3,
            )

            if not self._check_angle(normal):
                continue

            ground_mask[idx] = True
            added = True

        return added

    @override
    def classify(
        self,
        cloud: PointCloud,
    ) -> BoolArray1D:
        """
        Classify points as ground or non-ground.

        Raises
        ------
        GroundError
            If `cloud` is empty.

        Notes
        -----
        Found and fixed in PR20 coverage phase: this previously
        called `_extract_xyz(cloud)` with no prior empty-cloud
        check, unlike every sibling ground classifier in this same
        package (`PMFGroundClassifier`, `CSFGroundClassifier`), both
        of which cleanly reject an empty cloud with `GroundError`.
        Confirmed directly: an empty `PointCloud` crashed here with
        a raw, unhandled `ValueError` ("need at least one array to
        concatenate") from inside `_extract_xyz`, breaking the
        consistent `GroundError` contract every other classifier in
        this package honors.
        """
        if cloud.is_empty:
            raise GroundError("Cannot classify an empty point cloud.")

        x, y, z = _extract_xyz(cloud)

        seed_mask = self._get_seeds(
            x,
            y,
            z,
        )

        if not seed_mask.any():
            raise GroundError("No seed points found. Try increasing cell_size.")

        ground_mask = seed_mask.copy()

        point_indices = np.arange(
            len(x),
            dtype=np.int64,
        )

        for _ in range(self._max_iterations):
            ground_indices = point_indices[ground_mask]

            if ground_indices.size < 3:
                break

            try:
                tin = TIN.from_points(
                    [
                        Point3D(
                            x[i],
                            y[i],
                            z[i],
                        )
                        for i in ground_indices
                    ]
                )
            except Exception:
                break

            if not self._densify_tin(
                tin,
                ground_mask,
                point_indices,
                x,
                y,
                z,
            ):
                break

        return ground_mask

    def _get_seeds(
        self,
        x: FloatArray1D,
        y: FloatArray1D,
        z: FloatArray1D,
    ) -> BoolArray1D:
        """
        Select initial ground seed points.

        The lowest point of every grid cell is selected as an initial
        ground seed.

        Parameters
        ----------
        x
            X coordinates.
        y
            Y coordinates.
        z
            Elevations.

        Returns
        -------
        BoolArray1D
            Boolean mask indicating seed points.
        """
        cell_i = np.floor(x / self._cell_size).astype(np.int64)
        cell_j = np.floor(y / self._cell_size).astype(np.int64)

        cell_coords = np.column_stack((cell_i, cell_j))

        _, inverse = np.unique(
            cell_coords,
            axis=0,
            return_inverse=True,
        )

        seed_mask = np.zeros(
            len(z),
            dtype=bool,
        )

        for cell in np.unique(inverse):
            indices = np.flatnonzero(inverse == cell)
            seed_mask[indices[np.argmin(z[indices])]] = True

        return seed_mask

    def _interpolate_tin(
        self,
        p1: Point3D,
        p2: Point3D,
        p3: Point3D,
        x: float,
        y: float,
    ) -> float:
        """
        Interpolate TIN elevation using barycentric coordinates.
        """
        denominator = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)

        if np.isclose(
            denominator,
            0.0,
        ):
            return p1.z

        w1 = (((p2.x - x) * (p3.y - y)) - ((p3.x - x) * (p2.y - y))) / denominator

        w2 = (((p3.x - x) * (p1.y - y)) - ((p1.x - x) * (p3.y - y))) / denominator

        w3 = 1.0 - w1 - w2

        return w1 * p1.z + w2 * p2.z + w3 * p3.z

    def _compute_normal(
        self,
        p1: Point3D,
        p2: Point3D,
        p3: Point3D,
    ) -> FloatArray1D:
        """
        Compute the normalized normal vector of a TIN triangle.

        Parameters
        ----------
        p1
            First triangle vertex.
        p2
            Second triangle vertex.
        p3
            Third triangle vertex.

        Returns
        -------
        FloatArray1D
            Unit normal vector.
        """
        v1 = np.array(
            (
                p2.x - p1.x,
                p2.y - p1.y,
                p2.z - p1.z,
            ),
            dtype=np.float64,
        )

        v2 = np.array(
            (
                p3.x - p1.x,
                p3.y - p1.y,
                p3.z - p1.z,
            ),
            dtype=np.float64,
        )

        normal = np.cross(v1, v2)

        norm = float(np.linalg.norm(normal))

        if norm <= np.finfo(np.float64).eps:
            return normal

        return normal / norm

    def _check_angle(
        self,
        normal: FloatArray1D,
    ) -> bool:
        """
        Check whether the terrain slope satisfies the configured angle.

        Parameters
        ----------
        normal
            Unit normal vector.

        Returns
        -------
        bool
            True if the triangle slope is acceptable.
        """
        return bool(normal[2] >= self._cos_max_angle)

    @override
    def name(self) -> str:
        return "progressive_tin"


class ProgressiveTINGroundExtractor(GroundExtractor):
    """
    Progressive TIN ground extractor.

    Extracts ground points using the Progressive TIN classifier.

    Parameters
    ----------
    cell_size
        Size of grid cells for initial seed selection.
    max_distance
        Maximum distance from TIN surface for a point to be accepted.
    max_angle
        Maximum angle between point and TIN surface normal.
    max_iterations
        Maximum number of iterations.
    """

    __slots__ = ("_classifier",)

    def __init__(
        self,
        cell_size: float = 1.0,
        max_distance: float = 0.5,
        max_angle: float = 15.0,
        max_iterations: int = 10,
    ) -> None:
        self._classifier = ProgressiveTINGroundClassifier(
            cell_size,
            max_distance,
            max_angle,
            max_iterations,
        )

    @override
    def extract(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """Extract ground points."""
        mask = self._classifier.classify(cloud)

        if not mask.any():
            raise GroundError("No ground points found. Try adjusting parameters.")

        return _build_ground_cloud_from_mask(cloud, mask)

    @override
    def name(self) -> str:
        return "progressive_tin"


__all__ = [
    "ProgressiveTINGroundClassifier",
    "ProgressiveTINGroundExtractor",
]
