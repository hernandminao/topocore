"""
topocore.processing.registration.icp
====================================

Base ICP (Iterative Closest Point) implementation.

This module provides the common infrastructure for ICP-based registration
algorithms. It handles:
- Iterative refinement loop
- Convergence criteria
- Correspondence matching
- Error evaluation

Concrete ICP variants (point-to-point, point-to-plane, GICP, etc.)
extend this base class and implement their own transformation estimation.

References
----------
- Besl, P.J. and McKay, N.D. (1992). A method for registration
  of 3-D shapes. IEEE TPAMI, 14(2), 239-256.
- Chen, Y. and Medioni, G. (1992). Object modeling by registration
  of multiple range images. IEEE ICRA, 2724-2729.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import abstractmethod
from typing import override

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import RegistrationError
from topocore.processing.neighbors import NeighborhoodManager
from topocore.processing.types import FloatArray2D

from .base import Registrar, RegistrationResult, Transformation


class ICPBase(Registrar):
    """
    Base class for ICP-based registration algorithms.

    This class implements the common ICP iteration loop and delegates
    the transformation estimation to concrete subclasses.

    Parameters
    ----------
    max_iterations
        Maximum number of iterations.
    tolerance
        Convergence tolerance (relative change in RMSE).
    max_correspondence_distance
        Maximum distance for a point to be considered a match.
    use_adaptive_distance
        If True, the correspondence distance is reduced over iterations.
    """

    __slots__ = (
        "_max_correspondence_distance",
        "_max_iterations",
        "_source_transformed",
        "_tolerance",
        "_use_adaptive_distance",
    )

    def __init__(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        max_correspondence_distance: float = 1.0,
        use_adaptive_distance: bool = True,
    ) -> None:
        if max_iterations < 1:
            raise RegistrationError(f"max_iterations must be >= 1, got {max_iterations}.")

        if tolerance < 0:
            raise RegistrationError(f"tolerance must be non-negative, got {tolerance}.")

        if max_correspondence_distance <= 0:
            raise RegistrationError(f"max_correspondence_distance must be positive, got {max_correspondence_distance}.")

        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._max_correspondence_distance = max_correspondence_distance
        self._use_adaptive_distance = use_adaptive_distance
        self._source_transformed: PointCloud | None = None

    @override
    def register(
        self,
        source: PointCloud,
        target: PointCloud,
    ) -> RegistrationResult:
        """
        Register the source point cloud to the target using ICP.

        Parameters
        ----------
        source
            Source point cloud (to be transformed).
        target
            Target point cloud (fixed reference).

        Returns
        -------
        RegistrationResult
            The registration result.

        Raises
        ------
        RegistrationError
            If registration fails or input clouds are invalid.
        """

        self._validate_inputs(source, target)

        # Extract point coordinates once
        source_points = self._extract_points(source)
        target_points = self._extract_points(target)

        # Working copy of source coordinates
        current_points = source_points.copy()

        current_transformation = Transformation.identity()

        previous_error = float("inf")
        error = float("inf")

        max_distance = self._max_correspondence_distance

        target_manager = NeighborhoodManager.from_point_cloud(target)

        converged = False
        iterations = 0

        for iteration in range(self._max_iterations):
            iterations = iteration + 1

            # Find correspondences
            correspondences = self._find_correspondences(
                current_points,
                target_manager,
                max_distance,
            )

            if len(correspondences) < 3:
                raise RegistrationError(
                    f"Not enough correspondences ({len(correspondences)}) found. "
                    "Try increasing max_correspondence_distance."
                )

            # Estimate incremental transformation
            transformation = self._estimate_transformation(
                correspondences,
                current_points,
                target_points,
            )

            # Apply transformation directly to coordinates
            current_points = transformation.apply_points(current_points)

            # Accumulate global transformation
            current_transformation = transformation.compose(current_transformation)

            # Recompute correspondences and RMSE
            correspondences = self._find_correspondences(
                current_points,
                target_manager,
                max_distance,
            )

            error = self._compute_rmse(correspondences)

            if iteration > 0:
                relative_change = abs(previous_error - error) / max(
                    1e-12,
                    abs(previous_error),
                )

                # Converged if EITHER the relative change is small, OR
                # the RMSE itself is already below tolerance. The
                # relative-only check alone is a real bug (found and
                # fixed in PR19): once RMSE reaches the floating-point
                # noise floor (e.g. a near-perfect alignment converges
                # to RMSE ~1e-14), further iterations only move it by
                # rounding noise (~1e-16) -- but relative_change
                # divides that noise by an already-tiny previous_error,
                # producing values like 1e-3 to 1e-4 that never drop
                # below a typical tolerance (1e-6), so `converged`
                # stayed False and every max_iterations budget was
                # burned even on a perfect fit. Confirmed directly: a
                # synthetic registration with a known, exactly
                # recoverable transformation reached RMSE ~2e-14 by
                # iteration 35 of 50, yet relative_change fluctuated
                # between 1e-4 and 1e-3 for the remaining iterations,
                # never triggering convergence despite the transformation
                # itself already being correct to machine precision.
                if relative_change < self._tolerance or error < self._tolerance:
                    converged = True
                    break

            previous_error = error

            if self._use_adaptive_distance:
                max_distance = max(
                    0.1,
                    max_distance * 0.95,
                )

        # Build transformed point cloud only once
        transformed_cloud = self._clone_cloud(source)
        self._replace_points(
            transformed_cloud,
            current_points,
        )

        self._source_transformed = transformed_cloud

        fitness = self._compute_fitness(
            transformed_cloud,
            target,
        )

        return RegistrationResult(
            transformation=current_transformation,
            fitness=fitness,
            rmse=error if iterations > 0 else float("inf"),
            iterations=iterations,
            converged=converged,
            source_transformed=transformed_cloud,
        )

    def _replace_points(
        self,
        cloud: PointCloud,
        points: FloatArray2D,
    ) -> None:
        """
        Replace X/Y/Z coordinates of a point cloud.

        Parameters
        ----------
        cloud
            Cloud to update.

        points
            Coordinates of shape (N, 3).
        """

        offset = 0

        for chunk in cloud:
            size = chunk.size

            chunk[PointAttribute.X][:] = points[offset : offset + size, 0]
            chunk[PointAttribute.Y][:] = points[offset : offset + size, 1]
            chunk[PointAttribute.Z][:] = points[offset : offset + size, 2]

            offset += size

        cloud.update_bounds()

    def _validate_inputs(
        self,
        source: PointCloud,
        target: PointCloud,
    ) -> None:
        """Validate input point clouds."""

        required = {PointAttribute.X, PointAttribute.Y, PointAttribute.Z}

        if source.is_empty:
            raise RegistrationError("Source point cloud is empty.")

        if target.is_empty:
            raise RegistrationError("Target point cloud is empty.")

        if not required.issubset(set(source.attributes)):
            raise RegistrationError("Source point cloud must contain X, Y, and Z coordinates.")

        if not required.issubset(set(target.attributes)):
            raise RegistrationError("Target point cloud must contain X, Y, and Z coordinates.")

    def _clone_cloud(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """Create a shallow clone of a point cloud."""
        from topocore.pointcloud.chunk import Chunk

        new_cloud = PointCloud()
        for chunk in cloud:
            attributes = list(chunk.attributes)
            new_chunk = Chunk(
                size=chunk.size,
                attributes=attributes,
                source_id=chunk.source_id,
            )
            for attr in attributes:
                new_chunk[attr][:] = chunk[attr]
            new_cloud.add_chunk(new_chunk)

        new_cloud.update_bounds()
        return new_cloud

    def _extract_points(
        self,
        cloud: PointCloud,
    ) -> FloatArray2D:
        """Extract X/Y/Z coordinates from a point cloud."""
        xs, ys, zs = [], [], []
        for chunk in cloud:
            xs.append(chunk[PointAttribute.X])
            ys.append(chunk[PointAttribute.Y])
            zs.append(chunk[PointAttribute.Z])

        x = np.concatenate(xs)
        y = np.concatenate(ys)
        z = np.concatenate(zs)

        return np.column_stack((x, y, z))

    def _find_correspondences(
        self,
        source_points: FloatArray2D,
        target_manager: NeighborhoodManager,
        max_distance: float,
    ) -> list[tuple[int, int, float]]:
        """
        Find correspondences between source and target points.

        PR21.8 (transversal audit): replaced a genuine per-point
        query_point() loop -- repeated on EVERY ICP iteration, since
        correspondences are recomputed each time -- with one batched
        query_points_many() call. The `len(indices) == 0` guard the
        old loop had is not reproduced: target_manager is always
        built from a non-empty point set (NeighborhoodManager raises
        for zero points at construction), so query_point(k=1) could
        never actually return zero results either way -- confirmed
        this was unreachable defensive code, not a real behavior to
        preserve. Numerically identical (source_idx, target_idx,
        distance) tuples, in the same ascending source-index order,
        verified directly before this change.

        Returns
        -------
        list
            List of (source_idx, target_idx, distance) tuples.
        """
        indices, distances = target_manager.query_points_many(source_points, k=1)

        within_max_distance = np.flatnonzero(distances[:, 0] <= max_distance)

        return [
            (int(source_index), int(indices[source_index, 0]), float(distances[source_index, 0]))
            for source_index in within_max_distance
        ]

    @abstractmethod
    def _estimate_transformation(
        self,
        correspondences: list[tuple[int, int, float]],
        source_points: FloatArray2D,
        target_points: FloatArray2D,
    ) -> Transformation:
        """
        Estimate transformation from correspondences.

        This method must be implemented by concrete ICP variants.

        Parameters
        ----------
        correspondences
            List of (source_idx, target_idx, distance) tuples.
        source_points
            Source point coordinates (N, 3).
        target_points
            Target point coordinates (N, 3).

        Returns
        -------
        Transformation
            The estimated transformation.
        """
        ...

    def _compute_rmse(
        self,
        correspondences: list[tuple[int, int, float]],
    ) -> float:
        """Compute RMSE from correspondences."""
        if not correspondences:
            return float("inf")

        squared_errors = [d * d for _, _, d in correspondences]
        return float(np.sqrt(np.mean(squared_errors)))

    def _compute_fitness(
        self,
        source: PointCloud,
        target: PointCloud,
    ) -> float:
        """
        Compute fitness score (fraction of source points with correspondences).

        Fitness is the fraction of points in the source that have a
        valid correspondence in the target.
        """
        source_points = self._extract_points(source)
        target_manager = NeighborhoodManager.from_point_cloud(target)

        matched = 0
        for point in source_points:
            indices, distances = target_manager.query_point(
                point[0],
                point[1],
                point[2],
                k=1,
            )
            if len(indices) > 0 and distances[0] <= self._max_correspondence_distance:
                matched += 1

        return matched / len(source_points) if len(source_points) > 0 else 0.0

    @override
    def name(self) -> str:
        return "icp_base"

    @override
    def requires_normals(self) -> bool:
        return False


__all__ = [
    "ICPBase",
]
