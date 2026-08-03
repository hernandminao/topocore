"""
topocore.processing.registration.point_to_plane
===============================================

Point-to-Plane ICP (Chen & Medioni 1992).

This module implements the point-to-plane ICP algorithm that minimizes
the sum of squared distances from source points to the tangent planes
of their corresponding target points.

This is often more accurate than point-to-point ICP for surface
registration, especially for smooth surfaces.

References
----------
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

from typing import cast, override

import numpy as np
from numpy.typing import NDArray

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import RegistrationError
from topocore.processing.normals import NormalManager
from topocore.processing.types import FloatArray2D

from .base import RegistrationResult, Transformation
from .icp import ICPBase

FloatArray1D = NDArray[np.float64]
FloatArray2DND = NDArray[np.float64]

_MIN_NORMAL_K = 3
_MIN_CORRESPONDENCES = 3
_ROTATION_EPSILON = 1e-12


class PointToPlaneICP(ICPBase):
    """
    Point-to-Plane ICP (Chen & Medioni 1992).

    Minimizes the sum of squared distances from source points to the
    tangent planes of their corresponding target points using linear
    least squares.

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
    normal_k
        Number of neighbors for normal estimation (if normals not present).

    Examples
    --------
    >>> icp = PointToPlaneICP(max_iterations=50)
    >>> result = icp.register(source, target)
    >>> aligned_source = result.source_transformed
    """

    __slots__ = ("_normal_k", "_target_normals")

    def __init__(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        max_correspondence_distance: float = 1.0,
        use_adaptive_distance: bool = True,
        normal_k: int = 10,
    ) -> None:
        super().__init__(
            max_iterations=max_iterations,
            tolerance=tolerance,
            max_correspondence_distance=max_correspondence_distance,
            use_adaptive_distance=use_adaptive_distance,
        )

        if normal_k < _MIN_NORMAL_K:
            msg = f"normal_k must be >= {_MIN_NORMAL_K}, got {normal_k}."
            raise RegistrationError(msg)

        self._normal_k = normal_k
        self._target_normals: FloatArray2DND | None = None

    @override
    def register(
        self,
        source: PointCloud,
        target: PointCloud,
    ) -> RegistrationResult:
        """Register using point-to-plane ICP."""
        self._ensure_normals(target)
        return super().register(source, target)

    @override
    def _estimate_transformation(
        self,
        correspondences: list[tuple[int, int, float]],
        source_points: FloatArray2D,
        target_points: FloatArray2D,
    ) -> Transformation:
        """
        Estimate a rigid transformation using linearized point-to-plane ICP.

        The point-to-plane error is defined as:

            E = sum((n_i^T (p_i - (R q_i + t)))^2)

        where:
        - p_i is a source point,
        - q_i is the corresponding target point,
        - n_i is the normal at q_i.

        The rotation is linearized with a small-angle approximation and the
        resulting least-squares system is solved using ``numpy.linalg.lstsq``.
        """
        n_correspondences = len(correspondences)
        if n_correspondences < _MIN_CORRESPONDENCES:
            msg = f"Need at least {_MIN_CORRESPONDENCES} correspondences, got {n_correspondences}."
            raise RegistrationError(msg)

        normals = self._target_normals
        if normals is None:
            raise RegistrationError("Target normals not available.")

        source_indices = np.fromiter(
            (src_idx for src_idx, _, _ in correspondences),
            dtype=np.intp,
            count=n_correspondences,
        )
        target_indices = np.fromiter(
            (tgt_idx for _, tgt_idx, _ in correspondences),
            dtype=np.intp,
            count=n_correspondences,
        )

        source_matched = source_points[source_indices]
        target_matched = target_points[target_indices]
        target_normals = normals[target_indices]

        system_matrix, rhs = self._build_linear_system(
            source_points=source_matched,
            target_points=target_matched,
            normals=target_normals,
        )

        solution, _, _, _ = np.linalg.lstsq(system_matrix, rhs, rcond=None)
        omega = solution[:3]
        translation = solution[3:]

        rotation = self._rotation_matrix_from_omega(omega)
        return Transformation.from_rotation_translation(rotation, translation)

    @staticmethod
    def _build_linear_system(
        source_points: FloatArray2DND,
        target_points: FloatArray2DND,
        normals: FloatArray2DND,
    ) -> tuple[FloatArray2DND, FloatArray1D]:
        """
        Build the linear system A x = b for point-to-plane ICP.

        For each correspondence, one equation is generated using the
        linearized small-angle formulation.
        """
        cross_terms = np.cross(target_points, normals)
        system_matrix = np.hstack((cross_terms, normals))
        rhs = np.einsum("ij,ij->i", normals, source_points - target_points)

        return (system_matrix, rhs)

    @staticmethod
    def _rotation_matrix_from_omega(omega: FloatArray1D) -> FloatArray2DND:
        """Convert a small rotation vector into a rotation matrix."""
        angle = float(np.linalg.norm(omega))
        if angle <= _ROTATION_EPSILON:
            return cast(FloatArray2DND, np.eye(3, dtype=np.float64))

        axis = omega / angle
        kx, ky, kz = axis

        skew = np.array(
            [
                [0.0, -kz, ky],
                [kz, 0.0, -kx],
                [-ky, kx, 0.0],
            ],
            dtype=np.float64,
        )

        rotation = np.eye(3, dtype=np.float64) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
        return cast(FloatArray2DND, rotation)

    def _ensure_normals(
        self,
        cloud: PointCloud,
    ) -> None:
        """Ensure the target point cloud has normals available."""
        if PointAttribute.NORMAL in cloud.attributes:
            normals_chunks = [chunk[PointAttribute.NORMAL] for chunk in cloud]
            if not normals_chunks:
                raise RegistrationError("Target point cloud has no normal data.")

            self._target_normals = cast(
                FloatArray2DND,
                np.concatenate(normals_chunks, axis=0),
            )
            return

        normal_manager = NormalManager(
            method="pca",
            k=self._normal_k,
            orient_upward=True,
        )
        self._target_normals = normal_manager.estimate(cloud)

    @override
    def name(self) -> str:
        return "point_to_plane_icp"

    @override
    def requires_normals(self) -> bool:
        return True


__all__ = ["PointToPlaneICP"]
