"""
topocore.processing.registration.point_to_point
===============================================

Point-to-Point ICP (Besl & McKay, 1992).

This module implements the classic point-to-point ICP algorithm, which
minimizes the sum of squared distances between corresponding points.

The rigid transformation is estimated using Singular Value Decomposition
(SVD) on the covariance matrix of matched point pairs.

References
----------
- Besl, P. J., and McKay, N. D. (1992). A method for registration
  of 3-D shapes. IEEE TPAMI, 14(2), 239-256.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Final, override

import numpy as np

from topocore.processing.exceptions import RegistrationError
from topocore.processing.types import FloatArray2D

from .base import Transformation
from .icp import ICPBase

_MIN_CORRESPONDENCES: Final[int] = 3


class PointToPointICP(ICPBase):
    """
    Point-to-Point ICP (Besl & McKay, 1992).

    Minimizes the sum of squared distances between corresponding points
    using SVD.

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

    Examples
    --------
    >>> icp = PointToPointICP(max_iterations=50)
    >>> result = icp.register(source, target)
    >>> aligned_source = result.source_transformed
    """

    @override
    def _estimate_transformation(
        self,
        correspondences: list[tuple[int, int, float]],
        source_points: FloatArray2D,
        target_points: FloatArray2D,
    ) -> Transformation:
        """
        Estimate a rigid transformation from correspondences using SVD.

        The algorithm is:
        1. Compute centroids of matched source and target points.
        2. Center the matched points around their centroids.
        3. Compute the covariance matrix.
        4. Perform SVD on the covariance matrix.
        5. Compute the optimal rotation.
        6. Compute the translation.

        Parameters
        ----------
        correspondences
            Matched point pairs as (source_index, target_index, distance).
        source_points
            Source point cloud as an array of shape (N, 3).
        target_points
            Target point cloud as an array of shape (M, 3).

        Returns
        -------
        Transformation
            Estimated rigid transformation.

        Raises
        ------
        RegistrationError
            If there are fewer than 3 correspondences or if the SVD-based
            estimation fails due to degenerate input.
        """
        if len(correspondences) < _MIN_CORRESPONDENCES:
            raise RegistrationError(
                f"At least {_MIN_CORRESPONDENCES} correspondences are required "
                f"to estimate a rigid transformation, but got {len(correspondences)}."
            )

        matched_source_points, matched_target_points = self._extract_matched_points(
            correspondences=correspondences,
            source_points=source_points,
            target_points=target_points,
        )

        source_centroid = matched_source_points.mean(axis=0)
        target_centroid = matched_target_points.mean(axis=0)

        centered_source = matched_source_points - source_centroid
        centered_target = matched_target_points - target_centroid

        covariance_matrix = centered_source.T @ centered_target

        try:
            left_vectors, _, right_vectors_transposed = np.linalg.svd(covariance_matrix)
        except np.linalg.LinAlgError as error:
            raise RegistrationError("SVD failed while estimating the point-to-point transformation.") from error

        rotation = right_vectors_transposed.T @ left_vectors.T

        if np.linalg.det(rotation) < 0.0:
            corrected_right_vectors_transposed = right_vectors_transposed.copy()
            corrected_right_vectors_transposed[-1, :] *= -1.0
            rotation = corrected_right_vectors_transposed.T @ left_vectors.T

        translation = target_centroid - rotation @ source_centroid

        return Transformation.from_rotation_translation(rotation, translation)

    @staticmethod
    def _extract_matched_points(
        correspondences: list[tuple[int, int, float]],
        source_points: FloatArray2D,
        target_points: FloatArray2D,
    ) -> tuple[FloatArray2D, FloatArray2D]:
        """
        Extract matched source and target points from correspondence indices.

        Parameters
        ----------
        correspondences
            Matched point pairs as (source_index, target_index, distance).
        source_points
            Source point cloud.
        target_points
            Target point cloud.

        Returns
        -------
        tuple[FloatArray2D, FloatArray2D]
            Matched source points and matched target points.
        """
        source_indices = np.fromiter(
            (c[0] for c in correspondences),
            dtype=np.intp,
            count=len(correspondences),
        )

        target_indices = np.fromiter(
            (c[1] for c in correspondences),
            dtype=np.intp,
            count=len(correspondences),
        )

        matched_source_points = source_points[source_indices]
        matched_target_points = target_points[target_indices]

        return matched_source_points, matched_target_points

    @override
    def name(self) -> str:
        """Return the algorithm name."""
        return "point_to_point_icp"


__all__ = ["PointToPointICP"]
