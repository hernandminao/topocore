"""
topocore.processing.registration.base
=====================================

Base contracts for point cloud registration.

This module defines the core abstractions for point cloud registration:
- Transformation: 3D rigid transformation (rotation + translation)
- RegistrationResult: Result of a registration operation
- Registrar: Abstract interface for registration algorithms

All registration algorithms in TopoCore implement the Registrar interface,
ensuring consistent usage across different methods.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import RegistrationError
from topocore.processing.types import TransformationMatrix


@dataclass(frozen=True, slots=True)
class Transformation:
    """
    Immutable 3D rigid transformation.

    Represents a rotation and translation in 3D space.

    Parameters
    ----------
    matrix
        4x4 homogeneous transformation matrix.

    Notes
    -----
    The matrix is expected to be of the form:
        [R  t]
        [0  1]
    where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
    """

    matrix: TransformationMatrix

    def __post_init__(self) -> None:
        """Validate the transformation matrix."""
        if not np.isfinite(self.matrix).all():
            raise RegistrationError("Transformation matrix contains NaN or Inf.")

        if self.matrix.shape != (4, 4):
            raise RegistrationError(f"Transformation matrix must be (4, 4), got {self.matrix.shape}.")

        # Check that the matrix is rigid (rotation part is orthogonal)
        rotation = self.matrix[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise RegistrationError("Rotation matrix is not orthogonal.")

        # Check that the rotation is proper (determinant = +1)
        det = np.linalg.det(rotation)
        if not np.isclose(det, 1.0, atol=1e-6):
            raise RegistrationError(f"Rotation matrix must have determinant +1, got {det:.6f}.")

        # Check that the bottom row is [0, 0, 0, 1]
        if not np.allclose(self.matrix[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise RegistrationError("Transformation matrix bottom row must be [0, 0, 0, 1].")

    @classmethod
    def identity(cls) -> Transformation:
        """Return the identity transformation."""
        return cls(np.eye(4, dtype=np.float64))

    @classmethod
    def from_rotation_translation(
        cls,
        rotation: NDArray[np.float64],
        translation: NDArray[np.float64],
    ) -> Transformation:
        """
        Create a transformation from rotation and translation.

        Parameters
        ----------
        rotation
            3x3 rotation matrix.
        translation
            3x1 translation vector.

        Returns
        -------
        Transformation
        """

        if rotation.shape != (3, 3):
            raise RegistrationError(f"Rotation must be (3, 3), got {rotation.shape}.")

        translation = np.asarray(translation, dtype=np.float64).reshape(-1)
        if translation.shape != (3,):
            raise RegistrationError(f"Translation must be (3,), got {translation.shape}.")

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation

        return cls(matrix)

    @property
    def rotation(self) -> NDArray[np.float64]:
        """Return the rotation matrix (3x3)."""
        return self.matrix[:3, :3].copy()

    @property
    def translation(self) -> NDArray[np.float64]:
        """Return the translation vector (3,)."""
        return self.matrix[:3, 3].copy()

    def apply(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """
        Apply the transformation to a point cloud.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        PointCloud
            Transformed point cloud.
        """

        # Create a new point cloud with transformed coordinates
        new_cloud = PointCloud()

        for chunk in cloud:
            # Extract coordinates
            x = chunk[PointAttribute.X]
            y = chunk[PointAttribute.Y]
            z = chunk[PointAttribute.Z]

            points = np.column_stack((x, y, z))
            transformed = self.apply_points(points)

            # Create new chunk
            attributes = list(chunk.attributes)
            new_chunk = Chunk(size=chunk.size, attributes=attributes, source_id=chunk.source_id)

            # Copy transformed coordinates
            new_chunk[PointAttribute.X][:] = transformed[:, 0]
            new_chunk[PointAttribute.Y][:] = transformed[:, 1]
            new_chunk[PointAttribute.Z][:] = transformed[:, 2]

            # Copy other attributes
            for attr in attributes:
                if attr not in (PointAttribute.X, PointAttribute.Y, PointAttribute.Z):
                    new_chunk[attr][:] = chunk[attr]

            new_cloud.add_chunk(new_chunk)

        new_cloud.update_bounds()
        return new_cloud

    def inverse(self) -> Transformation:
        """
        Return the inverse transformation.

        Returns
        -------
        Transformation
        """
        rotation = self.matrix[:3, :3]
        translation = self.matrix[:3, 3]

        rotation_inv = rotation.T

        inv_matrix = np.eye(4, dtype=np.float64)
        inv_matrix[:3, :3] = rotation_inv
        inv_matrix[:3, 3] = -rotation_inv @ translation

        return Transformation(inv_matrix)

    def apply_points(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Apply the transformation to an array of points.

        Parameters
        ----------
        points
            Array of shape (N, 3).

        Returns
        -------
        NDArray[np.float64]
            Transformed points.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise RegistrationError(f"Points must have shape (N, 3), got {points.shape}.")

        # Apply transformation
        rotation = self.matrix[:3, :3]
        translation = self.matrix[:3, 3]

        transformed = points @ rotation.T
        transformed += translation

        return transformed

    def compose(
        self,
        other: Transformation,
    ) -> Transformation:
        """
        Compose this transformation with another.

        Returns
        -------
        Transformation
            self ∘ other (apply other first, then self).
        """
        return Transformation(self.matrix @ other.matrix)

    def __matmul__(
        self,
        other: Transformation,
    ) -> Transformation:
        """Compose transformations (self @ other)."""
        return self.compose(other)

    def __repr__(self) -> str:
        return f"Transformation(\n  rotation={self.rotation.tolist()},\n  translation={self.translation.tolist()}\n)"


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    """
    Result of a point cloud registration operation.

    Attributes
    ----------
    transformation
        The estimated transformation from source to target.
    fitness
        Fraction of points in the source that have a valid correspondence
        in the target (in [0, 1]).
    rmse
        Root Mean Square Error of the alignment.
    iterations
        Number of iterations performed.
    converged
        Whether the algorithm converged.
    source_transformed
        The transformed source point cloud (optional).
    """

    transformation: Transformation
    fitness: float
    rmse: float
    iterations: int
    converged: bool
    source_transformed: PointCloud | None = None

    @property
    def has_source_transformed(self) -> bool:
        """Return whether the transformed source cloud is available."""
        return self.source_transformed is not None

    def __repr__(self) -> str:
        return (
            f"RegistrationResult(\n"
            f"  fitness={self.fitness:.4f},\n"
            f"  rmse={self.rmse:.4f},\n"
            f"  iterations={self.iterations},\n"
            f"  converged={self.converged}\n"
            f")"
        )

    def __post_init__(self) -> None:
        """Validate registration metrics."""

        if not (0.0 <= self.fitness <= 1.0):
            raise RegistrationError(f"Fitness must be in [0, 1], got {self.fitness}.")

        if self.rmse < 0.0:
            raise RegistrationError(f"RMSE must be non-negative, got {self.rmse}.")

        if self.iterations < 0:
            raise RegistrationError(f"Iterations must be non-negative, got {self.iterations}.")


class Registrar(ABC):
    """
    Abstract interface for point cloud registration algorithms.

    All registration algorithms must implement this interface.

    Examples
    --------
    >>> registrar = PointToPointICP(max_iterations=50)
    >>> result = registrar.register(source, target)
    >>> aligned_source = result.source_transformed
    """

    @abstractmethod
    def register(
        self,
        source: PointCloud,
        target: PointCloud,
    ) -> RegistrationResult:
        """
        Register the source point cloud to the target.

        Parameters
        ----------
        source
            Source point cloud (to be transformed).
        target
            Target point cloud (fixed reference).

        Returns
        -------
        RegistrationResult
            The registration result including transformation and metrics.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the name of the registration algorithm."""
        ...

    @abstractmethod
    def requires_normals(self) -> bool:
        """Return whether this algorithm requires normals."""
        ...

    def __call__(
        self,
        source: PointCloud,
        target: PointCloud,
    ) -> RegistrationResult:
        """Callable interface."""
        return self.register(source, target)


__all__ = [
    "Transformation",
    "RegistrationResult",
    "Registrar",
]
