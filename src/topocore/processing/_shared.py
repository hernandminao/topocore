"""
topocore.processing._shared
===========================

Internal helpers shared across the Processing module.

Sampling, filtering, normal estimation, and segmentation modules
need common primitives:

- Flatten PointCloud chunks into contiguous attribute arrays.
- Rebuild PointCloud instances from selected indices.

Centralizing these operations avoids duplicated logic across
processing algorithms.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from topocore.core.types import FloatArray3D
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.cache import LRUCache
from topocore.processing.exceptions import ProcessingError
from topocore.processing.neighbors import NeighborhoodManager
from topocore.processing.types import FloatArray2D

AttributeData = NDArray[Any]
FlattenedAttributes = dict[PointAttribute, AttributeData]


_REQUIRED_ATTRIBUTES = frozenset(
    {
        PointAttribute.X,
        PointAttribute.Y,
        PointAttribute.Z,
    }
)


@dataclass(slots=True, frozen=True)
class PCAComputation:
    """
    Cached PCA decomposition.
    """

    points: FloatArray2D
    indices: NDArray[np.intp]
    distances: FloatArray2D
    neighbor_points: FloatArray3D
    eigenvalues: FloatArray2D
    eigenvectors: NDArray[np.float64]


_PCA_CACHE: LRUCache[tuple[int, int], PCAComputation] = LRUCache(maxsize=16)


def compute_pca(
    manager: NeighborhoodManager,
    *,
    k: int,
) -> PCAComputation:
    """
    Compute the neighborhood PCA once.
    """
    if k < 3:
        raise ProcessingError(f"k must be at least 3 for PCA, got {k}.")

    cache_key = (id(manager), k)
    cached = _PCA_CACHE.get(cache_key)

    if cached is not None:
        return cached

    points = manager.search.points

    if points.ndim != 2 or points.shape[1] != 3:
        raise ProcessingError(f"Point coordinates must have shape (N, 3), got {points.shape}.")

    point_count = points.shape[0]

    if point_count < k:
        raise ProcessingError(f"Point cloud has {point_count} points, but PCA requires at least {k}.")

    if not np.isfinite(points).all():
        raise ProcessingError("Point cloud coordinates contain NaN or Inf values.")

    try:
        indices, distances = manager.knn_many(
            k=k,
            include_self=True,
        )
    except Exception as exc:
        raise ProcessingError("Failed to compute PCA neighbourhood search.") from exc

    if indices.shape != (point_count, k):
        raise ProcessingError(f"Invalid PCA neighbour index shape: {indices.shape}, expected {(point_count, k)}.")

    if distances.shape != (point_count, k):
        raise ProcessingError(f"Invalid PCA distance shape: {distances.shape}, expected {(point_count, k)}.")

    neighbor_points = points[indices]

    if neighbor_points.shape != (point_count, k, 3):
        raise ProcessingError(
            f"Invalid PCA neighbour points shape: {neighbor_points.shape}, expected {(point_count, k, 3)}."
        )

    if not np.isfinite(neighbor_points).all():
        raise ProcessingError("PCA neighbourhood contains NaN or Inf values.")

    centroids = neighbor_points.mean(axis=1, keepdims=True)
    centered = neighbor_points - centroids

    covariances = np.einsum("nki,nkj->nij", centered, centered) / k

    if covariances.shape != (point_count, 3, 3):
        raise ProcessingError(f"Invalid PCA covariance shape: {covariances.shape}, expected {(point_count, 3, 3)}.")

    if not np.isfinite(covariances).all():
        raise ProcessingError("PCA covariance matrix contains NaN or Inf values.")

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariances)
    except np.linalg.LinAlgError as exc:
        raise ProcessingError("PCA eigendecomposition failed.") from exc

    if eigenvalues.shape != (point_count, 3):
        raise ProcessingError(f"Invalid PCA eigenvalues shape: {eigenvalues.shape}, expected {(point_count, 3)}.")

    if eigenvectors.shape != (point_count, 3, 3):
        raise ProcessingError(f"Invalid PCA eigenvectors shape: {eigenvectors.shape}, expected {(point_count, 3, 3)}.")

    if not np.isfinite(eigenvalues).all():
        raise ProcessingError("PCA eigenvalues contain NaN or Inf values.")

    if not np.isfinite(eigenvectors).all():
        raise ProcessingError("PCA eigenvectors contain NaN or Inf values.")

    result = PCAComputation(
        points=points,
        indices=indices,
        distances=distances,
        neighbor_points=neighbor_points,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )

    _PCA_CACHE.set(cache_key, result)
    return result


def _validate_cloud_attributes(cloud: PointCloud) -> None:
    """
    Validate that a cloud can be processed.
    """
    if not _REQUIRED_ATTRIBUTES.issubset(cloud.attributes):
        raise ProcessingError("Point cloud must contain X/Y/Z coordinates.")


def _validate_chunk_attributes(
    chunk: Chunk,
    expected: frozenset[PointAttribute],
) -> None:
    """
    Ensure that every chunk exposes the same attributes.
    """
    if frozenset(chunk.attributes) != expected:
        raise ProcessingError("All chunks must share the same attribute set.")


def _validate_flattened_attributes(
    flattened: FlattenedAttributes,
) -> int:
    """
    Validate flattened attribute dictionary and return common length.
    """

    if not flattened:
        raise ProcessingError("Flattened attributes cannot be empty.")

    lengths: dict[PointAttribute, int] = {attribute: int(values.shape[0]) for attribute, values in flattened.items()}

    unique_lengths: set[int] = set(
        lengths.values(),
    )

    if len(unique_lengths) != 1:
        raise ProcessingError(f"All flattened attributes must have the same length, got: {lengths}.")

    return next(iter(unique_lengths))


def flatten_attributes(cloud: PointCloud) -> FlattenedAttributes:
    """
    Flatten all PointCloud chunks into attribute arrays.
    """
    if cloud.is_empty:
        raise ProcessingError("Cannot process an empty point cloud.")

    _validate_cloud_attributes(cloud)

    attributes = frozenset(cloud.attributes)

    combined: dict[PointAttribute, list[AttributeData]] = {attribute: [] for attribute in attributes}

    for chunk in cloud:
        _validate_chunk_attributes(chunk, attributes)
        for attribute in attributes:
            combined[attribute].append(chunk[attribute])

    flattened: FlattenedAttributes = {
        attribute: np.concatenate(
            values,
            axis=0,
        )
        for attribute, values in combined.items()
    }

    _validate_flattened_attributes(flattened)
    return flattened


def build_cloud(
    flattened: FlattenedAttributes,
    indices: NDArray[np.intp],
) -> PointCloud:
    """
    Build a single-chunk PointCloud from selected indices.
    """
    if indices.ndim != 1:
        raise ProcessingError("Indices must be a one-dimensional array.")

    if not np.issubdtype(indices.dtype, np.integer):
        raise ProcessingError("Indices must be an integer array.")

    point_count = _validate_flattened_attributes(flattened)

    if indices.size > 0:
        if np.any(indices < 0) or np.any(indices >= point_count):
            raise ProcessingError("Indices contain out-of-range values.")

    attributes = list(flattened.keys())

    chunk = Chunk(size=int(indices.size), attributes=attributes)

    for attribute in attributes:
        chunk[attribute][:] = flattened[attribute][indices]

    cloud = PointCloud()
    cloud.add_chunk(chunk)
    cloud.update_bounds()

    return cloud


def build_cloud_from_mask(
    flattened: FlattenedAttributes,
    mask: NDArray[np.bool_],
) -> PointCloud:
    """
    Build a PointCloud from a boolean selection mask.
    """
    if mask.ndim != 1:
        raise ProcessingError("Mask must be a one-dimensional array.")

    if not np.issubdtype(mask.dtype, np.bool_):
        raise ProcessingError("Mask must be a boolean array.")

    point_count = _validate_flattened_attributes(flattened)

    if mask.shape[0] != point_count:
        raise ProcessingError(f"Mask length {mask.shape[0]} does not match point count {point_count}.")

    indices = np.flatnonzero(mask)
    return build_cloud(flattened, indices)


def extract_attribute(
    cloud: PointCloud,
    attribute: PointAttribute,
) -> AttributeData:
    """
    Extract a single attribute as a flat contiguous array.
    """
    if attribute not in cloud.attributes:
        raise ProcessingError(f"Attribute '{attribute.name}' not found in point cloud.")

    values = [chunk[attribute] for chunk in cloud]

    if not values:
        raise ProcessingError("Cannot extract attribute from an empty point cloud.")

    return np.concatenate(values, axis=0)


def concatenate_clouds(
    clouds: list[PointCloud],
) -> PointCloud:
    """
    Concatenate multiple point clouds into one.
    """
    if not clouds:
        raise ProcessingError("No point clouds to concatenate.")

    attributes = clouds[0].attributes

    result = PointCloud()
    if clouds[0].crs is not None:
        result.crs = clouds[0].crs

    for cloud in clouds:
        if cloud.attributes != attributes:
            raise ProcessingError("Cannot concatenate clouds with different attributes.")

        for chunk in cloud:
            result.add_chunk(chunk)

    result.update_bounds()
    return result


__all__ = [
    "flatten_attributes",
    "build_cloud",
    "build_cloud_from_mask",
    "extract_attribute",
    "concatenate_clouds",
    "PCAComputation",
    "compute_pca",
]
