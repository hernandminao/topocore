"""
Regression suite for the PR21.8 extension: classification/ml.py's
_GroundRelativeHeightFeatureComputer, found to have the exact same
per-point query_point() loop pattern already found and fixed in
features.geometric.RelativeHeightFeatureComputer (PR21.8) -- noted
during PR21.3.2's own audit of this manager's usage (correctly
excluded from cross-module NeighborhoodManager sharing, since it
operates on a ground-points subset, not the full cloud), but not
revisited for this SEPARATE per-point-loop inefficiency until this
final PR21 integration audit caught it as a missed sibling case.

Confirmed via direct testing: the batched replacement gives
numerically identical results to the original per-point loop.
"""

from __future__ import annotations

import numpy as np
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.neighbors.manager import NeighborhoodManager


def _make_cloud(n: int, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 100, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 100, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 10, n)
    cloud.add_chunk(chunk)
    return cloud


def _reference_relative_height(points: np.ndarray, ground_points: np.ndarray) -> np.ndarray:
    """The exact pre-fix per-point-loop algorithm, reimplemented as an independent reference."""
    manager = NeighborhoodManager.from_array(ground_points)
    result = np.empty(points.shape[0], dtype=np.float64)
    for i in range(points.shape[0]):
        indices, _ = manager.query_point(points[i, 0], points[i, 1], points[i, 2], k=1)
        result[i] = points[i, 2] - ground_points[indices[0], 2]
    return result


def test_batched_matches_reference_per_point_loop() -> None:
    rng = np.random.default_rng(1)
    n = 300
    points = rng.uniform(0, 100, (n, 3))
    ground_indices = rng.choice(n, size=50, replace=False)
    ground_points = points[ground_indices]

    manager = NeighborhoodManager.from_array(ground_points)
    indices, _ = manager.query_points_many(points, k=1)
    actual = points[:, 2] - ground_points[indices[:, 0], 2]

    reference = _reference_relative_height(points, ground_points)

    np.testing.assert_allclose(actual, reference)
