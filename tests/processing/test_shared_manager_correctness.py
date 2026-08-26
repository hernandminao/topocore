"""
Regression suite for PR21.3.4: sharing one NeighborhoodManager across
NormalManager, PCAFeatures, and RuleBasedClassifier must never change
any of their results, even when each requests a DIFFERENT k -- the
PR21.3.2 audit's own finding that k belongs to the query
(knn_many(k=...)), not the manager's identity.

See benchmarks/benchmark_shared_manager.py for the reproducible
timing benchmark (this file only verifies correctness, not speed).
"""

from __future__ import annotations

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.rules import RuleBasedClassifier
from topocore.processing.features.pca import PCAFeatures
from topocore.processing.neighbors.manager import NeighborhoodManager
from topocore.processing.normals.manager import NormalManager


def _random_cloud(n: int = 500, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 30, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 30, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 2, n)
    cloud.add_chunk(chunk)
    return cloud


def test_shared_manager_gives_identical_normals_with_different_k_elsewhere() -> None:
    """
    The decisive check: NormalManager(k=10) shares a manager with
    PCAFeatures(k=8) and RuleBasedClassifier(pca_neighbors=20) --
    normals must be identical to the separate (no shared manager) case.
    """
    cloud = _random_cloud()

    separate_normals = NormalManager(method="pca", k=10).estimate(cloud)

    shared_manager = NeighborhoodManager.from_point_cloud(cloud)
    shared_normals = NormalManager(method="pca", k=10).estimate(cloud, manager=shared_manager)

    # Use the SAME shared_manager for the other two modules with DIFFERENT k.
    PCAFeatures(k=8).compute_all(cloud, manager=shared_manager)
    RuleBasedClassifier(pca_neighbors=20).classify(cloud, manager=shared_manager)

    np.testing.assert_array_equal(separate_normals, shared_normals)


def test_shared_manager_gives_identical_pca_features_with_different_k_elsewhere() -> None:
    cloud = _random_cloud()

    separate_features = PCAFeatures(k=8).compute_all(cloud)

    shared_manager = NeighborhoodManager.from_point_cloud(cloud)
    NormalManager(method="pca", k=10).estimate(cloud, manager=shared_manager)
    shared_features = PCAFeatures(k=8).compute_all(cloud, manager=shared_manager)
    RuleBasedClassifier(pca_neighbors=20).classify(cloud, manager=shared_manager)

    np.testing.assert_array_equal(separate_features["planarity"], shared_features["planarity"])
    np.testing.assert_array_equal(separate_features["linearity"], shared_features["linearity"])


def test_shared_manager_gives_identical_classification_with_different_k_elsewhere() -> None:
    cloud = _random_cloud()

    separate_result = RuleBasedClassifier(pca_neighbors=20).classify(cloud)

    shared_manager = NeighborhoodManager.from_point_cloud(cloud)
    NormalManager(method="pca", k=10).estimate(cloud, manager=shared_manager)
    PCAFeatures(k=8).compute_all(cloud, manager=shared_manager)
    shared_result = RuleBasedClassifier(pca_neighbors=20).classify(cloud, manager=shared_manager)

    np.testing.assert_array_equal(separate_result.labels, shared_result.labels)


def test_full_pipeline_shared_manager_matches_full_pipeline_separate() -> None:
    """End-to-end: all three modules' outputs match exactly between the separate and shared-manager runs."""
    cloud = _random_cloud(n=800, seed=7)

    normals_a = NormalManager(method="pca", k=10).estimate(cloud)
    features_a = PCAFeatures(k=8).compute_all(cloud)
    result_a = RuleBasedClassifier(pca_neighbors=20).classify(cloud)

    shared_manager = NeighborhoodManager.from_point_cloud(cloud)
    normals_b = NormalManager(method="pca", k=10).estimate(cloud, manager=shared_manager)
    features_b = PCAFeatures(k=8).compute_all(cloud, manager=shared_manager)
    result_b = RuleBasedClassifier(pca_neighbors=20).classify(cloud, manager=shared_manager)

    np.testing.assert_array_equal(normals_a, normals_b)
    np.testing.assert_array_equal(features_a["planarity"], features_b["planarity"])
    np.testing.assert_array_equal(result_a.labels, result_b.labels)
