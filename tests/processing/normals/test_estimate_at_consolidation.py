"""
Regression suite for Phase 4 (Architectural bypass / duplication):
estimate_at() ↔ NormalManager.

Audit finding: the surrounding call each estimate_at() makes to
obtain the full (normals, curvature) result genuinely differs for a
legitimate architectural reason -- confirmed directly before any
change that NormalManager.estimate_at() reuses the SAME cached array
object across estimate()/estimate_at() calls (identity-checked, not
just value-equal). Delegating NormalManager.estimate_at() to the
underlying estimator's own estimate_at() would bypass this cache
entirely (the estimator has no knowledge of NormalManager's cache),
recomputing from scratch on every call -- a real regression, not a
cleanup. This asymmetry is therefore NOT resolved by delegation.

What WAS genuinely duplicated with no justification: the final
"select rows by indices, cast to float64" snippet, identical across
PCANormalEstimator.estimate_at() (normals), PCACurvatureEstimator.
estimate_at() (curvature), WeightedPCANormalEstimator.estimate_at()
(normals), and NormalManager.estimate_at() (normals). Extracted into
a shared select_at_indices() helper (normals/base.py), used by all
four. NORMALS-MANAGER-001 (the separate viewpoint-setter finding) is
intentionally not touched here.

All four call sites are confirmed to produce identical results to
their pre-consolidation implementations, and NormalManager's cache
reuse (identity, not just equality) is confirmed unaffected.
"""

from __future__ import annotations

import numpy as np
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.normals.manager import NormalManager
from topocore.processing.normals.pca import PCACurvatureEstimator, PCANormalEstimator
from topocore.processing.normals.weighted_pca import WeightedPCANormalEstimator


def _cloud(n: int = 30, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = np.zeros(n)
    cloud.add_chunk(chunk)
    return cloud


def test_pca_normal_estimator_estimate_at_matches_full_array_subset() -> None:
    cloud = _cloud()
    indices = np.array([0, 5, 10])
    estimator = PCANormalEstimator(k=5)

    full = estimator.estimate(cloud)
    subset = estimator.estimate_at(cloud, indices)

    np.testing.assert_array_equal(subset, full[indices])


def test_pca_curvature_estimator_estimate_at_matches_full_array_subset() -> None:
    cloud = _cloud()
    indices = np.array([0, 5, 10])
    estimator = PCACurvatureEstimator(k=5)

    full = estimator.estimate(cloud)
    subset = estimator.estimate_at(cloud, indices)

    np.testing.assert_array_equal(subset, full[indices])


def test_weighted_pca_estimate_at_matches_full_array_subset() -> None:
    cloud = _cloud()
    indices = np.array([0, 5, 10])
    estimator = WeightedPCANormalEstimator(k=5)

    full = estimator.estimate(cloud)
    subset = estimator.estimate_at(cloud, indices)

    np.testing.assert_array_equal(subset, full[indices])


def test_normal_manager_estimate_at_matches_full_array_subset() -> None:
    cloud = _cloud()
    indices = np.array([0, 5, 10])
    manager = NormalManager(method="pca", k=5)

    full = manager.estimate(cloud)
    subset = manager.estimate_at(cloud, indices)

    np.testing.assert_array_equal(subset, full[indices])


def test_normal_manager_estimate_at_without_indices_reuses_cache_identity() -> None:
    """The consolidation must not break NormalManager's cache reuse -- same object, not just equal values."""
    cloud = _cloud()
    manager = NormalManager(method="pca", k=5)

    first = manager.estimate(cloud)
    second = manager.estimate_at(cloud)

    assert first is second
