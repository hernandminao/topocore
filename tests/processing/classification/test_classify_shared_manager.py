"""
Regression suite for the PR21.3.3 addition: RuleBasedClassifier.classify()
now accepts an optional, keyword-only NeighborhoodManager, matching the
pattern already used by NormalManager.estimate()/PCAFeatures.compute_all().

Added specifically to unblock cross-module NeighborhoodManager sharing
(the PR21.3.2 audit's key finding: RuleBasedClassifier was the only one
of the three consumers -- NormalManager, PCAFeatures, RuleBasedClassifier
-- with no way for an external caller to pass in an already-built
manager). Backward compatibility is the central contract here:
classify(cloud) with no manager must behave EXACTLY as before.
"""

from __future__ import annotations

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.rules import RuleBasedClassifier
from topocore.processing.neighbors.manager import NeighborhoodManager


def _random_cloud(n: int = 300, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 30, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 30, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 1, n)
    cloud.add_chunk(chunk)
    return cloud


def test_classify_without_manager_behaves_as_before() -> None:
    cloud = _random_cloud()
    result = RuleBasedClassifier().classify(cloud)
    assert len(result.labels) == cloud.point_count


def test_classify_with_external_manager_produces_identical_results() -> None:
    """The core contract: sharing a pre-built manager must not change the classification outcome."""
    cloud = _random_cloud()
    classifier = RuleBasedClassifier()

    result_default = classifier.classify(cloud)

    shared_manager = NeighborhoodManager.from_point_cloud(cloud)
    result_shared = classifier.classify(cloud, manager=shared_manager)

    np.testing.assert_array_equal(result_default.labels, result_shared.labels)


def test_classify_with_manager_is_keyword_only() -> None:
    """manager cannot be passed positionally -- confirms the signature matches the sibling modules' own convention."""
    cloud = _random_cloud()
    manager = NeighborhoodManager.from_point_cloud(cloud)

    try:
        RuleBasedClassifier().classify(cloud, manager)  # type: ignore[misc]
        raised = False
    except TypeError:
        raised = True

    assert raised


def test_classify_with_manager_from_different_cloud_still_runs() -> None:
    """
    Not a scenario this API forbids outright (matching NormalManager's
    own documented stance: an advanced, uncommon override this layer
    doesn't specially validate) -- confirms it at least doesn't crash
    when given a manager built from a different (but same-sized) cloud.
    """
    cloud_a = _random_cloud(seed=1)
    cloud_b = _random_cloud(seed=2)
    manager_b = NeighborhoodManager.from_point_cloud(cloud_b)

    result = RuleBasedClassifier().classify(cloud_a, manager=manager_b)
    assert len(result.labels) == cloud_a.point_count
