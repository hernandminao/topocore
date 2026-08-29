"""
Regression suite for NORMALS-SIBLING-001 (fixed in this PR).

Bug: PCANormalEstimator.estimate_both() validated empty-cloud and
point_count<k conditions itself, raising a clear, domain-specific
NormalError. WeightedPCANormalEstimator.estimate_both() had no
equivalent checks, letting ProcessingError/NeighborError from
NeighborhoodManager/compute_pca() leak through unwrapped for the
exact same input conditions -- an inconsistency between sibling
estimator classes' public exception contracts.

Fix: WeightedPCANormalEstimator.estimate_both() now validates both
conditions identically (mirrored wording, adjusted to say "weighted
PCA" matching this file's own existing message convention from
__init__'s k < 3 check), before ever touching its dependencies. This
is a validation mirror, not a blanket try/except -- genuinely
unexpected errors from those dependencies are not swallowed or
reclassified, only these two specific, already-known input
conditions are now checked explicitly and early, exactly as
PCANormalEstimator itself already did.

The happy path is confirmed unaffected: normals/curvature shapes for
a valid cloud are unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import NormalError
from topocore.processing.normals.pca import PCANormalEstimator
from topocore.processing.normals.weighted_pca import WeightedPCANormalEstimator


@pytest.mark.parametrize("estimator_class", [PCANormalEstimator, WeightedPCANormalEstimator])
def test_empty_cloud_raises_normal_error_for_both_siblings(
    estimator_class: type[PCANormalEstimator | WeightedPCANormalEstimator],
) -> None:
    """Previously WeightedPCANormalEstimator let a raw NeighborError leak through here; now both siblings raise NormalError identically."""
    with pytest.raises(NormalError, match="empty point cloud"):
        estimator_class(k=5).estimate(PointCloud())


@pytest.mark.parametrize("estimator_class", [PCANormalEstimator, WeightedPCANormalEstimator])
def test_point_count_less_than_k_raises_normal_error_for_both_siblings(
    estimator_class: type[PCANormalEstimator | WeightedPCANormalEstimator],
) -> None:
    """Previously WeightedPCANormalEstimator let a raw ProcessingError leak through here; now both siblings raise NormalError identically."""
    small_cloud = PointCloud()
    chunk = Chunk(size=2, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [1.0, 2.0]
    chunk[PointAttribute.Y][:] = [1.0, 2.0]
    chunk[PointAttribute.Z][:] = [0.0, 0.0]
    small_cloud.add_chunk(chunk)

    with pytest.raises(NormalError, match="requires at least"):
        estimator_class(k=5).estimate(small_cloud)


def test_weighted_pca_happy_path_unaffected_by_the_fix() -> None:
    rng = np.random.default_rng(0)
    n = 30
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = np.zeros(n)
    cloud.add_chunk(chunk)

    normals, curvature = WeightedPCANormalEstimator(k=5).estimate_both(cloud)

    assert normals.shape == (30, 3)
    assert curvature.shape == (30,)
