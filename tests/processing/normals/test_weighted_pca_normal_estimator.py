"""
Coverage audit tests for topocore.processing.normals.weighted_pca.WeightedPCANormalEstimator.

Focuses only on this class's OWN branches -- compute_pca()'s own
already-audited guarantees (see test_shared_compute_pca.py) are not
retested here, and neither are the k<3/point_count<k/empty-cloud
checks that this class delegates to compute_pca()/NeighborhoodManager
rather than validating itself.

Architectural finding, documented but NOT fixed here (would change
the exception contract, out of scope for a coverage audit): unlike
PCANormalEstimator.estimate_both() (which validates empty-cloud/
point_count<k itself, raising a clear, domain-specific NormalError),
WeightedPCANormalEstimator.estimate_both() has no such pre-checks --
these errors propagate as raw ProcessingError/NeighborError from
compute_pca()/NeighborhoodManager instead. This is an inconsistency
between sibling estimator classes, not something to retest or
"fix" here.

The coincident-points case (all neighbor distances zero, giving
sigma2 == 0) is treated as a genuine FUNCTIONAL branch of the
algorithm (not merely a numerical defense): confirmed directly that
np.linalg.eigh's own eigenvector output remains a valid orthonormal
basis even for an all-zero covariance matrix, so normals stay unit
vectors, and curvature correctly stays 0.0 (via np.where's guard)
despite the internal RuntimeWarning from `eigenvalues[:, 0] / sum_l`
being evaluated (and discarded) for the sum_l == 0 case. That warning
is NOT part of the test's assertions -- only the fact that the
process completes and produces a valid result is.

NOT tested here by design -- documented as orphaned (zero callers
confirmed via grep, same pattern as PCANormalEstimator's own
orphaned interface methods): estimate_at(), name(), requires_k(),
supports_weighted(). NormalManager (this estimator's only real
consumer) does not delegate to any of these.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import NormalError
from topocore.processing.normals.weighted_pca import WeightedPCANormalEstimator


def _flat_plane_cloud(n: int = 30, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = np.zeros(n)
    cloud.add_chunk(chunk)
    return cloud


def _coincident_points_cloud(n: int = 10) -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = np.full(n, 5.0)
    chunk[PointAttribute.Y][:] = np.full(n, 5.0)
    chunk[PointAttribute.Z][:] = np.full(n, 5.0)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# Construction -- k < 3.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 1, 2, -1])
def test_k_less_than_three_rejected_at_construction(k: int) -> None:
    with pytest.raises(NormalError, match="k must be at least 3"):
        WeightedPCANormalEstimator(k=k)


# ----------------------------------------------------------------------
# Happy path -- the full contract, not just "shapes look right".
# ----------------------------------------------------------------------


def test_happy_path_full_contract_with_default_sigma() -> None:
    """sigma=None -- the estimator must fall back to the mean neighbor distance."""
    cloud = _flat_plane_cloud(n=30)
    normals, curvature = WeightedPCANormalEstimator(k=5).estimate_both(cloud)

    assert normals.shape == (30, 3)
    assert curvature.shape == (30,)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.isfinite(normals).all()
    assert np.isfinite(curvature).all()


def test_flat_plane_has_near_zero_curvature() -> None:
    cloud = _flat_plane_cloud(n=30)
    _, curvature = WeightedPCANormalEstimator(k=5).estimate_both(cloud)

    np.testing.assert_allclose(curvature, 0.0, atol=1e-6)


def test_explicit_sigma_produces_valid_full_contract() -> None:
    cloud = _flat_plane_cloud(n=30)
    normals, curvature = WeightedPCANormalEstimator(k=5, sigma=2.0).estimate_both(cloud)

    assert normals.shape == (30, 3)
    assert curvature.shape == (30,)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_estimate_matches_estimate_both_normals() -> None:
    cloud = _flat_plane_cloud(n=30)
    estimator = WeightedPCANormalEstimator(k=5)

    normals_via_estimate = estimator.estimate(cloud)
    normals_via_both, _ = estimator.estimate_both(cloud)

    np.testing.assert_array_equal(normals_via_estimate, normals_via_both)


# ----------------------------------------------------------------------
# Coincident points -- a genuine functional branch, not just a numerical
# defense. The internal RuntimeWarning is deliberately not asserted on.
# ----------------------------------------------------------------------


def test_coincident_points_does_not_crash_and_produces_valid_result() -> None:
    cloud = _coincident_points_cloud(n=10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the internal 0/0 division warning is not this test's concern
        normals, curvature = WeightedPCANormalEstimator(k=5).estimate_both(cloud)

    assert normals.shape == (10, 3)
    assert curvature.shape == (10,)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.isfinite(normals).all()
    assert np.isfinite(curvature).all()


def test_coincident_points_curvature_is_zero() -> None:
    """sum_l == 0 for an all-zero covariance -- np.where's guard must produce 0.0, not NaN."""
    cloud = _coincident_points_cloud(n=10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, curvature = WeightedPCANormalEstimator(k=5).estimate_both(cloud)

    np.testing.assert_allclose(curvature, 0.0)


# ----------------------------------------------------------------------
# Orientation -- both branches, via the public estimate() contract.
# ----------------------------------------------------------------------


def test_orient_upward_default_flips_normals_to_positive_z() -> None:
    cloud = _flat_plane_cloud(n=30)
    normals = WeightedPCANormalEstimator(k=5, orient_upward=True).estimate(cloud)

    assert (normals[:, 2] >= 0).all()


def test_orient_toward_viewpoint_below_plane_flips_normals_to_negative_z() -> None:
    cloud = _flat_plane_cloud(n=30)
    viewpoint = np.array([5.0, 5.0, -100.0])

    normals = WeightedPCANormalEstimator(k=5, viewpoint=viewpoint).estimate(cloud)

    assert (normals[:, 2] <= 0).all()
