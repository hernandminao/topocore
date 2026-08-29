"""
Coverage audit tests for topocore.processing.normals.pca.PCANormalEstimator.

Focuses only on this class's OWN branches -- compute_pca()'s own
already-audited guarantees (see test_shared_compute_pca.py) are not
retested here.

Audit findings (documented here, not force-tested):

The NaN/Inf checks on both curvature (line 176) and oriented normals
(line 173) are confirmed unreachable under the current implementation.
Curvature: eigenvalues are already guaranteed finite by compute_pca()'s
own contract (that guarantee is itself documented as dead code there,
but it means these values can never actually be non-finite given the
current NeighborhoodManager/scipy contracts). Normals: confirmed
directly that a pathological (NaN-containing) viewpoint does NOT
propagate NaN into the output -- `NaN < 0.0` evaluates to False in
NumPy, so the sign-flip decision (`flip = alignment < 0.0`) simply
never triggers for NaN-affected points, and `normals[flip] *= -1.0`
never touches an already-finite value in a way that could introduce
NaN. This correction was made only after directly testing the
hypothesis and finding it false -- an earlier assumption that a
pathological viewpoint would reach this branch was wrong.

Also NOT tested here, documented as orphaned (zero callers confirmed
via grep across the whole repository): name(), requires_k(),
supports_weighted(), and estimate_at() -- NormalManager (this
estimator's only real consumer) does not delegate to any of these;
it maintains its own separate method/k properties and reimplements
estimate_at()'s own indexing logic independently rather than calling
the estimator's version. This is flagged as architectural debt (the
estimator's own public interface methods are bypassed by its primary
consumer), not resolved here.

_orient_normals() (a private method) is not tested directly -- both
its branches (viewpoint-based and upward-based orientation) are
exercised through estimate()/estimate_both()'s own public contract,
which is more representative of real behavior than testing a private
method in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import NormalError
from topocore.processing.neighbors.manager import NeighborhoodManager
from topocore.processing.normals.pca import PCANormalEstimator


def _flat_plane_cloud(n: int = 30, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = np.zeros(n)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# Construction -- k < 3.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 1, 2, -1])
def test_k_less_than_three_rejected_at_construction(k: int) -> None:
    with pytest.raises(NormalError, match="k must be at least 3"):
        PCANormalEstimator(k=k)


# ----------------------------------------------------------------------
# estimate_both() -- input validation.
# ----------------------------------------------------------------------


def test_empty_cloud_rejected() -> None:
    with pytest.raises(NormalError, match="empty point cloud"):
        PCANormalEstimator(k=5).estimate(PointCloud())


def test_point_count_less_than_k_rejected() -> None:
    small_cloud = PointCloud()
    chunk = Chunk(size=2, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [1.0, 2.0]
    chunk[PointAttribute.Y][:] = [1.0, 2.0]
    chunk[PointAttribute.Z][:] = [0.0, 0.0]
    small_cloud.add_chunk(chunk)

    with pytest.raises(NormalError, match="requires at least"):
        PCANormalEstimator(k=5).estimate(small_cloud)


def test_mismatched_external_manager_point_count_rejected() -> None:
    """An externally-provided manager built from a DIFFERENT point set than `cloud` must be rejected."""
    cloud = _flat_plane_cloud(n=30)
    other_cloud = _flat_plane_cloud(n=10, seed=1)
    mismatched_manager = NeighborhoodManager.from_point_cloud(other_cloud)

    with pytest.raises(NormalError, match="Invalid PCA eigenvalues size"):
        PCANormalEstimator(k=5).estimate(cloud, manager=mismatched_manager)


# ----------------------------------------------------------------------
# Happy path -- shapes, unit normals, curvature.
# ----------------------------------------------------------------------


def test_estimate_returns_unit_normals_with_correct_shape() -> None:
    cloud = _flat_plane_cloud()
    normals = PCANormalEstimator(k=5).estimate(cloud)

    assert normals.shape == (30, 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_estimate_both_returns_matching_normals_and_curvature() -> None:
    cloud = _flat_plane_cloud()
    estimator = PCANormalEstimator(k=5)

    normals_via_estimate = estimator.estimate(cloud)
    normals_via_both, curvature = estimator.estimate_both(cloud)

    np.testing.assert_array_equal(normals_via_estimate, normals_via_both)
    assert curvature.shape == (30,)


def test_flat_plane_has_near_zero_curvature() -> None:
    cloud = _flat_plane_cloud()
    _, curvature = PCANormalEstimator(k=5).estimate_both(cloud)

    np.testing.assert_allclose(curvature, 0.0, atol=1e-6)


# ----------------------------------------------------------------------
# Orientation -- both branches, via the public estimate() contract.
# ----------------------------------------------------------------------


def test_orient_upward_default_flips_normals_to_positive_z() -> None:
    cloud = _flat_plane_cloud()
    normals = PCANormalEstimator(k=5, orient_upward=True).estimate(cloud)

    assert (normals[:, 2] >= 0).all()


def test_orient_toward_viewpoint_below_plane_flips_normals_to_negative_z() -> None:
    cloud = _flat_plane_cloud()
    viewpoint = np.array([5.0, 5.0, -100.0])

    normals = PCANormalEstimator(k=5, viewpoint=viewpoint).estimate(cloud)

    assert (normals[:, 2] <= 0).all()


def test_viewpoint_orientation_takes_priority_over_orient_upward() -> None:
    """When both are set, viewpoint-based orientation must win (per _orient_normals()'s own branch order)."""
    cloud = _flat_plane_cloud()
    viewpoint = np.array([5.0, 5.0, -100.0])

    normals = PCANormalEstimator(k=5, orient_upward=True, viewpoint=viewpoint).estimate(cloud)

    assert (normals[:, 2] <= 0).all()
