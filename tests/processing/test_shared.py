"""
Regression suite for topocore.processing._shared.compute_pca -- PR19.

Includes a SEVERE, non-deterministic bug found and fixed in this
session: compute_pca() cached its result keyed by (id(manager), k),
where `manager` is an internal, EPHEMERAL NeighborhoodManager freshly
constructed on every PCANormalEstimator.estimate() call and eligible
for garbage collection immediately after. Python's memory allocator
readily reuses freed addresses, so id(manager) for one point cloud
was frequently reused for the very next point cloud's manager shortly
after -- causing compute_pca() to silently return a COMPLETELY
DIFFERENT point cloud's stale PCAComputation (wrong neighbors, wrong
covariance, wrong eigenvalues/eigenvectors -- the entire result, not
a numeric approximation).

Confirmed with a definitive trace (session investigation): calling
estimate() on a flat plane then immediately on a tilted plane
occasionally (~1 in 10-30 attempts in a tight loop) returned the
FLAT plane's normal for the TILTED plane's own points. Traced to a
literal cache hit where the "new" manager's id exactly matched the
just-freed previous manager's id.

Root cause investigation ruled out, in order: NormalManager's own
cache (bug reproduced with fresh, uncached estimators); id(cloud)
collision (ids were confirmed distinct); non-deterministic KNN
tie-breaking (neighbor index sets were confirmed identical across
calls); eigenvalue degeneracy (eigenvalues were confirmed
well-separated, 0 vs ~1.06). The actual cause was one level deeper,
inside compute_pca()'s own now-removed cache.

Fixed by removing the cache entirely: it could never legitimately
hit under normal usage in the first place (a fresh NeighborhoodManager
is constructed per call), so removing it costs no genuine performance
benefit while eliminating a serious, silent correctness bug.
NormalManager's own cache (keyed by id(cloud), the caller-owned,
stable-lifetime object -- already audited and fixed earlier in this
session) remains the correct place for caching at this level.
"""

from __future__ import annotations

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing._shared import compute_pca
from topocore.processing.neighbors import NeighborhoodManager
from topocore.processing.normals.pca import PCANormalEstimator


def _grid_cloud(z_fn) -> PointCloud:  # type: ignore[no-untyped-def]
    xs, ys, zs = [], [], []
    for i in range(5):
        for j in range(5):
            xs.append(float(i))
            ys.append(float(j))
            zs.append(z_fn(i, j))
    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# The exact regression: alternating flat/tilted plane computations,
# many times, must never cross-contaminate.
# ----------------------------------------------------------------------


def test_alternating_planes_never_cross_contaminate() -> None:
    """
    The exact reproduction, run many times (the bug was
    probabilistic, ~1-in-10 to 1-in-30 per attempt in the original
    investigation) -- 0 failures expected now, not merely "rare".
    """
    flat_cloud = _grid_cloud(lambda i, j: 0.0)
    tilted_cloud = _grid_cloud(lambda i, j: float(i))

    for _ in range(500):
        estimator = PCANormalEstimator(k=9, orient_upward=True)
        estimator.estimate(flat_cloud)

        n1 = estimator.estimate(tilted_cloud)[12]
        n2 = estimator.estimate(tilted_cloud)[12]

        assert not np.allclose(n1, [0.0, 0.0, 1.0]), "tilted plane must never report the flat plane's normal"
        np.testing.assert_allclose(n1, n2)


def test_repeated_new_estimators_never_cross_contaminate() -> None:
    """
    Same reproduction, but with a FRESH PCANormalEstimator (and
    therefore fresh, short-lived NeighborhoodManager) constructed
    every single time -- the exact condition that made id() reuse
    likely in the first place.
    """
    for _ in range(300):
        flat_cloud = _grid_cloud(lambda i, j: 0.0)
        tilted_cloud = _grid_cloud(lambda i, j: float(i))

        PCANormalEstimator(k=9, orient_upward=True).estimate(flat_cloud)
        result = PCANormalEstimator(k=9, orient_upward=True).estimate(tilted_cloud)[12]

        assert not np.allclose(result, [0.0, 0.0, 1.0])


# ----------------------------------------------------------------------
# Hernán's minimal deterministic contract: compute_pca() itself must
# be a pure, deterministic function of its actual inputs.
# ----------------------------------------------------------------------


def test_compute_pca_is_deterministic_for_the_same_manager() -> None:
    tilted_cloud = _grid_cloud(lambda i, j: float(i))
    manager = NeighborhoodManager.from_point_cloud(tilted_cloud)

    result1 = compute_pca(manager, k=9)
    result2 = compute_pca(manager, k=9)

    np.testing.assert_allclose(result1.eigenvalues, result2.eigenvalues)
    np.testing.assert_allclose(np.abs(result1.eigenvectors), np.abs(result2.eigenvectors))


def test_compute_pca_is_deterministic_across_different_manager_instances() -> None:
    """
    Two DIFFERENT NeighborhoodManager instances wrapping the
    IDENTICAL underlying cloud must give the identical PCA result --
    confirms the result depends on the actual data, not on which
    manager object (or its id()) happens to be passed in.
    """
    tilted_cloud = _grid_cloud(lambda i, j: float(i))

    manager1 = NeighborhoodManager.from_point_cloud(tilted_cloud)
    manager2 = NeighborhoodManager.from_point_cloud(tilted_cloud)
    assert manager1 is not manager2

    result1 = compute_pca(manager1, k=9)
    result2 = compute_pca(manager2, k=9)

    np.testing.assert_allclose(result1.eigenvalues, result2.eigenvalues)
    np.testing.assert_allclose(np.abs(result1.eigenvectors), np.abs(result2.eigenvectors))


def test_normal_estimation_is_deterministic_for_the_same_cloud() -> None:
    tilted_cloud = _grid_cloud(lambda i, j: float(i))
    estimator = PCANormalEstimator(k=9, orient_upward=True)

    n1 = estimator.estimate(tilted_cloud)
    n2 = estimator.estimate(tilted_cloud)

    np.testing.assert_allclose(n1, n2)


def test_compute_pca_does_not_mutate_its_inputs() -> None:
    """
    Verifies compute_pca() doesn't silently modify the point array
    it reads from (a plausible root cause the investigation
    explicitly needed to rule out).
    """
    tilted_cloud = _grid_cloud(lambda i, j: float(i))
    manager = NeighborhoodManager.from_point_cloud(tilted_cloud)
    original_points = manager.search.points.copy()

    compute_pca(manager, k=9)

    np.testing.assert_array_equal(manager.search.points, original_points)
