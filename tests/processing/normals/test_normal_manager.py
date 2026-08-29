"""
Coverage audit tests for topocore.processing.normals.manager.NormalManager.

Focuses on this class's own branches -- PCANormalEstimator/
WeightedPCANormalEstimator's own already-audited contracts (see
test_pca_normal_estimator.py and test_weighted_pca_normal_estimator.py)
are not retested here.

Confirmed via 2 real callers (segmentation/region_growing.py,
registration/point_to_plane.py): both configure NormalManager entirely
at construction time and call only estimate()/estimate_both() --
never estimate_at(), estimate_curvature(), __call__, or the property
setters. Despite this, the setters ARE tested here (not documented as
orphaned): unlike estimate_at()/__call__ (bypassed by an alternative,
actively-used mechanism), the setters have no competing
implementation -- they are simply unused by today's 2 callers, while
carrying real, distinct validation logic and explicit cache-clearing
behavior that constitutes a genuine, testable public contract.

Subtle finding, documented but not changing what's tested: cache
invalidation via `self._cache.clear()` in each setter is not
strictly load-bearing for CORRECTNESS -- confirmed that
`_cache_key()` already includes the resolved method/k/orient_upward/
viewpoint/sigma values, so a changed setting naturally produces a
DIFFERENT cache key (a cache miss) regardless of whether `clear()`
ran. `clear()` is a memory-bounding optimization (preventing
unbounded cache growth across repeated configuration changes on a
long-lived manager), not what prevents stale/wrong results. The
tests below verify the OBSERVABLE contract (a changed setting
produces a result reflecting the NEW setting, confirmed by direct
comparison against a freshly-constructed manager with the same
target settings) rather than inspecting the cache's private internals.

Documented as unreachable under the current _SUPPORTED_METHODS
registry (both registered factories -- pca, weighted_pca -- produce
NormalAndCurvatureEstimator instances; confirmed no other class
implements NormalEstimator without also implementing
NormalAndCurvatureEstimator): estimate_curvature()'s own
CurvatureEstimator branch and final raise, and
_estimate_both_cached()'s "else" branch. These exist as
forward-looking defensive dispatch for a hypothetical normal-only or
curvature-only estimator that isn't currently registered.

Documented as orphaned (zero callers confirmed via grep), not
tested: estimate_at(), __call__.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import NormalError
from topocore.processing.normals.manager import NormalManager


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
# Construction validation.
# ----------------------------------------------------------------------


def test_unsupported_method_rejected_at_construction() -> None:
    with pytest.raises(NormalError, match="Unsupported method"):
        NormalManager(method="not_a_real_method")


def test_k_less_than_three_rejected_at_construction() -> None:
    with pytest.raises(NormalError, match="k must be at least 3"):
        NormalManager(k=1)


# ----------------------------------------------------------------------
# Getters -- return the values passed at construction.
# ----------------------------------------------------------------------


def test_getters_reflect_constructor_arguments() -> None:
    viewpoint = np.array([1.0, 2.0, 3.0])
    manager = NormalManager(method="weighted_pca", k=7, orient_upward=False, viewpoint=viewpoint)

    assert manager.method == "weighted_pca"
    assert manager.k == 7
    assert manager.orient_upward is False
    np.testing.assert_array_equal(manager.viewpoint, viewpoint)


# ----------------------------------------------------------------------
# method setter.
# ----------------------------------------------------------------------


def test_method_setter_accepts_valid_method() -> None:
    manager = NormalManager(method="pca")
    manager.method = "weighted_pca"
    assert manager.method == "weighted_pca"


def test_method_setter_rejects_invalid_method() -> None:
    manager = NormalManager(method="pca")
    with pytest.raises(NormalError, match="Unsupported method"):
        manager.method = "not_a_real_method"


def test_changing_method_produces_result_matching_a_fresh_manager_with_that_method() -> None:
    cloud = _flat_plane_cloud()
    manager = NormalManager(method="pca", k=5)
    manager.estimate(cloud)  # populate cache under method="pca"

    manager.method = "weighted_pca"
    result_after_change = manager.estimate(cloud)

    fresh_manager = NormalManager(method="weighted_pca", k=5)
    result_fresh = fresh_manager.estimate(cloud)

    np.testing.assert_array_equal(result_after_change, result_fresh)


# ----------------------------------------------------------------------
# k setter.
# ----------------------------------------------------------------------


def test_k_setter_accepts_valid_k() -> None:
    manager = NormalManager(k=5)
    manager.k = 10
    assert manager.k == 10


def test_k_setter_rejects_k_less_than_three() -> None:
    manager = NormalManager(k=5)
    with pytest.raises(NormalError, match="k must be at least 3"):
        manager.k = 2


def test_changing_k_produces_result_matching_a_fresh_manager_with_that_k() -> None:
    cloud = _flat_plane_cloud()
    manager = NormalManager(method="pca", k=5)
    manager.estimate(cloud)

    manager.k = 10
    result_after_change = manager.estimate(cloud)

    fresh_manager = NormalManager(method="pca", k=10)
    result_fresh = fresh_manager.estimate(cloud)

    np.testing.assert_array_equal(result_after_change, result_fresh)


# ----------------------------------------------------------------------
# orient_upward setter.
# ----------------------------------------------------------------------


def test_orient_upward_setter_changes_value() -> None:
    manager = NormalManager(orient_upward=True)
    manager.orient_upward = False
    assert manager.orient_upward is False


def test_changing_orient_upward_produces_result_matching_a_fresh_manager() -> None:
    cloud = _flat_plane_cloud()
    manager = NormalManager(method="pca", k=5, orient_upward=True)
    manager.estimate(cloud)

    manager.orient_upward = False
    result_after_change = manager.estimate(cloud)

    fresh_manager = NormalManager(method="pca", k=5, orient_upward=False)
    result_fresh = fresh_manager.estimate(cloud)

    np.testing.assert_array_equal(result_after_change, result_fresh)


# ----------------------------------------------------------------------
# viewpoint setter.
# ----------------------------------------------------------------------


def test_viewpoint_setter_accepts_correct_shape() -> None:
    manager = NormalManager()
    manager.viewpoint = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(manager.viewpoint, [1.0, 2.0, 3.0])


def test_viewpoint_setter_rejects_wrong_shape() -> None:
    manager = NormalManager()
    with pytest.raises(NormalError, match="viewpoint must have shape"):
        manager.viewpoint = np.array([1.0, 2.0])


def test_changing_viewpoint_produces_result_matching_a_fresh_manager() -> None:
    cloud = _flat_plane_cloud()
    manager = NormalManager(method="pca", k=5)
    manager.estimate(cloud)  # populate cache with viewpoint=None (orient_upward)

    viewpoint = np.array([5.0, 5.0, -100.0])
    manager.viewpoint = viewpoint
    result_after_change = manager.estimate(cloud)

    fresh_manager = NormalManager(method="pca", k=5, viewpoint=viewpoint)
    result_fresh = fresh_manager.estimate(cloud)

    np.testing.assert_array_equal(result_after_change, result_fresh)


# ----------------------------------------------------------------------
# estimate() / estimate_both() -- confirmed active via real callers.
# ----------------------------------------------------------------------


def test_estimate_and_estimate_both_return_consistent_normals() -> None:
    cloud = _flat_plane_cloud()
    manager = NormalManager(method="pca", k=5)

    normals_only = manager.estimate(cloud)
    normals_both, curvature = manager.estimate_both(cloud)

    np.testing.assert_array_equal(normals_only, normals_both)
    assert curvature.shape == (30,)


def test_repeated_estimate_call_hits_cache() -> None:
    """Same manager, same cloud, same params -- second call must return the identical cached array object."""
    cloud = _flat_plane_cloud()
    manager = NormalManager(method="pca", k=5)

    first = manager.estimate(cloud)
    second = manager.estimate(cloud)

    assert first is second


def test_clear_cache_forces_recomputation() -> None:
    cloud = _flat_plane_cloud()
    manager = NormalManager(method="pca", k=5)

    first = manager.estimate(cloud)
    manager.clear_cache()
    second = manager.estimate(cloud)

    assert first is not second
    np.testing.assert_array_equal(first, second)
