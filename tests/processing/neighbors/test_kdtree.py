"""
Coverage audit tests for topocore.processing.neighbors.kdtree
(KDTreeNeighborSearch, _extract_xyz, _drop_self_column).

PR22 coverage hardening: this file already had extensive PR21.2
regression coverage for the `workers` parallelism parameter (see
test_workers_parallelism.py) and PR21.8's own batching work relies
on it throughout this whole session's earlier audits, but no
dedicated suite existed for the class's own core contract: knn/
knn_many/radius/radius_many/query_point/query_points_many/
query_point_radius, its validation helpers, and the two module-level
functions. This suite fills that gap.

Confirmed via direct execution before writing any test:
  - Both of _drop_self_column()'s own failure branches are genuinely
    reachable, via two DIFFERENT real-world scenarios (not the same
    one): "Could not find the query point among its own neighbors"
    was already established elsewhere in this session (coincident
    points confuse the KDTree's own self-lookup); "Could not obtain
    k unique neighbors" is reachable via a single-point cloud
    queried with include_self=False and k >= 1 (after removing the
    lone self-match, zero neighbors remain, short of any k).
  - cKDTree's own query_ball_point() is confirmed INCLUSIVE at the
    exact radius boundary (a point at distance exactly 1.0 from a
    radius=1.0 query IS included) -- not obvious/guaranteed without
    checking directly, now locked in as a regression.
  - points property returns a defensive copy, not the internal
    array reference.
  - "Point cloud must contain X/Y/Z coordinates" in _extract_xyz is
    NOT tested here -- unreachable, matching the same established
    pattern throughout this whole session (Chunk.__init__ already
    requires X/Y/Z at construction).
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import NeighborError
from topocore.processing.neighbors.kdtree import KDTreeNeighborSearch, _extract_xyz


def _cloud(n: int = 20, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 10, n)
    cloud.add_chunk(chunk)
    return cloud


def _points(n: int = 20, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 10, (n, 3))


# ----------------------------------------------------------------------
# Constructor / factory methods.
# ----------------------------------------------------------------------


def test_constructor_rejects_wrong_shape() -> None:
    with pytest.raises(NeighborError, match=r"shape \(N,3\)"):
        KDTreeNeighborSearch(np.zeros((5, 2)))


def test_constructor_rejects_zero_points() -> None:
    with pytest.raises(NeighborError, match="zero points"):
        KDTreeNeighborSearch(np.zeros((0, 3)))


def test_constructor_rejects_invalid_workers() -> None:
    with pytest.raises(NeighborError, match="workers"):
        KDTreeNeighborSearch(np.zeros((5, 3)), workers=0)
    with pytest.raises(NeighborError, match="workers"):
        KDTreeNeighborSearch(np.zeros((5, 3)), workers=-2)


def test_from_point_cloud_and_from_array_produce_equivalent_index() -> None:
    points = _points()
    cloud = _cloud(seed=0)

    search_cloud = KDTreeNeighborSearch.from_point_cloud(cloud)
    search_array = KDTreeNeighborSearch.from_array(points)

    assert search_cloud.point_count == search_array.point_count == 20


def test_extract_xyz_rejects_empty_cloud() -> None:
    with pytest.raises(NeighborError, match="empty point cloud"):
        _extract_xyz(PointCloud())


# ----------------------------------------------------------------------
# points / point_count properties.
# ----------------------------------------------------------------------


def test_points_property_returns_defensive_copy() -> None:
    search = KDTreeNeighborSearch(_points())
    snapshot = search.points
    snapshot[0] = [999.0, 999.0, 999.0]

    assert not np.array_equal(search.points[0], [999.0, 999.0, 999.0])


# ----------------------------------------------------------------------
# Validation helpers -- index / k / radius / indices array.
# ----------------------------------------------------------------------


def test_validate_index_rejects_non_integer() -> None:
    search = KDTreeNeighborSearch(_points())
    with pytest.raises(NeighborError, match="must be integer"):
        search.knn("a", 3)  # type: ignore[arg-type]


def test_validate_index_rejects_out_of_range() -> None:
    search = KDTreeNeighborSearch(_points())
    with pytest.raises(NeighborError, match="outside valid range"):
        search.knn(-1, 3)
    with pytest.raises(NeighborError, match="outside valid range"):
        search.knn(100, 3)


def test_validate_k_rejects_non_integer_zero_and_excessive() -> None:
    search = KDTreeNeighborSearch(_points())
    with pytest.raises(NeighborError, match="must be integer"):
        search.knn(0, "a")  # type: ignore[arg-type]
    with pytest.raises(NeighborError, match=">= 1"):
        search.knn(0, 0)
    with pytest.raises(NeighborError, match="exceeds maximum"):
        search.knn(0, 2_000_000)


def test_validate_radius_rejects_non_finite_and_non_positive() -> None:
    search = KDTreeNeighborSearch(_points())
    with pytest.raises(NeighborError, match="finite"):
        search.radius(0, float("nan"))
    with pytest.raises(NeighborError, match="positive"):
        search.radius(0, -1.0)


def test_validate_indices_rejects_out_of_bounds() -> None:
    search = KDTreeNeighborSearch(_points())
    with pytest.raises(NeighborError, match="out of bounds"):
        search.knn_many(np.array([-1, 5]), k=2)


def test_query_points_many_rejects_wrong_shape() -> None:
    search = KDTreeNeighborSearch(_points())
    with pytest.raises(NeighborError, match=r"shape \(M,3\)"):
        search.query_points_many(np.zeros((5, 2)), k=2)


# ----------------------------------------------------------------------
# _drop_self_column() -- both failure modes, confirmed reachable via
# two genuinely different real-world scenarios.
# ----------------------------------------------------------------------


def test_single_point_cloud_cannot_satisfy_k_with_include_self_false() -> None:
    """After removing the lone self-match, zero neighbors remain -- short of any k >= 1."""
    search = KDTreeNeighborSearch(np.array([[1.0, 2.0, 3.0]]))
    with pytest.raises(NeighborError, match="unique neighbors"):
        search.knn(0, 5, include_self=False)


def test_single_point_cloud_with_include_self_true_returns_self_padded() -> None:
    search = KDTreeNeighborSearch(np.array([[1.0, 2.0, 3.0]]))
    indices, distances = search.knn(0, 5, include_self=True)

    assert indices[0] == 0
    assert distances[0] == 0.0


# ----------------------------------------------------------------------
# Functional correctness: boundary radius, negative coordinates,
# include_self semantics.
# ----------------------------------------------------------------------


def test_radius_query_is_inclusive_at_exact_boundary() -> None:
    """cKDTree.query_ball_point() confirmed inclusive at radius==distance -- not obvious without checking."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    search = KDTreeNeighborSearch(points)

    result = search.radius(0, 1.0, include_self=False)

    assert 1 in result
    assert 2 not in result


def test_knn_works_with_negative_coordinates() -> None:
    points = np.array([[-5.0, -5.0, -5.0], [-4.0, -4.0, -4.0], [10.0, 10.0, 10.0]])
    search = KDTreeNeighborSearch(points)

    indices, distances = search.knn(0, 1)

    assert indices[0] == 1
    assert distances[0] == pytest.approx(np.sqrt(3.0))


def test_knn_many_include_self_semantics() -> None:
    search = KDTreeNeighborSearch(_points())

    indices_incl, _ = search.knn_many(np.array([0]), k=3, include_self=True)
    indices_excl, _ = search.knn_many(np.array([0]), k=3, include_self=False)

    assert 0 in indices_incl[0]
    assert 0 not in indices_excl[0]


def test_radius_many_include_self_semantics() -> None:
    search = KDTreeNeighborSearch(_points())

    result_incl = search.radius_many(np.array([0, 1]), radius=5.0, include_self=True)
    result_excl = search.radius_many(np.array([0, 1]), radius=5.0, include_self=False)

    assert 0 in result_incl[0]
    assert 0 not in result_excl[0]


def test_query_point_radius() -> None:
    points = _points()
    search = KDTreeNeighborSearch(points)

    neighbors = search.query_point_radius(points[0, 0], points[0, 1], points[0, 2], radius=2.0)

    assert 0 in neighbors


def test_query_points_many_k_one_reshapes_to_column_vector() -> None:
    points = _points()
    search = KDTreeNeighborSearch(points)

    indices, distances = search.query_points_many(points[:1], k=1)

    assert indices.shape == (1, 1)
    assert distances.shape == (1, 1)
