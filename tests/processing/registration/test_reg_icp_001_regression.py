"""
Regression suite for REG-ICP-001 (fixed in this PR).

Performance debt: _compute_fitness() called
target_manager.query_point() individually inside a Python loop over
every source point -- the same per-point KDTree query pattern already
found and fixed repeatedly elsewhere in TopoCore (PR21.8).

Fix: replaced with a single batched query_points_many() call. The old
`len(indices) > 0` guard is not reproduced -- confirmed directly that
query_point()/query_points_many() with k=1 on a non-empty tree always
returns exactly one result per query point, and target_manager here
is always built from a non-empty target (register()'s own
_validate_inputs() already rejects an empty target before
_compute_fitness() is ever called).

This is a pure performance change -- the algorithm and its observable
result are unchanged. The tests below confirm exact numerical
equivalence against an independent reference reimplementation of the
old per-point loop, across a normal case, an exact-match case
(fitness == 1.0), and a far-apart case (fitness == 0.0), plus an
instrumentation check confirming the batched path is genuinely used
and the old per-point method is not.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.neighbors.manager import NeighborhoodManager
from topocore.processing.registration.base import Transformation
from topocore.processing.registration.icp import ICPBase


class _IdentityICP(ICPBase):
    def _estimate_transformation(self, correspondences, source_points, target_points):  # type: ignore[no-untyped-def]
        return Transformation.identity()


def _make_cloud(points: np.ndarray) -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(
        size=len(points),
        attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z],
    )
    chunk[PointAttribute.X][:] = points[:, 0]
    chunk[PointAttribute.Y][:] = points[:, 1]
    chunk[PointAttribute.Z][:] = points[:, 2]
    cloud.add_chunk(chunk)
    return cloud


def _reference_fitness(source_points: np.ndarray, target_manager: NeighborhoodManager, max_distance: float) -> float:
    """The exact pre-REG-ICP-001 per-point-loop algorithm, reimplemented as an independent reference."""
    if len(source_points) == 0:
        return 0.0

    matched = 0
    for point in source_points:
        indices, distances = target_manager.query_point(point[0], point[1], point[2], k=1)
        if len(indices) > 0 and distances[0] <= max_distance:
            matched += 1
    return matched / len(source_points)


def test_fitness_matches_reference_per_point_loop_normal_case() -> None:
    rng = np.random.default_rng(0)
    source_points = rng.uniform(0, 10, (50, 3))
    target_points = rng.uniform(0, 10, (40, 3))
    target_cloud = _make_cloud(target_points)
    target_manager = NeighborhoodManager.from_point_cloud(target_cloud)

    icp = _IdentityICP(max_correspondence_distance=2.0)
    fitness_actual = icp._compute_fitness(_make_cloud(source_points), target_cloud)
    fitness_reference = _reference_fitness(source_points, target_manager, 2.0)

    assert fitness_actual == fitness_reference


def test_fitness_is_one_when_source_equals_target() -> None:
    rng = np.random.default_rng(1)
    points = rng.uniform(0, 10, (30, 3))
    cloud = _make_cloud(points)

    icp = _IdentityICP(max_correspondence_distance=2.0)
    fitness = icp._compute_fitness(_make_cloud(points), cloud)

    assert fitness == 1.0


def test_fitness_is_zero_when_source_is_far_from_target() -> None:
    rng = np.random.default_rng(2)
    target_points = rng.uniform(0, 10, (30, 3))
    far_source = target_points + 10000.0

    icp = _IdentityICP(max_correspondence_distance=0.001)
    fitness = icp._compute_fitness(_make_cloud(far_source), _make_cloud(target_points))

    assert fitness == 0.0


def test_fitness_of_empty_source_is_zero() -> None:
    rng = np.random.default_rng(3)
    target_points = rng.uniform(0, 10, (10, 3))

    empty_source = PointCloud()
    empty_chunk = Chunk(size=0, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    empty_source.add_chunk(empty_chunk)

    icp = _IdentityICP()
    fitness = icp._compute_fitness(empty_source, _make_cloud(target_points))

    assert fitness == 0.0


def test_compute_fitness_uses_batched_query_not_per_point_loop() -> None:
    """
    Instrumentation check: query_points_many() must be called (the
    batched path), and query_point() (the old per-point path) must
    NOT be called by _compute_fitness().
    """
    rng = np.random.default_rng(4)
    source_points = rng.uniform(0, 10, (20, 3))
    target_points = rng.uniform(0, 10, (20, 3))
    target_cloud = _make_cloud(target_points)

    icp = _IdentityICP(max_correspondence_distance=2.0)

    original_query_points_many = NeighborhoodManager.query_points_many
    original_query_point = NeighborhoodManager.query_point

    with (
        patch.object(NeighborhoodManager, "query_points_many", autospec=True) as batched_spy,
        patch.object(NeighborhoodManager, "query_point", autospec=True) as per_point_spy,
    ):
        batched_spy.side_effect = original_query_points_many
        per_point_spy.side_effect = original_query_point

        icp._compute_fitness(_make_cloud(source_points), target_cloud)

        assert batched_spy.call_count == 1
        assert per_point_spy.call_count == 0
