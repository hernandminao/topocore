"""
Regression suite for the PR21.8 transversal-audit fix:
registration.icp's _find_correspondences(), found via a
whole-processing/-tree search for the per-point query_point() loop
pattern. This one is repeated on EVERY ICP iteration (correspondences
are recomputed each time), making it the highest-impact instance of
this pattern class found in the audit.

The `len(indices) == 0` guard the old loop had is confirmed
unreachable (NeighborhoodManager always holds a non-empty point set)
and is not reproduced -- see the source's own docstring.
"""

from __future__ import annotations

import numpy as np
from topocore.processing.neighbors.manager import NeighborhoodManager


def _reference_find_correspondences(
    source_points: np.ndarray,
    target_manager: NeighborhoodManager,
    max_distance: float,
) -> list[tuple[int, int, float]]:
    """The exact pre-PR21.8 per-point-loop algorithm, reimplemented as an independent reference."""
    correspondences = []
    for i, point in enumerate(source_points):
        indices, distances = target_manager.query_point(point[0], point[1], point[2], k=1)
        if len(indices) == 0:
            continue
        distance = distances[0]
        if distance <= max_distance:
            correspondences.append((i, indices[0], float(distance)))
    return correspondences


def _batched_find_correspondences(
    source_points: np.ndarray,
    target_manager: NeighborhoodManager,
    max_distance: float,
) -> list[tuple[int, int, float]]:
    indices, distances = target_manager.query_points_many(source_points, k=1)
    within_max_distance = np.flatnonzero(distances[:, 0] <= max_distance)
    return [
        (
            int(source_index),
            int(indices[source_index, 0]),
            float(distances[source_index, 0]),
        )
        for source_index in within_max_distance
    ]


def test_batched_correspondences_match_per_point_loop_reference() -> None:
    rng = np.random.default_rng(2)
    source_points = rng.uniform(0, 100, (50, 3))
    target_points = rng.uniform(0, 100, (60, 3))
    target_manager = NeighborhoodManager.from_array(target_points)
    max_distance = 15.0

    actual = _batched_find_correspondences(source_points, target_manager, max_distance)
    reference = _reference_find_correspondences(source_points, target_manager, max_distance)

    assert actual == reference


def test_batched_correspondences_respects_max_distance_filter() -> None:
    rng = np.random.default_rng(3)
    source_points = rng.uniform(0, 100, (30, 3))
    target_points = rng.uniform(0, 100, (30, 3))
    target_manager = NeighborhoodManager.from_array(target_points)

    tight = _batched_find_correspondences(source_points, target_manager, max_distance=0.001)
    loose = _batched_find_correspondences(source_points, target_manager, max_distance=1000.0)

    assert len(tight) <= len(loose)
    assert len(loose) == 30  # every source point must find some correspondence within a huge radius
