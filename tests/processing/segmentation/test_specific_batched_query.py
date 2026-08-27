"""
Regression suite for the PR21.8 transversal-audit fix:
segmentation.specific's relative-height-above-ground helper's
per-point query_point() loop, found via a whole-processing/-tree
search for the same pattern already fixed in
features.geometric.RelativeHeightFeatureComputer,
classification.ml, and ground.manager.
"""

from __future__ import annotations

import numpy as np
from topocore.processing.neighbors.manager import NeighborhoodManager


def test_batched_query_matches_per_point_loop_reference() -> None:
    rng = np.random.default_rng(1)
    n = 300
    points = rng.uniform(0, 100, (n, 3))
    ground_indices = rng.choice(n, size=40, replace=False)
    ground_points = points[ground_indices]
    manager = NeighborhoodManager.from_array(ground_points)

    indices, _ = manager.query_points_many(points, k=1)
    actual = points[:, 2] - ground_points[indices[:, 0], 2]

    reference = np.empty(n)
    for i in range(n):
        idx, _ = manager.query_point(points[i, 0], points[i, 1], points[i, 2], k=1)
        reference[i] = points[i, 2] - ground_points[idx[0], 2]

    np.testing.assert_allclose(actual, reference)
