"""
Regression suite for the PR21.8 transversal-audit fix:
ground.manager.GroundManager._nearest_ground_elevation()'s per-point
query_point() loop, found via a whole-processing/-tree search for
the (NeighborhoodManager.from_array + per-point query_point loop)
pattern -- the same pattern already found and fixed in
features.geometric.RelativeHeightFeatureComputer and
classification.ml.
"""

from __future__ import annotations

import numpy as np
from topocore.processing.neighbors.manager import NeighborhoodManager


def test_batched_query_matches_per_point_loop_reference() -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    z = rng.uniform(0, 10, n)
    ground_indices = rng.choice(n, size=40, replace=False)
    ground_points = np.column_stack((x[ground_indices], y[ground_indices], z[ground_indices]))
    manager = NeighborhoodManager.from_array(ground_points)

    query_points = np.column_stack((x, y, z))
    indices, _ = manager.query_points_many(query_points, k=1)
    actual = ground_points[indices[:, 0], 2]

    reference = np.empty(n)
    for i in range(n):
        idx, _ = manager.query_point(x[i], y[i], z[i], k=1)
        reference[i] = ground_points[idx[0], 2]

    np.testing.assert_allclose(actual, reference)
