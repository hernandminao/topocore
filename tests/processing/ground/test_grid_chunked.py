"""
Regression suite for PR21.7.8: GridGroundClassifier/
GridGroundElevationEstimator's rewritten two-pass, chunk-wise
accumulator (_classify_chunked/_compute_cell_minimums_chunked)
versus the prior concatenate-everything approach
(_extract_xyz + _compute_cell_minimums, kept as this module's own
regression reference).

Unlike VoxelSampler/StratifiedSampler (PR21.7.5/7.6), this module's
`np.minimum.at`-based per-cell reduction was already confirmed
genuinely O(N), not O(N x G) -- this fix targets only the memory
overhead from concatenating every chunk's X/Y/Z into one global
array before computing cell minimums.

Minimum is commutative and associative, so merging per-chunk local
minimums into a global per-cell minimum is correct regardless of
chunk order or how a cell's members are split across chunks -- unlike
"closest" (PR21.7.5/7.6), there is no tie-breaking rule to preserve,
since only the minimum VALUE is tracked, never which point achieved
it. The decisive property verified below: a single cell whose member
points are split across ALL THREE chunks, with the true minimum
arriving from the MIDDLE chunk (not the first or last), must still
correctly become the global minimum used for every point in that
cell.

Also fixed during this suite's own development: GridGroundElevationEstimator's
new chunked path was initially missing the same empty-cloud
ValueError check `_classify_chunked` already had (confirmed the
pre-PR21.7.8 estimate() raised this same ValueError, via the same
shared `_extract_xyz()` classify() used) -- both paths now raise
identically.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.ground.grid import (
    GridGroundClassifier,
    GridGroundElevationEstimator,
    _compute_cell_minimums,
)


def _chunk(xs: list[float], ys: list[float], zs: list[float]) -> Chunk:
    n = len(xs)
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    return chunk


def _reference_ground_z(cloud: PointCloud, cell_size: float) -> np.ndarray:
    """The exact pre-PR21.7.8 algorithm, reimplemented via the still-present reference function."""
    xs = np.concatenate([c[PointAttribute.X] for c in cloud])
    ys = np.concatenate([c[PointAttribute.Y] for c in cloud])
    zs = np.concatenate([c[PointAttribute.Z] for c in cloud])
    return _compute_cell_minimums(xs, ys, zs, cell_size)


# ----------------------------------------------------------------------
# The decisive cross-chunk-boundary case: true minimum from the middle chunk.
# ----------------------------------------------------------------------


def test_cell_minimum_from_middle_chunk_used_for_every_member() -> None:
    """
    Cell A's points are split across all 3 chunks (z=5, z=50 in
    chunk 1; z=1 -- the TRUE minimum -- in chunk 2; z=3, z=30 in
    chunk 3). Every point in cell A must use z=1 as its ground
    elevation, confirming the minimum from a chunk OTHER than the
    first is correctly propagated globally.
    """
    cloud = PointCloud()
    cloud.add_chunk(_chunk([1.0, 11.0], [1.0, 11.0], [5.0, 50.0]))  # A(z=5), B(z=50)
    cloud.add_chunk(_chunk([2.0], [2.0], [1.0]))  # A(z=1) -- the true min
    cloud.add_chunk(_chunk([3.0, 21.0], [3.0, 21.0], [3.0, 30.0]))  # A(z=3), C(z=30)

    elevations = GridGroundElevationEstimator(cell_size=10.0).estimate(cloud)

    np.testing.assert_array_equal(elevations, [1.0, 50.0, 1.0, 1.0, 30.0])


def test_classify_matches_expected_mask_for_the_same_scenario() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_chunk([1.0, 11.0], [1.0, 11.0], [5.0, 50.0]))
    cloud.add_chunk(_chunk([2.0], [2.0], [1.0]))
    cloud.add_chunk(_chunk([3.0, 21.0], [3.0, 21.0], [3.0, 30.0]))

    mask = GridGroundClassifier(cell_size=10.0, height_threshold=0.5).classify(cloud)

    # ground_z = [1,50,1,1,30]; z=[5,50,1,3,30] -> diffs=[4,0,0,2,0] <= 0.5 -> [F,T,T,F,T]
    np.testing.assert_array_equal(mask, [False, True, True, False, True])


# ----------------------------------------------------------------------
# Equivalence against the pre-PR21.7.8 reference, on random data.
# ----------------------------------------------------------------------


def test_matches_reference_on_random_multi_chunk_data() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    xs, ys, zs = rng.uniform(0, 100, n), rng.uniform(0, 100, n), rng.uniform(0, 20, n)

    cloud = PointCloud()
    for start, end in ((0, 700), (700, 1300), (1300, 2000)):
        cloud.add_chunk(_chunk(list(xs[start:end]), list(ys[start:end]), list(zs[start:end])))

    reference = _reference_ground_z(cloud, cell_size=5.0)
    actual = GridGroundElevationEstimator(cell_size=5.0).estimate(cloud)

    np.testing.assert_allclose(actual, reference)


def test_classify_matches_reference_on_random_multi_chunk_data() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    xs, ys, zs = rng.uniform(0, 100, n), rng.uniform(0, 100, n), rng.uniform(0, 20, n)

    cloud = PointCloud()
    for start, end in ((0, 700), (700, 1300), (1300, 2000)):
        cloud.add_chunk(_chunk(list(xs[start:end]), list(ys[start:end]), list(zs[start:end])))

    reference_ground_z = _reference_ground_z(cloud, cell_size=5.0)
    reference_mask = (zs - reference_ground_z) <= 0.2

    actual_mask = GridGroundClassifier(cell_size=5.0, height_threshold=0.2).classify(cloud)

    np.testing.assert_array_equal(actual_mask, reference_mask)


def test_matches_reference_with_uneven_chunk_sizes() -> None:
    rng = np.random.default_rng(2)
    n = 1500
    xs, ys, zs = rng.uniform(0, 50, n), rng.uniform(0, 50, n), rng.uniform(0, 10, n)

    cloud = PointCloud()
    for start, end in ((0, 5), (5, 800), (800, 900), (900, 1500)):
        cloud.add_chunk(_chunk(list(xs[start:end]), list(ys[start:end]), list(zs[start:end])))

    reference = _reference_ground_z(cloud, cell_size=3.0)
    actual = GridGroundElevationEstimator(cell_size=3.0).estimate(cloud)

    np.testing.assert_allclose(actual, reference)


# ----------------------------------------------------------------------
# Points exactly on cell boundaries.
# ----------------------------------------------------------------------


def test_points_exactly_on_cell_boundaries() -> None:
    """floor(x/cell_size) at exact multiples of cell_size -- confirms consistent cell assignment across chunks."""
    cloud = PointCloud()
    cloud.add_chunk(_chunk([0.0, 10.0], [0.0, 10.0], [5.0, 8.0]))  # exactly on boundaries
    cloud.add_chunk(_chunk([9.999, 10.001], [9.999, 10.001], [1.0, 2.0]))

    reference = _reference_ground_z(cloud, cell_size=10.0)
    actual = GridGroundElevationEstimator(cell_size=10.0).estimate(cloud)

    np.testing.assert_allclose(actual, reference)


# ----------------------------------------------------------------------
# Empty cloud -- both paths must raise identically.
# ----------------------------------------------------------------------


def test_empty_cloud_rejected_for_classify() -> None:
    with pytest.raises(ValueError, match="concatenate"):
        GridGroundClassifier(cell_size=1.0).classify(PointCloud())


def test_empty_cloud_rejected_for_estimate() -> None:
    """The exact PR21.7.8 fix: estimate() previously did NOT raise for an empty cloud, unlike classify()."""
    with pytest.raises(ValueError, match="concatenate"):
        GridGroundElevationEstimator(cell_size=1.0).estimate(PointCloud())


# ----------------------------------------------------------------------
# NaN -- confirms the pre-existing (imperfect) behavior is preserved, not newly validated.
# ----------------------------------------------------------------------


def test_nan_coordinates_do_not_crash_matching_prior_silent_behavior() -> None:
    """
    Confirms no NEW validation was introduced: NaN silently
    propagates (with a RuntimeWarning), matching the pre-PR21.7.8
    behavior exactly -- this PR's scope is memory, not adding new
    input validation.
    """
    cloud = PointCloud()
    cloud.add_chunk(_chunk([1.0, 2.0, float("nan")], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))

    with pytest.warns(RuntimeWarning):
        mask = GridGroundClassifier(cell_size=1.0).classify(cloud)

    assert mask.shape == (3,)
