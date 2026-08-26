"""
Regression/coverage suite for topocore.processing.ground.grid_adaptive
-- PR20 coverage phase.

Includes a real, severe bug found and fixed in this session:
_estimate_local_slope() computed
``arctan(sqrt(mean(|dx|)**2 + mean(|dy|)**2))`` over the k-nearest
neighbors -- a formula that NEVER read the Z coordinate at all.
Confirmed directly: a perfectly flat terrain (z=0 everywhere) and a
genuinely 45-degree-sloped terrain, sharing identical XY point
spacing, produced nearly IDENTICAL "estimated slope" (~46.9 deg vs
~47.4 deg), dominated by horizontal point spacing/density rather
than actual terrain steepness. Since this feeds directly into
_compute_adaptive_threshold() (meant to widen the height threshold
on steep slopes), the entire "slope-aware classification" feature
this classifier's own docstring claims to provide was not
functioning -- the threshold was effectively constant regardless of
real terrain slope.

Fixed by reusing compute_pca (the same shared, already-audited
local-plane-fitting primitive used by PCANormalEstimator elsewhere
in this codebase -- not a new, parallel implementation) to fit a
local plane via PCA and derive slope from the plane normal's angle
from vertical. Verified the eigenvector indexing is correct with an
analytically-known case (a perfectly flat plane gives normal exactly
(0,0,1)) before trusting the fix, then reproduced the exact original
decisive comparison, now correctly distinguishing 0 deg (flat) from
45 deg (real slope) with near-exact precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import GroundError
from topocore.processing.ground.grid_adaptive import (
    AdaptiveGridGroundClassifier,
    AdaptiveGridGroundExtractor,
)


def _flat_ground_with_building(spacing: float = 1.0, extent: float = 30.0) -> tuple[PointCloud, int, int]:
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.arange(0, extent, spacing), np.arange(0, extent, spacing))
    ground_x, ground_y = gx.ravel(), gy.ravel()
    ground_z = np.zeros_like(ground_x) + rng.normal(0, 0.02, ground_x.size)

    lo, hi = extent * 0.4, extent * 0.6
    bx, by = np.meshgrid(np.arange(lo, hi, 0.5), np.arange(lo, hi, 0.5))
    building_x, building_y = bx.ravel(), by.ravel()
    building_z = np.full(building_x.size, 4.0)

    xs = np.concatenate([ground_x, building_x])
    ys = np.concatenate([ground_y, building_y])
    zs = np.concatenate([ground_z, building_z])

    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)

    return cloud, len(ground_x), len(building_x)


# ----------------------------------------------------------------------
# The bug: _estimate_local_slope ignoring Z entirely.
# ----------------------------------------------------------------------


def test_eigenvector_indexing_matches_known_flat_plane_normal() -> None:
    """
    Decisive sanity check before trusting the fix: a perfectly flat
    plane's local-PCA normal must be exactly (0, 0, 1), confirming
    pca.eigenvectors[:, 2, 0] genuinely extracts the Z-component of
    the smallest-eigenvalue eigenvector.
    """
    from topocore.processing._shared import compute_pca
    from topocore.processing.neighbors import NeighborhoodManager

    gx, gy = np.meshgrid(np.arange(0, 10, 1.0), np.arange(0, 10, 1.0))
    x, y = gx.ravel().astype(np.float64), gy.ravel().astype(np.float64)
    z = np.zeros_like(x)
    points = np.column_stack((x, y, z))

    manager = NeighborhoodManager.from_array(points)
    pca = compute_pca(manager, k=10)

    normal_z = np.abs(pca.eigenvectors[:, 2, 0])
    np.testing.assert_allclose(normal_z, 1.0, atol=1e-10)


def test_slope_distinguishes_flat_from_steep_terrain() -> None:
    """
    The exact regression: before the fix, flat and 45-degree-sloped
    terrain with identical XY spacing gave nearly identical
    "estimated slope" (~46.9 vs ~47.4 degrees). Now correctly gives
    0 and 45 degrees.
    """
    classifier = AdaptiveGridGroundClassifier()

    gx, gy = np.meshgrid(np.arange(0, 10, 1.0), np.arange(0, 10, 1.0))
    x, y = gx.ravel().astype(np.float64), gy.ravel().astype(np.float64)

    z_flat = np.zeros_like(x)
    z_steep = x.copy() * 1.0  # true 45-degree grade: dz/dx = 1.0

    slope_flat = classifier._estimate_local_slope(x, y, z_flat)
    slope_steep = classifier._estimate_local_slope(x, y, z_steep)

    interior = (x > 1) & (x < 8) & (y > 1) & (y < 8)  # away from grid-edge PCA asymmetry

    assert np.degrees(slope_flat[interior].mean()) == pytest.approx(0.0, abs=0.5)
    assert np.degrees(slope_steep[interior].mean()) == pytest.approx(45.0, abs=0.5)


def test_slope_estimate_handles_degenerate_small_clouds() -> None:
    classifier = AdaptiveGridGroundClassifier()

    x2 = np.array([0.0, 1.0])
    y2 = np.array([0.0, 1.0])
    z2 = np.array([0.0, 5.0])
    slope2 = classifier._estimate_local_slope(x2, y2, z2)  # must not raise
    np.testing.assert_array_equal(slope2, [0.0, 0.0])

    x3 = np.array([0.0, 1.0, 0.0])
    y3 = np.array([0.0, 0.0, 1.0])
    z3 = np.array([0.0, 0.0, 0.0])
    slope3 = classifier._estimate_local_slope(x3, y3, z3)  # must not raise
    assert slope3.shape == (3,)


# ----------------------------------------------------------------------
# End-to-end classification accuracy (both single- and multi-resolution).
# ----------------------------------------------------------------------


def test_multiresolution_classifies_flat_ground_and_building_correctly() -> None:
    cloud, n_ground, _n_building = _flat_ground_with_building()

    classifier = AdaptiveGridGroundClassifier(use_multiresolution=True)
    mask = classifier.classify(cloud)

    assert mask[:n_ground].mean() == pytest.approx(1.0)
    assert (~mask[n_ground:]).mean() == pytest.approx(1.0)


def test_single_resolution_classifies_flat_ground_and_building_correctly() -> None:
    cloud, n_ground, _n_building = _flat_ground_with_building()

    classifier = AdaptiveGridGroundClassifier(use_multiresolution=False)
    mask = classifier.classify(cloud)

    assert mask[:n_ground].mean() == pytest.approx(1.0)
    assert (~mask[n_ground:]).mean() == pytest.approx(1.0)


def test_extractor_returns_only_ground_points() -> None:
    cloud, n_ground, _n_building = _flat_ground_with_building()

    extractor = AdaptiveGridGroundExtractor()
    result = extractor.extract(cloud)

    assert result.point_count == n_ground


def test_classifier_and_extractor_names() -> None:
    assert AdaptiveGridGroundClassifier().name() == "adaptive_grid"
    assert AdaptiveGridGroundExtractor().name() == "adaptive_grid"


# ----------------------------------------------------------------------
# Parameter validation.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_cell_size": -1.0},
        {"base_cell_size": 0.0},
        {"min_cell_size": -1.0},
        {"min_cell_size": 2.0, "base_cell_size": 1.0},  # min > base
        {"max_cell_size": 0.5, "base_cell_size": 1.0},  # max < base
        {"base_height_threshold": -1.0},
        {"slope_threshold": -1.0},
        {"slope_threshold": 91.0},
    ],
)
def test_rejects_invalid_parameters(kwargs: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(GroundError):
        AdaptiveGridGroundClassifier(**kwargs)


def test_boundary_parameters_accepted() -> None:
    # min_cell_size == base_cell_size, max_cell_size == base_cell_size, slope_threshold at the edges
    AdaptiveGridGroundClassifier(base_cell_size=1.0, min_cell_size=1.0, max_cell_size=1.0)
    AdaptiveGridGroundClassifier(slope_threshold=0.0)
    AdaptiveGridGroundClassifier(slope_threshold=90.0)
