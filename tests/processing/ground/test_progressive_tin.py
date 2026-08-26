"""
Regression/coverage suite for topocore.processing.ground.progressive_tin
-- PR20 coverage phase.

ProgressiveTINGroundClassifier implements Axelsson (2000)'s
Progressive TIN Densification algorithm. Every test targets real
domain behavior:

- _compute_normal's cross-product winding is confirmed consistently
  upward-pointing for a real, multi-triangle flat TIN (32 triangles,
  ALL normal_z == 1.0 exactly) -- the same winding-consistency check
  that mattered for terrain/breaklines.py earlier in this session.
  Without this, _check_angle's `normal[2] >= cos(max_angle)` formula
  would incorrectly reject ground points purely due to triangle
  winding, not real terrain steepness.
- _check_angle verified against a real 45-degree triangle (correctly
  rejected at max_angle=30) and a flat triangle (correctly accepted),
  plus the degenerate collinear-points case (zero-area triangle,
  confirmed no division-by-zero -- returns a zero vector safely).
- End-to-end classify(): flat ground + elevated building, ~98%
  ground accuracy, 100% building rejection.
- _get_seeds: verified the lowest point per grid cell is selected
  with a hand-computed 2-cell case.
- The <3-ground-points early-break path (a 2-point cloud, both
  becoming seeds in separate cells, correctly returned as ground
  without ever attempting TIN construction).
- Convergence: confirmed exact ground-point COUNTS (not just
  percentages) are identical whether max_iterations=1, 2, or 3 for
  a scene that converges after the first productive pass -- this is
  expected, correct early-stopping behavior for this class of
  convergence-detection algorithm (one extra pass is needed to
  CONFIRM no more points can be added), not a bug.

No bugs found -- this module was already correct; only test
coverage was added.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.geometry.point3d import Point3D
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import GroundError
from topocore.processing.ground.progressive_tin import (
    ProgressiveTINGroundClassifier,
    ProgressiveTINGroundExtractor,
)
from topocore.terrain.tin import TIN


def _flat_ground_with_building() -> tuple[PointCloud, int, int]:
    gx, gy = np.meshgrid(np.arange(0, 30, 1.0), np.arange(0, 30, 1.0))
    ground_x, ground_y = gx.ravel(), gy.ravel()
    rng = np.random.default_rng(0)
    ground_z = np.zeros_like(ground_x) + rng.normal(0, 0.02, ground_x.size)

    bx, by = np.meshgrid(np.arange(12, 18, 0.5), np.arange(12, 18, 0.5))
    building_x, building_y = bx.ravel(), by.ravel()
    building_z = np.full(building_x.size, 5.0)

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
# _compute_normal -- winding consistency (the critical geometric check).
# ----------------------------------------------------------------------


def test_flat_multi_triangle_tin_has_consistent_upward_normals() -> None:
    rng = np.random.default_rng(0)
    points = [
        Point3D(0, 0, 5.0),
        Point3D(20, 0, 5.0),
        Point3D(0, 20, 5.0),
        Point3D(20, 20, 5.0),
    ]
    for _ in range(15):
        points.append(Point3D(float(rng.uniform(0, 20)), float(rng.uniform(0, 20)), 5.0))
    tin = TIN.from_points(tuple(points))

    classifier = ProgressiveTINGroundClassifier()
    normal_zs = []
    for i in range(tin.triangle_count):
        p1, p2, p3 = tin.triangle_vertices(i)
        normal = classifier._compute_normal(p1, p2, p3)
        normal_zs.append(normal[2])

    normal_zs_arr = np.array(normal_zs)
    assert (normal_zs_arr > 0).all()
    assert normal_zs_arr == pytest.approx(1.0)


def test_compute_normal_handles_degenerate_collinear_triangle() -> None:
    classifier = ProgressiveTINGroundClassifier()
    p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(2, 0, 0)
    normal = classifier._compute_normal(p1, p2, p3)
    assert normal == pytest.approx([0.0, 0.0, 0.0])


# ----------------------------------------------------------------------
# _check_angle
# ----------------------------------------------------------------------


def test_check_angle_rejects_steep_triangle() -> None:
    classifier = ProgressiveTINGroundClassifier(max_angle=30.0)
    p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 0, 1), Point3D(0, 1, 0)  # 45 degrees
    normal = classifier._compute_normal(p1, p2, p3)
    assert normal[2] == pytest.approx(np.cos(np.radians(45)))
    assert classifier._check_angle(normal) is False


def test_check_angle_accepts_flat_triangle() -> None:
    classifier = ProgressiveTINGroundClassifier(max_angle=30.0)
    p1, p2, p3 = Point3D(0, 0, 5), Point3D(1, 0, 5), Point3D(0, 1, 5)
    normal = classifier._compute_normal(p1, p2, p3)
    assert classifier._check_angle(normal) is True


# ----------------------------------------------------------------------
# _get_seeds
# ----------------------------------------------------------------------


def test_get_seeds_picks_lowest_point_per_cell() -> None:
    classifier = ProgressiveTINGroundClassifier(cell_size=1.0)
    x = np.array([0.2, 0.8, 1.2, 1.8])
    y = np.array([0.2, 0.2, 0.2, 0.2])
    z = np.array([5.0, 2.0, 8.0, 3.0])  # cell (0,0): min at idx1; cell (1,0): min at idx3

    seeds = classifier._get_seeds(x, y, z)
    assert list(seeds) == [False, True, False, True]


# ----------------------------------------------------------------------
# End-to-end classify() / extractor.
# ----------------------------------------------------------------------


def test_flat_ground_and_building_classified_with_high_accuracy() -> None:
    cloud, n_ground, _n_building = _flat_ground_with_building()

    classifier = ProgressiveTINGroundClassifier(cell_size=2.0, max_distance=0.3, max_angle=20.0)
    mask = classifier.classify(cloud)

    assert mask[:n_ground].mean() > 0.95
    assert (~mask[n_ground:]).mean() == pytest.approx(1.0)


def test_extractor_returns_only_ground_points() -> None:
    cloud, _n_ground, _n_building = _flat_ground_with_building()

    extractor = ProgressiveTINGroundExtractor(cell_size=2.0)
    result = extractor.extract(cloud)

    assert 0 < result.point_count <= cloud.point_count


def test_classifier_and_extractor_names() -> None:
    assert ProgressiveTINGroundClassifier().name() == "progressive_tin"
    assert ProgressiveTINGroundExtractor().name() == "progressive_tin"


# ----------------------------------------------------------------------
# Convergence -- confirms exact ground counts, not just percentages,
# are stable once converged (expected early-stop behavior, not a bug).
# ----------------------------------------------------------------------


def test_convergence_gives_identical_ground_counts_across_iteration_limits() -> None:
    gx, gy = np.meshgrid(np.arange(0, 20, 0.5), np.arange(0, 20, 0.5))
    ground_x, ground_y = gx.ravel(), gy.ravel()
    rng = np.random.default_rng(1)
    ground_z = np.zeros_like(ground_x) + rng.normal(0, 0.05, ground_x.size)

    cloud = PointCloud()
    chunk = Chunk(
        size=len(ground_x),
        attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z],
    )
    chunk[PointAttribute.X][:] = ground_x
    chunk[PointAttribute.Y][:] = ground_y
    chunk[PointAttribute.Z][:] = ground_z
    cloud.add_chunk(chunk)

    counts = []
    for n_iter in (1, 2, 3):
        classifier = ProgressiveTINGroundClassifier(cell_size=3.0, max_distance=0.3, max_iterations=n_iter)
        mask = classifier.classify(cloud)
        counts.append(int(mask.sum()))

    assert counts[0] == counts[1] == counts[2]


# ----------------------------------------------------------------------
# Degenerate inputs and validation.
# ----------------------------------------------------------------------


def test_fewer_than_three_ground_points_breaks_iteration_loop_early() -> None:
    """2 points in separate cells become 2 seeds; the <3-points guard prevents any TIN construction attempt."""
    cloud = PointCloud()
    chunk = Chunk(size=2, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [0.0, 100.0]
    chunk[PointAttribute.Y][:] = [0.0, 100.0]
    chunk[PointAttribute.Z][:] = [1.0, 2.0]
    cloud.add_chunk(chunk)

    mask = ProgressiveTINGroundClassifier().classify(cloud)
    assert list(mask) == [True, True]


def test_rejects_empty_cloud() -> None:
    with pytest.raises(GroundError):
        ProgressiveTINGroundClassifier().classify(PointCloud())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cell_size": -1.0},
        {"max_distance": 0.0},
        {"max_angle": -1.0},
        {"max_iterations": 0},
    ],
)
def test_rejects_invalid_parameters(kwargs: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(GroundError):
        ProgressiveTINGroundClassifier(**kwargs)
