"""
Regression suite for topocore.analysis.quality.c2c.CloudToCloudDistance
and .c2m.CloudToMeshDistance -- PR19.

This module was synced mid-session against Hernán's own corrected
real repository. Both c2c.py and c2m.py were changed from EXCLUDING
unmatched points (beyond max_distance) from summary statistics, to
REJECTING the whole computation outright when any point has no
correspondence -- with explicit reasoning: "Silently dropping points
outside max_distance would make the quality result look better than
the actual coverage of the correspondence." My own earlier audit had
verified the OLD (exclude) behavior as internally consistent and
reasonable, without recognizing that for a QUALITY metric
specifically, "no match found" is itself a quality signal that must
remain visible, not averaged away.

c2m's new per-instance _cache_tin_id/_cache_index caching (keyed by
id(tin)) was reviewed and judged a different, lower-risk pattern than
the id()-based bugs found elsewhere this session (compute_pca,
various managers): it's a private, per-instance cache (not shared
globally), TINs are normally long-lived caller-managed objects (not
ephemeral per-call constructs), and the residual risk is explicitly
documented in the source. Verified directly that switching to a
genuinely different TIN correctly invalidates and rebuilds the cache.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import topocore.analysis.quality.c2c as c2c_module
from topocore.analysis.exceptions import QualityError
from topocore.analysis.quality.c2c import CloudToCloudDistance
from topocore.analysis.quality.c2m import CloudToMeshDistance
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN

# ----------------------------------------------------------------------
# CloudToCloudDistance -- the exclude -> reject correction.
# ----------------------------------------------------------------------


def test_c2c_rejects_when_any_point_unmatched() -> None:
    """
    The exact regression: before the fix, an unmatched point (beyond
    max_distance) was silently excluded from the mean/std/etc, now
    it correctly aborts the whole computation instead.
    """
    reference = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    compared = np.array([[0.1, 0.0, 0.0], [10.2, 0.0, 0.0]])  # no match near (100,0,0)

    with pytest.raises(QualityError):
        CloudToCloudDistance(max_distance=5.0).compute(reference, compared)


def test_c2c_succeeds_when_every_point_matched() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    compared = np.array([[0.1, 0.0, 0.0], [10.2, 0.0, 0.0]])

    result = CloudToCloudDistance(max_distance=5.0).compute(reference, compared)
    assert result.mean == pytest.approx(0.15, abs=1e-6)


def test_c2c_kdtree_and_bruteforce_paths_agree_on_rejection() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    compared = np.array([[0.1, 0.0, 0.0]])

    with pytest.raises(QualityError):
        CloudToCloudDistance(max_distance=5.0).compute(reference, compared)

    original = c2c_module._HAS_SCIPY
    c2c_module._HAS_SCIPY = False
    try:
        with pytest.raises(QualityError):
            CloudToCloudDistance(max_distance=5.0).compute(reference, compared)
    finally:
        c2c_module._HAS_SCIPY = original


def test_c2c_rejects_nonfinite_max_distance() -> None:
    with pytest.raises(QualityError):
        CloudToCloudDistance(max_distance=float("inf"))


def test_c2c_rejects_empty_cloud() -> None:
    with pytest.raises(QualityError):
        CloudToCloudDistance().compute(np.empty((0, 3)), np.array([[1.0, 1.0, 1.0]]))


def test_c2c_rejects_wrong_shape() -> None:
    with pytest.raises(QualityError):
        CloudToCloudDistance().compute(np.array([[1.0, 1.0]]), np.array([[1.0, 1.0, 1.0]]))


# ----------------------------------------------------------------------
# CloudToMeshDistance -- core geometric primitive (unaffected by the
# exclude->reject change in spirit, but same max_distance correction
# applied) plus the new per-instance TIN cache.
# ----------------------------------------------------------------------


def test_point_above_centroid_gives_perpendicular_distance() -> None:
    v0, v1, v2 = (
        np.array([0.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
        np.array([0.0, 3.0, 0.0]),
    )
    centroid = (v0 + v1 + v2) / 3
    point = centroid + np.array([0.0, 0.0, 5.0])

    distance = CloudToMeshDistance._point_to_triangle_distance(point, v0, v1, v2)
    assert distance == pytest.approx(5.0)


def test_point_outside_projection_gives_nearest_vertex_distance() -> None:
    v0, v1, v2 = (
        np.array([0.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
        np.array([0.0, 3.0, 0.0]),
    )
    point = np.array([-10.0, -10.0, 0.0])

    distance = CloudToMeshDistance._point_to_triangle_distance(point, v0, v1, v2)
    assert distance == pytest.approx(math.sqrt(200))


def test_point_near_edge_gives_perpendicular_edge_distance() -> None:
    v0, v1, v2 = (
        np.array([0.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
        np.array([0.0, 3.0, 0.0]),
    )
    point = np.array([2.0, -3.0, 0.0])

    distance = CloudToMeshDistance._point_to_triangle_distance(point, v0, v1, v2)
    assert distance == pytest.approx(3.0)


def test_cloud_to_mesh_end_to_end() -> None:
    points = (
        Point3D(-50, -50, 0.0),
        Point3D(50, -50, 0.0),
        Point3D(-50, 50, 0.0),
        Point3D(50, 50, 0.0),
    )
    tin = TIN.from_points(points)

    cloud = np.array([[0.0, 0.0, 5.0], [10.0, 10.0, 3.0], [-20.0, -20.0, 0.0]])
    result = CloudToMeshDistance().compute(cloud, tin)

    np.testing.assert_allclose(result.distances, [5.0, 3.0, 0.0])


def test_cloud_to_mesh_rejects_when_point_unmatched() -> None:
    points = (
        Point3D(-50, -50, 0.0),
        Point3D(50, -50, 0.0),
        Point3D(-50, 50, 0.0),
        Point3D(50, 50, 0.0),
    )
    tin = TIN.from_points(points)

    cloud = np.array([[0.0, 0.0, 5.0], [1000.0, 1000.0, 3.0]])  # far outside the mesh footprint

    with pytest.raises(QualityError):
        CloudToMeshDistance(max_distance=10.0).compute(cloud, tin)


def test_cloud_to_mesh_rejects_empty_cloud() -> None:
    points = (
        Point3D(-10, -10, 0.0),
        Point3D(10, -10, 0.0),
        Point3D(-10, 10, 0.0),
        Point3D(10, 10, 0.0),
    )
    tin = TIN.from_points(points)

    with pytest.raises(QualityError):
        CloudToMeshDistance().compute(np.empty((0, 3)), tin)


def test_cloud_to_mesh_cache_rebuilds_for_a_different_tin() -> None:
    """
    Confirms the new per-instance TIN cache correctly distinguishes
    between two DIFFERENT TIN objects reused on the SAME
    CloudToMeshDistance instance -- not a stale reuse of the first
    TIN's spatial index.
    """
    points_low = (
        Point3D(-50, -50, 0.0),
        Point3D(50, -50, 0.0),
        Point3D(-50, 50, 0.0),
        Point3D(50, 50, 0.0),
    )
    tin_low = TIN.from_points(points_low)

    points_high = (
        Point3D(-50, -50, 20.0),
        Point3D(50, -50, 20.0),
        Point3D(-50, 50, 20.0),
        Point3D(50, 50, 20.0),
    )
    tin_high = TIN.from_points(points_high)

    c2m = CloudToMeshDistance()
    cloud = np.array([[0.0, 0.0, 5.0]])

    result_low = c2m.compute(cloud, tin_low)
    result_high = c2m.compute(cloud, tin_high)  # same instance, different TIN

    assert result_low.distances[0] == pytest.approx(5.0)
    assert result_high.distances[0] == pytest.approx(15.0)  # |5 - 20|


def test_cloud_to_mesh_cache_hit_gives_identical_repeated_result() -> None:
    points = (
        Point3D(-50, -50, 0.0),
        Point3D(50, -50, 0.0),
        Point3D(-50, 50, 0.0),
        Point3D(50, 50, 0.0),
    )
    tin = TIN.from_points(points)
    cloud = np.array([[0.0, 0.0, 5.0]])

    c2m = CloudToMeshDistance()
    result1 = c2m.compute(cloud, tin)
    result2 = c2m.compute(cloud, tin)

    np.testing.assert_array_equal(result1.distances, result2.distances)
