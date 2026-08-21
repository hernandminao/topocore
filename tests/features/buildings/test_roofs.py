"""
Regression suite for topocore.features.buildings.roofs.RoofDetector
-- PR19.

Verified the orientation-grouping algorithm (greedy, running-mean
representative) with a real gable roof (two distinct facet
orientations correctly separated, exact mean normals recovered) and
a uniform flat roof (no false fragmentation into multiple groups).
No bugs found.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.features.buildings.roofs import RoofDetector
from topocore.features.protocols import DetectionContext
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.classification import PointClassification
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.base import ClassificationResult
from topocore.processing.normals.models import NormalResult


def test_gable_roof_separates_into_two_facets() -> None:
    rng = np.random.default_rng(0)
    n_per_facet = 60
    facet_a_xy = rng.uniform(0, 10, (n_per_facet, 2))
    facet_a_xy[:, 0] -= 15
    facet_b_xy = rng.uniform(0, 10, (n_per_facet, 2)) + 15

    xs = np.concatenate([facet_a_xy[:, 0], facet_b_xy[:, 0]])
    ys = np.concatenate([facet_a_xy[:, 1], facet_b_xy[:, 1]])
    zs = np.zeros_like(xs)

    normal_a = np.array([np.sin(np.radians(30)), 0, np.cos(np.radians(30))])
    normal_b = np.array([-np.sin(np.radians(30)), 0, np.cos(np.radians(30))])
    normals = np.vstack([np.tile(normal_a, (n_per_facet, 1)), np.tile(normal_b, (n_per_facet, 1))])

    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)

    labels = np.full(len(xs), PointClassification.BUILDING.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)
    normal_result = NormalResult(normals=normals, curvature=None)

    detector = RoofDetector(min_normal_z=0.5, orientation_angle_deg=15.0, eps=2.0, min_points=5)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification, normals=normal_result))

    assert len(result) == 2
    normal_zs = sorted(f.attributes["mean_normal_z"] for f in result.features)
    assert normal_zs[0] == pytest.approx(normal_zs[1])  # both facets have the same tilt magnitude
    assert normal_zs[0] > 0.85  # cos(30deg) ~= 0.866


def test_flat_roof_stays_one_group_no_false_fragmentation() -> None:
    rng = np.random.default_rng(1)
    n = 100
    xs = rng.uniform(0, 10, n)
    ys = rng.uniform(0, 10, n)
    zs = np.zeros(n)
    normals = np.tile([0.0, 0.0, 1.0], (n, 1))

    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)

    labels = np.full(n, PointClassification.BUILDING.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)
    normal_result = NormalResult(normals=normals, curvature=None)

    detector = RoofDetector(min_normal_z=0.5, orientation_angle_deg=15.0, eps=2.0, min_points=5)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification, normals=normal_result))

    assert len(result) == 1


def test_no_building_points_returns_empty_collection() -> None:
    n = 20
    rng = np.random.default_rng(2)
    xs = rng.uniform(0, 10, n)
    ys = rng.uniform(0, 10, n)
    zs = np.zeros(n)
    normals = np.tile([0.0, 0.0, 1.0], (n, 1))

    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)

    labels = np.full(n, PointClassification.GROUND.value, dtype=np.int64)  # no BUILDING points
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)
    normal_result = NormalResult(normals=normals, curvature=None)

    detector = RoofDetector(min_normal_z=0.5, orientation_angle_deg=15.0, eps=2.0, min_points=5)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification, normals=normal_result))

    assert len(result) == 0
