"""
Regression suite for topocore.features.buildings.buildings.BuildingDetector
-- PR19.

Verified this detector correctly uses ClusterDetectorBase's own
shared, correctly-incrementing template method (does not override
_detect() at all, unlike WallDetector/RetainingWallDetector) --
confirmed with a real noise-sized cluster rejected mid-sequence by
the active min_horizontal_extent filter, with no feature_id gap. No
bugs found.
"""

from __future__ import annotations

import numpy as np

from topocore.features.buildings.buildings import BuildingDetector
from topocore.features.protocols import DetectionContext
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.classification import PointClassification
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.base import ClassificationResult


def _cluster(cx: float, cy: float, extent: float, n: int = 25) -> tuple:
    rng = np.random.default_rng(1)
    xs = cx + rng.uniform(-extent / 2, extent / 2, n)
    ys = cy + rng.uniform(-extent / 2, extent / 2, n)
    zs = np.zeros(n)
    return xs, ys, zs


def _build_cloud(*clusters: tuple) -> PointCloud:
    xs = np.concatenate([c[0] for c in clusters])
    ys = np.concatenate([c[1] for c in clusters])
    zs = np.concatenate([c[2] for c in clusters])
    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


def test_noise_cluster_rejected_no_id_gap() -> None:
    c1 = _cluster(0, 0, 10.0)
    c2 = _cluster(50, 50, 0.5)  # noise, below min_horizontal_extent=2.0
    c3 = _cluster(100, 100, 10.0)
    cloud = _build_cloud(c1, c2, c3)

    labels = np.full(cloud.point_count, PointClassification.BUILDING.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)

    detector = BuildingDetector(eps=1.0, min_points=5, min_horizontal_extent=2.0, max_horizontal_extent=200.0)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))

    ids = [f.feature_id for f in result.features]
    assert len(result) == 2
    assert ids == [1, 2]


def test_oversized_cluster_rejected() -> None:
    c1 = _cluster(0, 0, 10.0)
    c2 = _cluster(500, 500, 500.0)  # implausibly large, above max_horizontal_extent=200
    cloud = _build_cloud(c1, c2)

    labels = np.full(cloud.point_count, PointClassification.BUILDING.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)

    detector = BuildingDetector(eps=5.0, min_points=5, min_horizontal_extent=2.0, max_horizontal_extent=200.0)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))

    assert len(result) == 1
