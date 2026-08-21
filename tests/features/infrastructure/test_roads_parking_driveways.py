"""
Regression suite for topocore.features.infrastructure.roads.RoadDetector,
.parking.ParkingDetector, and .driveways.DrivewayDetector -- PR19.

All three share the same ROAD_SURFACE classification code and use
ClusterDetectorBase's own shared, correctly-incrementing template
method (none overrides _detect()) -- same feature_id-gap check
already confirmed clean here (same class of check that found real
bugs in buildings/walls.py and retaining_walls.py). Verified
RoadDetector (min_elongation) and ParkingDetector (max_elongation)
correctly complement each other with no overlap for a clear
elongated-vs-compact pair of clusters, and DrivewayDetector's
combined elongation+extent bounds correctly reject an oversized
cluster. The partial threshold overlap between all three detectors
is confirmed to be an explicitly documented, deliberate design
choice ("candidate separation, not definitive classification"), not
a bug. No bugs found.
"""

from __future__ import annotations

import numpy as np

from topocore.features.infrastructure.driveways import DrivewayDetector
from topocore.features.infrastructure.parking import ParkingDetector
from topocore.features.infrastructure.roads import RoadDetector
from topocore.features.models import FeatureType
from topocore.features.protocols import DetectionContext
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.classification import PointClassification
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.base import ClassificationResult


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


def _road_surface_classification(cloud: PointCloud) -> ClassificationResult:
    labels = np.full(cloud.point_count, PointClassification.ROAD_SURFACE.value, dtype=np.int64)
    return ClassificationResult(labels=labels, confidence=None, cloud=cloud)


def test_road_and_parking_detectors_complement_without_overlap() -> None:
    rng = np.random.default_rng(0)
    road_xs = rng.uniform(0, 100, 200)
    road_ys = rng.uniform(0, 5, 200)
    park_xs = rng.uniform(500, 520, 200)
    park_ys = rng.uniform(500, 520, 200)

    cloud = _build_cloud(
        (road_xs, road_ys, np.zeros(200)),
        (park_xs, park_ys, np.zeros(200)),
    )
    classification = _road_surface_classification(cloud)

    road_result = RoadDetector(eps=3.0, min_points=50, min_elongation=2.0).detect(
        DetectionContext(cloud=cloud, classification=classification)
    )
    parking_result = ParkingDetector(eps=3.0, min_points=50, max_elongation=2.0).detect(
        DetectionContext(cloud=cloud, classification=classification)
    )

    assert len(road_result) == 1
    assert road_result.features[0].feature_type == FeatureType.ROAD
    assert len(parking_result) == 1
    assert parking_result.features[0].feature_type == FeatureType.PARKING


def test_road_detector_rejects_compact_cluster() -> None:
    rng = np.random.default_rng(1)
    xs = rng.uniform(0, 20, 100)
    ys = rng.uniform(0, 20, 100)  # compact, elongation ~1.0
    cloud = _build_cloud((xs, ys, np.zeros(100)))
    classification = _road_surface_classification(cloud)

    result = RoadDetector(eps=3.0, min_points=50, min_elongation=2.0).detect(
        DetectionContext(cloud=cloud, classification=classification)
    )
    assert len(result) == 0


def test_driveway_rejects_oversized_cluster() -> None:
    rng = np.random.default_rng(0)
    driveway_xs = rng.uniform(0, 15, 60)
    driveway_ys = rng.uniform(0, 4, 60)
    oversized_xs = rng.uniform(500, 600, 60)
    oversized_ys = rng.uniform(500, 504, 60)

    cloud = _build_cloud(
        (driveway_xs, driveway_ys, np.zeros(60)),
        (oversized_xs, oversized_ys, np.zeros(60)),
    )
    classification = _road_surface_classification(cloud)

    result = DrivewayDetector(eps=2.0, min_points=20).detect(
        DetectionContext(cloud=cloud, classification=classification)
    )

    assert len(result) == 1
    assert result.features[0].feature_type == FeatureType.DRIVEWAY


def test_no_road_surface_points_returns_empty() -> None:
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 20, 50)
    ys = rng.uniform(0, 20, 50)
    cloud = _build_cloud((xs, ys, np.zeros(50)))
    labels = np.full(cloud.point_count, PointClassification.GROUND.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)

    for detector in (RoadDetector(), ParkingDetector(), DrivewayDetector()):
        result = detector.detect(DetectionContext(cloud=cloud, classification=classification))
        assert len(result) == 0
