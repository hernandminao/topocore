"""
Regression suite for topocore.features.drainage.manholes.ManholeDetector
and .inspection_chambers.InspectionChamberDetector -- PR19.

Both use ClusterDetectorBase's own shared, correctly-incrementing
template method (neither overrides _detect()) -- verified with a
real oversized cluster rejected mid-sequence, no feature_id gap
(same class of check that found real bugs in
buildings/walls.py and retaining_walls.py, confirmed NOT present
here). Also confirms InspectionChamberDetector correctly overrides
feature_type (not silently inheriting MANHOLE from its parent). No
bugs found.
"""

from __future__ import annotations

import numpy as np

from topocore.features.drainage.inspection_chambers import InspectionChamberDetector
from topocore.features.drainage.manholes import ManholeDetector
from topocore.features.models import FeatureType
from topocore.features.protocols import DetectionContext
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.classification import PointClassification
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.base import ClassificationResult


def _circle_cluster(cx: float, cy: float, radius: float, n: int = 15) -> tuple:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return cx + radius * np.cos(angles), cy + radius * np.sin(angles), np.zeros(n)


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


def _classification(cloud: PointCloud, code: PointClassification) -> ClassificationResult:
    labels = np.full(cloud.point_count, code.value, dtype=np.int64)
    return ClassificationResult(labels=labels, confidence=None, cloud=cloud)


def test_manhole_oversized_cluster_rejected_no_id_gap() -> None:
    c1 = _circle_cluster(0, 0, 0.3)
    c2 = _circle_cluster(50, 50, 2.0)  # too large -- exceeds max_horizontal_extent
    c3 = _circle_cluster(100, 100, 0.3)
    cloud = _build_cloud(c1, c2, c3)
    classification = _classification(cloud, PointClassification.UNCLASSIFIED)

    detector = ManholeDetector(eps=0.5, min_points=5, min_horizontal_extent=0.3, max_horizontal_extent=1.2)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))

    ids = [f.feature_id for f in result.features]
    assert len(result) == 2
    assert ids == [1, 2]


def test_manhole_produces_point_geometry() -> None:
    c1 = _circle_cluster(0, 0, 0.3)
    cloud = _build_cloud(c1)
    classification = _classification(cloud, PointClassification.UNCLASSIFIED)

    detector = ManholeDetector(eps=0.5, min_points=5)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))

    assert len(result) == 1
    assert result.features[0].geometry.vertex_count == 1
    assert result.features[0].feature_type == FeatureType.MANHOLE


def test_manhole_configurable_classification_bucket() -> None:
    c1 = _circle_cluster(0, 0, 0.3)
    cloud = _build_cloud(c1)
    classification = _classification(cloud, PointClassification.WATER)

    detector_wrong_bucket = ManholeDetector(classification_code=PointClassification.UNCLASSIFIED, eps=0.5, min_points=5)
    result_wrong = detector_wrong_bucket.detect(DetectionContext(cloud=cloud, classification=classification))
    assert len(result_wrong) == 0

    detector_right_bucket = ManholeDetector(classification_code=PointClassification.WATER, eps=0.5, min_points=5)
    result_right = detector_right_bucket.detect(DetectionContext(cloud=cloud, classification=classification))
    assert len(result_right) == 1


def test_inspection_chamber_overrides_feature_type_not_manhole() -> None:
    c1 = _circle_cluster(0, 0, 0.6)
    cloud = _build_cloud(c1)
    classification = _classification(cloud, PointClassification.UNCLASSIFIED)

    detector = InspectionChamberDetector(eps=0.8, min_points=8)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))

    assert len(result) == 1
    assert result.features[0].feature_type == FeatureType.INSPECTION_CHAMBER
    assert result.features[0].feature_type != FeatureType.MANHOLE


def test_no_candidate_points_returns_empty_collection() -> None:
    c1 = _circle_cluster(0, 0, 0.3)
    cloud = _build_cloud(c1)
    classification = _classification(cloud, PointClassification.GROUND)  # not the searched bucket

    detector = ManholeDetector(classification_code=PointClassification.UNCLASSIFIED, eps=0.5, min_points=5)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))

    assert len(result) == 0
