"""
Regression suite for topocore.features.utilities.poles.PoleDetector,
.light_poles.LightPoleDetector, and .signs.SignDetector -- PR19.

All three use ClusterDetectorBase's own shared template method
(none overrides _detect()). Verified point_strategy="min_z" is
correctly implemented (returns the cluster's lowest point -- the
pole base -- not the centroid, which would be ~half the pole's
height off the ground). Confirmed LightPoleDetector/SignDetector
correctly get their own feature_type dynamically through the
inherited template method (unlike CurbDetector, which needed an
explicit replace() rewrite because its parent hardcoded a literal
feature_type) -- no rewrite needed here since PoleDetector's shared
_build_feature() uses self.feature_type, not a literal. Verified
height-range semantic separation between the three detectors (a
short, sign-height cluster is picked up by SignDetector but
correctly rejected by PoleDetector's stricter min_height). No bugs
found.
"""

from __future__ import annotations

import numpy as np

from topocore.features.models import FeatureType
from topocore.features.protocols import DetectionContext
from topocore.features.utilities.light_poles import LightPoleDetector
from topocore.features.utilities.poles import PoleDetector
from topocore.features.utilities.signs import SignDetector
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.classification import PointClassification
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.base import ClassificationResult


def _vertical_column_cloud(height: float, n: int = 30) -> PointCloud:
    rng = np.random.default_rng(0)
    xs = 10.0 + rng.uniform(-0.05, 0.05, n)
    ys = 10.0 + rng.uniform(-0.05, 0.05, n)
    zs = np.linspace(0.0, height, n)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


def _classification(cloud: PointCloud, code: PointClassification) -> ClassificationResult:
    labels = np.full(cloud.point_count, code.value, dtype=np.int64)
    return ClassificationResult(labels=labels, confidence=None, cloud=cloud)


def test_pole_representative_point_is_the_base_not_centroid() -> None:
    """
    The decisive check: point_strategy="min_z" must return the
    lowest point (the pole base, Z~0), not the centroid (which
    would be at roughly half the pole's height, Z~3).
    """
    cloud = _vertical_column_cloud(height=6.0)
    classification = _classification(cloud, PointClassification.UNCLASSIFIED)

    detector = PoleDetector(eps=0.5, min_points=10, min_height=2.0, max_height=12.0)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))

    assert len(result) == 1
    assert result.features[0].geometry.vertices[0, 2] < 0.5  # near the base, not the ~3.0 centroid


def test_pole_produces_point_geometry() -> None:
    cloud = _vertical_column_cloud(height=6.0)
    classification = _classification(cloud, PointClassification.UNCLASSIFIED)

    detector = PoleDetector(eps=0.5, min_points=10)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))

    assert result.features[0].geometry.vertex_count == 1
    assert result.features[0].feature_type == FeatureType.POLE


def test_light_pole_correctly_reports_own_feature_type() -> None:
    """
    Unlike CurbDetector (which needed an explicit rewrite due to its
    parent hardcoding a literal feature_type), LightPoleDetector
    inherits _detect() unmodified and still correctly reports its
    own feature_type -- confirms PoleDetector's shared template uses
    self.feature_type dynamically, not a literal.
    """
    cloud = _vertical_column_cloud(height=6.0)
    classification = _classification(cloud, PointClassification.UNCLASSIFIED)

    detector = LightPoleDetector(eps=0.5, min_points=10)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))

    assert len(result) == 1
    assert result.features[0].feature_type == FeatureType.LIGHT_POLE
    assert result.features[0].feature_type != FeatureType.POLE


def test_sign_detects_short_cluster_pole_detector_rejects_it() -> None:
    """Height-range semantic separation: a sign-height cluster (1.5m) is below PoleDetector's min_height=2.0."""
    cloud = _vertical_column_cloud(height=1.5, n=20)
    classification = _classification(cloud, PointClassification.UNCLASSIFIED)

    sign_result = SignDetector(eps=0.5, min_points=5).detect(
        DetectionContext(cloud=cloud, classification=classification)
    )
    pole_result = PoleDetector(eps=0.5, min_points=5).detect(
        DetectionContext(cloud=cloud, classification=classification)
    )

    assert len(sign_result) == 1
    assert sign_result.features[0].feature_type == FeatureType.SIGN
    assert len(pole_result) == 0


def test_configurable_classification_bucket() -> None:
    cloud = _vertical_column_cloud(height=6.0)
    classification = _classification(cloud, PointClassification.WATER)

    wrong_bucket = PoleDetector(classification_code=PointClassification.UNCLASSIFIED, eps=0.5, min_points=10)
    result_wrong = wrong_bucket.detect(DetectionContext(cloud=cloud, classification=classification))
    assert len(result_wrong) == 0

    right_bucket = PoleDetector(classification_code=PointClassification.WATER, eps=0.5, min_points=10)
    result_right = right_bucket.detect(DetectionContext(cloud=cloud, classification=classification))
    assert len(result_right) == 1


def test_oversized_footprint_rejected() -> None:
    rng = np.random.default_rng(0)
    n = 30
    xs = 10.0 + rng.uniform(-2.0, 2.0, n)  # too wide for a pole (max_horizontal_extent=0.6)
    ys = 10.0 + rng.uniform(-2.0, 2.0, n)
    zs = np.linspace(0.0, 6.0, n)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    classification = _classification(cloud, PointClassification.UNCLASSIFIED)

    detector = PoleDetector(eps=5.0, min_points=10, max_horizontal_extent=0.6)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))
    assert len(result) == 0
