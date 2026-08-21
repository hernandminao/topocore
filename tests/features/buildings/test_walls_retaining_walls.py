"""
Regression suite for topocore.features.buildings.walls.WallDetector
and .retaining_walls.RetainingWallDetector -- PR19.

Includes a real bug found and fixed in this session: both detectors
override ClusterDetectorBase._detect() with a custom implementation
that used `enumerate(clusters, start=1)` for local_id -- unlike the
shared base class's own template method (used correctly by
BuildingDetector, which does NOT override _detect()), which only
increments local_id on successful addition. A cluster rejected
mid-sequence (by _passes_filters() for WallDetector, or the inline
min_height check for RetainingWallDetector) still consumed a
local_id value, leaving gaps in the surviving features' feature_id
(confirmed directly: [1, 3] instead of [1, 2] for 2 kept clusters out
of 3, with a real short cluster failing RetainingWallDetector's
default-on min_height=0.4m filter). Not currently reachable through
WallDetector's own constructor (it never exposes the optional
ClusterFilterConfig bounds), but genuinely reachable and confirmed
via RetainingWallDetector's own min_height parameter. Fixed in both
to match the established correct increment-on-success pattern.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.features.buildings.retaining_walls import RetainingWallDetector
from topocore.features.buildings.walls import WallDetector
from topocore.features.models import FeatureType
from topocore.features.protocols import DetectionContext
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.classification import PointClassification
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.base import ClassificationResult
from topocore.processing.features.pca import PCAFeatureResult
from topocore.processing.normals.models import NormalResult


def _cluster(cx: float, cy: float, param: float, n: int = 20, mode: str = "height") -> tuple:
    rng = np.random.default_rng(0)
    if mode == "height":
        xs = cx + rng.uniform(-0.3, 0.3, n)
        ys = cy + rng.uniform(-0.3, 0.3, n)
        zs = rng.uniform(0, param, n)
    else:
        xs = cx + rng.uniform(-param / 2, param / 2, n)
        ys = cy + rng.uniform(-param / 2, param / 2, n)
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


# ----------------------------------------------------------------------
# RetainingWallDetector -- the confirmed, reachable regression.
# ----------------------------------------------------------------------


def test_retaining_wall_feature_ids_consecutive_despite_mid_sequence_rejection() -> None:
    """
    The exact regression: before the fix, a short (fails min_height)
    cluster in the middle of the sequence left a gap ([1, 3] instead
    of [1, 2]) in the surviving features' feature_id.
    """
    c1 = _cluster(0, 0, 2.0)
    c2 = _cluster(50, 50, 0.1)  # too short -- fails min_height=0.4
    c3 = _cluster(100, 100, 2.0)
    cloud = _build_cloud(c1, c2, c3)

    labels = np.full(cloud.point_count, PointClassification.GROUND.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)
    verticality = np.full(cloud.point_count, 0.9)
    pca_features: PCAFeatureResult = {"verticality": verticality}  # type: ignore[typeddict-item]

    detector = RetainingWallDetector(min_verticality=0.5, eps=1.0, min_points=5, min_height=0.4)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification, pca_features=pca_features))

    ids = [f.feature_id for f in result.features]
    assert len(ids) == 2
    assert ids == [1, 2]


def test_retaining_wall_detects_vertical_ground_structures() -> None:
    c1 = _cluster(0, 0, 2.0)
    cloud = _build_cloud(c1)
    labels = np.full(cloud.point_count, PointClassification.GROUND.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)
    pca_features: PCAFeatureResult = {"verticality": np.full(cloud.point_count, 0.9)}  # type: ignore[typeddict-item]

    detector = RetainingWallDetector(min_verticality=0.5, eps=1.0, min_points=5, min_height=0.4)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification, pca_features=pca_features))

    assert len(result) == 1
    assert result.features[0].feature_type == FeatureType.RETAINING_WALL


def test_retaining_wall_ignores_non_vertical_points() -> None:
    c1 = _cluster(0, 0, 2.0)
    cloud = _build_cloud(c1)
    labels = np.full(cloud.point_count, PointClassification.GROUND.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)
    pca_features: PCAFeatureResult = {  # type: ignore[typeddict-item]
        "verticality": np.full(cloud.point_count, 0.1)
    }  # below min_verticality

    detector = RetainingWallDetector(min_verticality=0.5, eps=1.0, min_points=5, min_height=0.4)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification, pca_features=pca_features))

    assert len(result) == 0


def test_retaining_wall_rejects_invalid_min_verticality() -> None:
    from topocore.features.exceptions import DetectionError

    with pytest.raises(DetectionError):
        RetainingWallDetector(min_verticality=1.5)


# ----------------------------------------------------------------------
# WallDetector.
# ----------------------------------------------------------------------


def test_wall_detects_building_points_with_near_horizontal_normal() -> None:
    c1 = _cluster(0, 0, 2.0)
    cloud = _build_cloud(c1)
    labels = np.full(cloud.point_count, PointClassification.BUILDING.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)
    normals = np.tile([1.0, 0.0, 0.0], (cloud.point_count, 1))  # perfectly vertical wall
    normal_result = NormalResult(normals=normals, curvature=None)

    detector = WallDetector(max_normal_z=0.3, eps=1.0, min_points=5)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification, normals=normal_result))

    assert len(result) == 1
    assert result.features[0].feature_type == FeatureType.WALL


def test_wall_ignores_roof_like_horizontal_normals() -> None:
    c1 = _cluster(0, 0, 2.0)
    cloud = _build_cloud(c1)
    labels = np.full(cloud.point_count, PointClassification.BUILDING.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)
    normals = np.tile([0.0, 0.0, 1.0], (cloud.point_count, 1))  # horizontal, roof-like
    normal_result = NormalResult(normals=normals, curvature=None)

    detector = WallDetector(max_normal_z=0.3, eps=1.0, min_points=5)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification, normals=normal_result))

    assert len(result) == 0


def test_wall_feature_ids_still_consecutive_across_multiple_clusters() -> None:
    c1 = _cluster(0, 0, 2.0)
    c2 = _cluster(50, 50, 2.0)
    cloud = _build_cloud(c1, c2)
    labels = np.full(cloud.point_count, PointClassification.BUILDING.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)
    normals = np.tile([1.0, 0.0, 0.0], (cloud.point_count, 1))
    normal_result = NormalResult(normals=normals, curvature=None)

    detector = WallDetector(max_normal_z=0.3, eps=1.0, min_points=5)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification, normals=normal_result))

    ids = [f.feature_id for f in result.features]
    assert ids == list(range(1, len(ids) + 1))
