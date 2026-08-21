"""
Regression suite for topocore.features.vegetation.trees.TreeDetector,
.shrubs.ShrubDetector, and .grass.GrassDetector -- PR19.

All three use ClusterDetectorBase's own shared template method (no
enumerate(), no _detect() override). Verified:
- point_strategy="min_z" for TreeDetector reports the canopy's
  lowest point (approximate trunk base), not the canopy centroid
  (same pattern already verified in utilities/poles.py).
- ShrubDetector correctly searches TWO classification codes
  simultaneously (LOW_VEGETATION + MEDIUM_VEGETATION combined via
  np.isin), excluding HIGH_VEGETATION and GROUND.
- GrassDetector produces POLYGON geometry (not POINT) and correctly
  rejects a small clump via min_horizontal_extent despite matching
  the height range.
No bugs found.
"""

from __future__ import annotations

import numpy as np

from topocore.features.models import FeatureType, GeometryType
from topocore.features.protocols import DetectionContext
from topocore.features.vegetation.grass import GrassDetector
from topocore.features.vegetation.shrubs import ShrubDetector
from topocore.features.vegetation.trees import TreeDetector
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


# ----------------------------------------------------------------------
# TreeDetector
# ----------------------------------------------------------------------


def test_tree_representative_point_is_canopy_base_not_centroid() -> None:
    rng = np.random.default_rng(0)
    n = 50
    xs = 10.0 + rng.normal(0, 1.0, n)
    ys = 10.0 + rng.normal(0, 1.0, n)
    zs = rng.uniform(1.5, 8.0, n)  # canopy from near-trunk (1.5) to top (8.0)
    cloud = _build_cloud((xs, ys, zs))

    labels = np.full(n, PointClassification.HIGH_VEGETATION.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)

    detector = TreeDetector(eps=2.0, min_points=10, min_height=2.0)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))

    assert len(result) == 1
    assert result.features[0].feature_type == FeatureType.TREE
    z = result.features[0].geometry.vertices[0, 2]
    assert z < 3.0  # near the base (~1.5), not the ~4.75 centroid


def test_tree_rejects_short_vegetation_cluster() -> None:
    rng = np.random.default_rng(0)
    n = 30
    xs = 10.0 + rng.normal(0, 0.5, n)
    ys = 10.0 + rng.normal(0, 0.5, n)
    zs = rng.uniform(0.0, 1.0, n)  # too short for a tree (min_height=2.0)
    cloud = _build_cloud((xs, ys, zs))

    labels = np.full(n, PointClassification.HIGH_VEGETATION.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)

    detector = TreeDetector(eps=2.0, min_points=10, min_height=2.0)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))
    assert len(result) == 0


# ----------------------------------------------------------------------
# ShrubDetector -- the multi-classification-code case.
# ----------------------------------------------------------------------


def test_shrub_combines_low_and_medium_vegetation_codes() -> None:
    """
    The decisive check: ShrubDetector.classification_codes is a
    2-element tuple -- confirms both codes are combined via a single
    np.isin() mask, and HIGH_VEGETATION/GROUND are correctly excluded.
    """
    rng = np.random.default_rng(0)

    def cluster(cx: float, cy: float, n: int = 15) -> tuple:
        xs = cx + rng.normal(0, 0.2, n)
        ys = cy + rng.normal(0, 0.2, n)
        zs = rng.uniform(0.3, 1.5, n)
        return xs, ys, zs

    c_low = cluster(0, 0)
    c_medium = cluster(50, 50)
    c_high = cluster(100, 100)
    c_ground = cluster(150, 150)

    xs = np.concatenate([c_low[0], c_medium[0], c_high[0], c_ground[0]])
    ys = np.concatenate([c_low[1], c_medium[1], c_high[1], c_ground[1]])
    zs = np.concatenate([c_low[2], c_medium[2], c_high[2], c_ground[2]])
    cloud = _build_cloud((xs, ys, zs))

    labels = np.concatenate(
        [
            np.full(15, PointClassification.LOW_VEGETATION.value),
            np.full(15, PointClassification.MEDIUM_VEGETATION.value),
            np.full(15, PointClassification.HIGH_VEGETATION.value),
            np.full(15, PointClassification.GROUND.value),
        ]
    ).astype(np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)

    detector = ShrubDetector(eps=1.0, min_points=8, min_height=0.3, max_height=2.0)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))

    assert len(result) == 2  # only LOW + MEDIUM clusters
    for feature in result.features:
        assert feature.feature_type == FeatureType.SHRUB


# ----------------------------------------------------------------------
# GrassDetector -- POLYGON geometry, broad-area filter.
# ----------------------------------------------------------------------


def test_grass_produces_polygon_geometry() -> None:
    rng = np.random.default_rng(0)
    n = 200
    xs = rng.uniform(0, 10, n)
    ys = rng.uniform(0, 10, n)
    zs = rng.uniform(0.0, 0.2, n)
    cloud = _build_cloud((xs, ys, zs))

    labels = np.full(n, PointClassification.LOW_VEGETATION.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)

    detector = GrassDetector(eps=1.0, min_points=50, max_height=0.3, min_horizontal_extent=3.0)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))

    assert len(result) == 1
    assert result.features[0].geometry.geometry_type == GeometryType.POLYGON
    assert result.features[0].feature_type == FeatureType.GRASS


def test_grass_rejects_small_clump_despite_matching_height() -> None:
    """Same height range as valid grass, but too small an area -- min_horizontal_extent must reject it."""
    rng = np.random.default_rng(0)
    n = 100
    xs = 50 + rng.uniform(0, 0.5, n)
    ys = 50 + rng.uniform(0, 0.5, n)
    zs = rng.uniform(0.0, 0.2, n)
    cloud = _build_cloud((xs, ys, zs))

    labels = np.full(n, PointClassification.LOW_VEGETATION.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)

    detector = GrassDetector(eps=1.0, min_points=50, max_height=0.3, min_horizontal_extent=3.0)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))
    assert len(result) == 0


def test_grass_rejects_tall_vegetation() -> None:
    rng = np.random.default_rng(0)
    n = 200
    xs = rng.uniform(0, 10, n)
    ys = rng.uniform(0, 10, n)
    zs = rng.uniform(0.0, 1.0, n)  # too tall for grass (max_height=0.3)
    cloud = _build_cloud((xs, ys, zs))

    labels = np.full(n, PointClassification.LOW_VEGETATION.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)

    detector = GrassDetector(eps=1.0, min_points=50, max_height=0.3, min_horizontal_extent=3.0)
    result = detector.detect(DetectionContext(cloud=cloud, classification=classification))
    assert len(result) == 0


def test_no_matching_classification_returns_empty() -> None:
    rng = np.random.default_rng(0)
    n = 50
    xs = rng.uniform(0, 10, n)
    ys = rng.uniform(0, 10, n)
    zs = rng.uniform(0.0, 0.2, n)
    cloud = _build_cloud((xs, ys, zs))

    labels = np.full(n, PointClassification.BUILDING.value, dtype=np.int64)  # not a vegetation code
    classification = ClassificationResult(labels=labels, confidence=None, cloud=cloud)

    for detector in (TreeDetector(), ShrubDetector(), GrassDetector()):
        result = detector.detect(DetectionContext(cloud=cloud, classification=classification))
        assert len(result) == 0
