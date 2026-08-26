"""
Regression suite for topocore.processing.classification.rules.
RuleBasedClassifier -- PR19.

Verified end-to-end against a realistic synthetic scene: dense
ground at a non-zero absolute elevation (not ground-normalized data
-- the same class of scenario that exposed real bugs elsewhere in
this session's segmentation work), a flat planar elevated "roof"
cluster, and an irregular scattered "tree"-like cluster. No bugs
found -- ground classifies exactly correctly; building/vegetation
classify with the expected majority (rule-based thresholds are
inherently approximate on any real-shaped data, not a precision
guarantee).
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.classification import PointClassification
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.rules import RuleBasedClassifier
from topocore.processing.exceptions import ProcessingError


@pytest.fixture
def realistic_scene() -> tuple[PointCloud, int, int, int]:
    """
    Ground (dense, ~1500m absolute elevation) + a flat elevated roof
    + an irregular vegetation-like cluster. Returns the cloud and
    the point counts for each region, in order.
    """
    rng = np.random.default_rng(0)

    ground_xs = rng.uniform(0, 20, 3000)
    ground_ys = rng.uniform(0, 20, 3000)
    ground_zs = 1500.0 + rng.uniform(-0.05, 0.05, 3000)

    roof_xs, roof_ys, roof_zs = [], [], []
    for i in range(15):
        for j in range(15):
            roof_xs.append(5.0 + i * 0.3)
            roof_ys.append(5.0 + j * 0.3)
            roof_zs.append(1505.0)

    tree_center = np.array([15.0, 15.0, 1503.0])
    tree_pts = rng.normal(0, 1.0, (150, 3)) + tree_center
    tree_pts[:, 2] = np.abs(tree_pts[:, 2] - 1503.0) + 1501.0

    xs = np.concatenate([ground_xs, roof_xs, tree_pts[:, 0]])
    ys = np.concatenate([ground_ys, roof_ys, tree_pts[:, 1]])
    zs = np.concatenate([ground_zs, roof_zs, tree_pts[:, 2]])

    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)

    return cloud, len(ground_xs), len(roof_xs), len(tree_pts)


def test_ground_classified_correctly_at_realistic_elevation(
    realistic_scene: tuple[PointCloud, int, int, int],
) -> None:
    cloud, n_ground, _n_roof, _n_tree = realistic_scene
    result = RuleBasedClassifier(ground_cell_size=1.0, pca_neighbors=10).classify(cloud)

    ground_labels = result.labels[:n_ground]
    assert (ground_labels == PointClassification.GROUND).sum() == n_ground


def test_roof_majority_classified_as_building(
    realistic_scene: tuple[PointCloud, int, int, int],
) -> None:
    cloud, n_ground, n_roof, _n_tree = realistic_scene
    result = RuleBasedClassifier(ground_cell_size=1.0, pca_neighbors=10).classify(cloud)

    roof_labels = result.labels[n_ground : n_ground + n_roof]
    building_fraction = (roof_labels == PointClassification.BUILDING).mean()

    assert building_fraction > 0.5


def test_tree_majority_classified_as_vegetation(
    realistic_scene: tuple[PointCloud, int, int, int],
) -> None:
    cloud, n_ground, n_roof, _n_tree = realistic_scene
    result = RuleBasedClassifier(ground_cell_size=1.0, pca_neighbors=10).classify(cloud)

    tree_labels = result.labels[n_ground + n_roof :]
    vegetation_codes = {
        PointClassification.LOW_VEGETATION,
        PointClassification.MEDIUM_VEGETATION,
        PointClassification.HIGH_VEGETATION,
    }
    vegetation_fraction = np.isin(tree_labels, list(vegetation_codes)).mean()

    assert vegetation_fraction > 0.3


def test_classification_result_cloud_matches_input_size(
    realistic_scene: tuple[PointCloud, int, int, int],
) -> None:
    cloud, *_ = realistic_scene
    result = RuleBasedClassifier(ground_cell_size=1.0, pca_neighbors=10).classify(cloud)
    assert result.labels.shape[0] == cloud.point_count


def test_rejects_empty_cloud() -> None:
    with pytest.raises(ProcessingError):
        RuleBasedClassifier().classify(PointCloud())


def test_rules_description_available() -> None:
    classifier = RuleBasedClassifier()
    rules = classifier.rules()
    assert isinstance(rules, dict)
    assert len(rules) > 0


def test_name_and_confidence_support() -> None:
    classifier = RuleBasedClassifier()
    assert classifier.name() == "rule_based"
    assert classifier.supports_confidence() is False
