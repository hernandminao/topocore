"""
Regression suite for topocore.processing.classification.ml.
MachineLearningClassifier -- PR19.

Includes a SEVERE, systemic bug found and fixed in this session:
MachineLearningClassifier.__init__() created a FeatureManager() with
ZERO feature computers registered. Confirmed directly:
RandomForestClassifier().fit(cloud, labels) -- and by extension
every ML classifier subclassing this base (GradientBoost, XGBoost,
LightGBM) -- failed immediately with ProcessingError: "Feature
'height_above_ground' was not computed", on the very first call.
The entire ML classification subsystem was unusable.

Fixed in two layers:
1. Registered the actual feature computers (HeightFeatureComputer,
   PCAFeatureComputer for curvature/planarity/linearity/sphericity/
   verticality, DensityFeatureComputer) needed for each supported
   feature name.
2. Discovered a second layer of the same underlying problem while
   verifying the first fix: RelativeHeightFeatureComputer (initially
   registered for "height_above_ground") requires the cloud to
   ALREADY carry a CLASSIFICATION attribute identifying ground
   points -- but ML classifiers are meant to operate on RAW,
   unclassified clouds (predicting classification is the whole
   point). Fixed with a new _GroundRelativeHeightFeatureComputer
   that classifies ground geometrically via GroundManager (no
   pre-existing classification needed) -- mirroring the identical
   fix already applied to
   topocore.processing.segmentation.specific (TreeSegmenter/
   BuildingSegmenter) elsewhere in this session. This also made the
   `ground_class` constructor parameter (which named a classification
   VALUE that was never actually usable) dead; replaced with
   `ground_method` (matching the same parameter already used in
   TreeSegmenter/BuildingSegmenter for the same purpose) across the
   base class and all 4 concrete subclasses.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.gradient_boost import GradientBoostClassifier
from topocore.processing.classification.lightgbm import LightGBMClassifier
from topocore.processing.classification.random_forest import RandomForestClassifier
from topocore.processing.classification.xgboost import XGBoostClassifier
from topocore.processing.exceptions import ProcessingError


def _cloud(n: int) -> PointCloud:
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 10, n)
    ys = rng.uniform(0, 10, n)
    zs = rng.uniform(0, 1, n)

    cloud = PointCloud()
    chunk = Chunk(
        size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z]
    )
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


@pytest.fixture
def training_cloud_and_labels() -> tuple[PointCloud, np.ndarray]:
    cloud = _cloud(100)
    labels = np.random.default_rng(1).integers(0, 2, 100)
    return cloud, labels


# ----------------------------------------------------------------------
# The severe bug: unregistered feature computers, fixed for all 4
# concrete implementations (not just one).
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "classifier_class",
    [
        RandomForestClassifier,
        GradientBoostClassifier,
        XGBoostClassifier,
        LightGBMClassifier,
    ],
)
def test_fit_and_classify_work_on_raw_unclassified_cloud(
    classifier_class: type, training_cloud_and_labels: tuple[PointCloud, np.ndarray]
) -> None:
    """
    Before the fix: every one of these raised ProcessingError on the
    very first fit() call, regardless of which concrete algorithm.
    """
    cloud, labels = training_cloud_and_labels

    classifier = classifier_class()
    classifier.fit(cloud, labels)
    result = classifier.classify(cloud)

    assert result.labels.shape == (100,)


def test_default_features_include_height_above_ground(
    training_cloud_and_labels: tuple[PointCloud, np.ndarray],
) -> None:
    """
    The exact reproduction: "height_above_ground" is in the default
    feature set, and previously could never be computed at all.
    """
    cloud, labels = training_cloud_and_labels
    classifier = RandomForestClassifier()

    assert "height_above_ground" in classifier._feature_names
    classifier.fit(cloud, labels)  # must not raise


def test_custom_feature_subset_works() -> None:
    cloud = _cloud(100)
    labels = np.random.default_rng(2).integers(0, 2, 100)

    classifier = RandomForestClassifier(feature_names=["height", "density"])
    classifier.fit(cloud, labels)
    result = classifier.classify(cloud)

    assert result.labels.shape == (100,)


def test_all_pca_derived_features_work_together() -> None:
    cloud = _cloud(100)
    labels = np.random.default_rng(3).integers(0, 2, 100)

    classifier = RandomForestClassifier(
        feature_names=[
            "curvature",
            "planarity",
            "linearity",
            "sphericity",
            "verticality",
        ]
    )
    classifier.fit(cloud, labels)
    result = classifier.classify(cloud)

    assert result.labels.shape == (100,)


# ----------------------------------------------------------------------
# ground_method (replaces the dead ground_class parameter).
# ----------------------------------------------------------------------


def test_ground_method_parameter_is_accepted() -> None:
    cloud = _cloud(100)
    labels = np.random.default_rng(4).integers(0, 2, 100)

    classifier = RandomForestClassifier(
        feature_names=["height_above_ground"], ground_method="grid"
    )
    classifier.fit(cloud, labels)  # must not raise


def test_ground_class_parameter_no_longer_exists() -> None:
    """
    Confirms the dead parameter was genuinely removed, not merely
    ignored -- passing it should be a clear TypeError, not a
    silently-accepted no-op.
    """
    with pytest.raises(TypeError):
        RandomForestClassifier(ground_class="GROUND")  # type: ignore[call-arg]


# ----------------------------------------------------------------------
# Basic validation, unaffected by the fix.
# ----------------------------------------------------------------------


def test_rejects_empty_training_cloud() -> None:
    with pytest.raises(ProcessingError):
        RandomForestClassifier().fit(PointCloud(), np.array([0, 1]))


def test_rejects_mismatched_label_count() -> None:
    cloud = _cloud(100)
    with pytest.raises(ProcessingError):
        RandomForestClassifier().fit(cloud, np.zeros(50, dtype=np.int64))


def test_rejects_single_class_training() -> None:
    cloud = _cloud(100)
    labels = np.zeros(100, dtype=np.int64)
    with pytest.raises(ProcessingError):
        RandomForestClassifier().fit(cloud, labels)


def test_rejects_classify_before_fit() -> None:
    cloud = _cloud(10)
    with pytest.raises(ProcessingError):
        RandomForestClassifier().classify(cloud)


def test_rejects_unknown_feature_name() -> None:
    with pytest.raises(ValueError):
        RandomForestClassifier(feature_names=["not_a_real_feature"])


def test_rejects_k_below_three() -> None:
    with pytest.raises(ValueError):
        RandomForestClassifier(k=2)
