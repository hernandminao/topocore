"""
Regression suite for topocore.processing.classification.manager.
ClassificationManager -- PR19.

Includes a real bug found and fixed in this session, discovered
while verifying the earlier MachineLearningClassifier feature-
registration fix: save() pickles the entire trained classifier,
including its internal FeatureManager's LRUCache (which holds a
threading.RLock -- see the dedicated LRUCache regression suite for
the actual fix). Confirmed here at the full end-to-end
fit -> save -> load -> classify level.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.manager import ClassificationManager
from topocore.processing.exceptions import ClassificationError, ProcessingError


def _cloud(n: int) -> PointCloud:
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 10, n)
    ys = rng.uniform(0, 10, n)
    zs = rng.uniform(0, 1, n)

    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
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
# End-to-end: auto resolution, fit, classify, save, load -- the exact
# reproduction that exposed both the feature-registration bug and the
# LRUCache pickling bug in this session.
# ----------------------------------------------------------------------


def test_auto_method_resolves_and_trains(
    training_cloud_and_labels: tuple[PointCloud, np.ndarray],
) -> None:
    cloud, labels = training_cloud_and_labels
    manager = ClassificationManager(method="auto")

    assert manager.method == "auto"
    manager.fit(cloud, labels)

    assert manager.method in ("xgboost", "lightgbm", "gradient_boost", "random_forest")
    assert manager.is_trained()


def test_classify_after_fit_works(
    training_cloud_and_labels: tuple[PointCloud, np.ndarray],
) -> None:
    cloud, labels = training_cloud_and_labels
    manager = ClassificationManager(method="auto")
    manager.fit(cloud, labels)

    result = manager.classify(cloud)
    assert result.labels.shape == (100,)


def test_save_and_load_round_trip(
    training_cloud_and_labels: tuple[PointCloud, np.ndarray],
) -> None:
    """
    The exact regression: before the LRUCache fix, save() raised
    TypeError: cannot pickle '_thread.RLock' object.
    """
    cloud, labels = training_cloud_and_labels
    manager = ClassificationManager(method="auto")
    manager.fit(cloud, labels)
    original_result = manager.classify(cloud)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.joblib")
        manager.save(path)  # must not raise

        loaded = ClassificationManager.load(path)
        loaded_result = loaded.classify(cloud)

        np.testing.assert_array_equal(original_result.labels, loaded_result.labels)


def test_loaded_classifier_cache_is_genuinely_usable(
    training_cloud_and_labels: tuple[PointCloud, np.ndarray],
) -> None:
    """
    Confirms the reloaded classifier's internal FeatureManager cache
    (with its freshly-recreated lock) actually works across repeated
    calls, not merely that loading itself didn't crash.
    """
    cloud, labels = training_cloud_and_labels
    manager = ClassificationManager(method="auto")
    manager.fit(cloud, labels)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.joblib")
        manager.save(path)
        loaded = ClassificationManager.load(path)

        first = loaded.classify(cloud)
        second = loaded.classify(cloud)
        np.testing.assert_array_equal(first.labels, second.labels)


# ----------------------------------------------------------------------
# rule_based path
# ----------------------------------------------------------------------


def test_rule_based_does_not_require_training() -> None:
    manager = ClassificationManager(method="rule_based")
    assert manager.is_trained() is True
    assert manager.supports_training() is False


def test_rule_based_classify_works_directly() -> None:
    cloud = _cloud(200)
    manager = ClassificationManager(method="rule_based")
    result = manager.classify(cloud)
    assert result.labels.shape == (200,)


def test_rule_based_fit_raises() -> None:
    cloud, labels = _cloud(10), np.zeros(10, dtype=np.int64)
    manager = ClassificationManager(method="rule_based")
    with pytest.raises(ClassificationError):
        manager.fit(cloud, labels)


def test_rule_based_save_raises() -> None:
    manager = ClassificationManager(method="rule_based")
    with pytest.raises(ClassificationError):
        manager.save("model.joblib")


# ----------------------------------------------------------------------
# Explicit ML method selection, validation.
# ----------------------------------------------------------------------


def test_explicit_random_forest_method(
    training_cloud_and_labels: tuple[PointCloud, np.ndarray],
) -> None:
    cloud, labels = training_cloud_and_labels
    manager = ClassificationManager(method="random_forest")
    manager.fit(cloud, labels)
    assert manager.method == "random_forest"


def test_rejects_unsupported_method() -> None:
    with pytest.raises(ClassificationError):
        ClassificationManager(method="bogus")


def test_available_methods_includes_rule_based_and_auto() -> None:
    manager = ClassificationManager(method="rule_based")
    methods = manager.available_methods
    assert "auto" in methods
    assert "rule_based" in methods


def test_classify_before_fit_raises_for_ml_method() -> None:
    manager = ClassificationManager(method="random_forest")
    with pytest.raises(ProcessingError, match="Classifier has not been trained"):
        manager.classify(_cloud(10))


def test_rejects_empty_cloud_for_fit() -> None:
    manager = ClassificationManager(method="random_forest")
    with pytest.raises(ClassificationError):
        manager.fit(PointCloud(), np.array([0, 1]))


def test_rejects_mismatched_label_size() -> None:
    manager = ClassificationManager(method="random_forest")
    with pytest.raises(ClassificationError):
        manager.fit(_cloud(100), np.zeros(50, dtype=np.int64))
