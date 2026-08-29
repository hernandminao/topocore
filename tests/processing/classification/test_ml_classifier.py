"""
Coverage audit tests for topocore.processing.classification.ml.MachineLearningClassifier.

MachineLearningClassifier is directly instantiable (not truly
abstract in practice): its `model` constructor parameter is typed
via the SklearnModel Protocol, so a plain fake object implementing
fit()/predict()/predict_proba() exercises this class's own logic
directly, without needing a real sklearn dependency or going through
a concrete subclass (RandomForestClassifier, etc.).

Confirmed via direct execution before writing this suite:
`_build_feature_matrix()` allocates its output via
`np.empty((n_points, n_features))` with n_points/n_features taken
directly from cloud.point_count/len(self._feature_names) -- meaning
its own shape ALWAYS matches those two values by construction. This
makes the following three defensive checks unreachable, and they are
NOT tested here (per explicit review -- forcing them would require
breaking np.empty()'s own construction guarantee or the feature
registry's exact 1:1 coverage of _SUPPORTED_FEATURES, an internal
algorithm invariant, not a legitimate external input):
  - fit(): "Feature matrix size does not match labels."
  - classify(): "Feature dimension mismatch."
  - _build_feature_matrix(): "Feature 'name' was not computed."
    (confirmed the 11 names in _SUPPORTED_FEATURES are covered
    exactly by _register_feature_computers (8) +
    _add_radiometric_features (3), with no gap.)

_validate_feature()'s 4 checks ARE tested here, via a deliberately
malformed custom FeatureComputer registered through FeatureManager --
this exercises the real contract boundary between an external
feature computer and this classifier, not an artificial internal
state.

joblib-absence (save()/load()) is tested via
`patch.dict(sys.modules, {"joblib": None})` -- confirmed directly
this is a clean, standard technique (forcing the next `import
joblib` to raise ImportError) rather than a fragile, intrusive patch
of the import machinery itself.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.ml import MachineLearningClassifier
from topocore.processing.exceptions import ClassificationError, ProcessingError
from topocore.processing.features.base import ScalarFeatureComputer


class _FakeModel:
    """A minimal SklearnModel-protocol-conforming fake, for direct MachineLearningClassifier testing."""

    def __init__(
        self,
        *,
        fit_raises: bool = False,
        predict_raises: bool = False,
        proba_raises: bool = False,
    ) -> None:
        self._fit_raises = fit_raises
        self._predict_raises = predict_raises
        self._proba_raises = proba_raises

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        if self._fit_raises:
            raise RuntimeError("simulated backend training failure")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._predict_raises:
            raise RuntimeError("simulated backend prediction failure")
        return np.zeros(X.shape[0], dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._proba_raises:
            raise RuntimeError("simulated backend probability failure")
        return np.column_stack([np.full(X.shape[0], 0.3), np.full(X.shape[0], 0.7)])


class _BrokenFeatureComputer(ScalarFeatureComputer):
    """Deliberately returns a malformed feature array -- exercises _validate_feature()'s own contract boundary."""

    def __init__(self, mode: str) -> None:
        self._mode = mode

    def compute(self, cloud: PointCloud) -> np.ndarray:
        n = cloud.point_count
        if self._mode == "wrong_ndim":
            return np.zeros((n, 1))
        if self._mode == "wrong_size":
            return np.zeros(n - 1)
        if self._mode == "wrong_dtype":
            return np.array(["a"] * n)
        if self._mode == "non_finite":
            return np.full(n, np.nan)
        raise ValueError(f"unknown mode {self._mode}")

    def name(self) -> str:
        return "height"  # reuses an already-supported name so _validate_features() accepts it

    def requires_neighbors(self) -> bool:
        return False

    def default_k(self) -> int | None:
        return None

    def default_radius(self) -> float | None:
        return None


def _training_cloud(n: int = 50, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 100, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 100, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 10, n)
    cloud.add_chunk(chunk)
    return cloud


def _binary_labels(n: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n).astype(np.int64)


# ----------------------------------------------------------------------
# A. Constructor.
# ----------------------------------------------------------------------


def test_k_less_than_three_rejected() -> None:
    with pytest.raises(ValueError, match="k must be at least 3"):
        MachineLearningClassifier(_FakeModel(), k=2)


def test_radius_not_positive_rejected() -> None:
    with pytest.raises(ValueError, match="radius must be greater than zero"):
        MachineLearningClassifier(_FakeModel(), radius=0.0)


def test_valid_construction() -> None:
    clf = MachineLearningClassifier(_FakeModel(), k=5, radius=1.0)
    assert clf.is_trained() is False


# ----------------------------------------------------------------------
# B. Feature name validation.
# ----------------------------------------------------------------------


def test_empty_feature_names_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        MachineLearningClassifier(_FakeModel(), feature_names=[])


def test_duplicate_feature_names_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate feature names"):
        MachineLearningClassifier(_FakeModel(), feature_names=["height", "height"])


def test_unknown_feature_name_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown features"):
        MachineLearningClassifier(_FakeModel(), feature_names=["not_a_real_feature"])


def test_valid_feature_names_accepted() -> None:
    clf = MachineLearningClassifier(_FakeModel(), feature_names=["height", "density"])
    assert clf._feature_names == ["height", "density"]


# ----------------------------------------------------------------------
# C. _validate_feature() -- via a deliberately broken custom computer.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected_message"),
    [
        ("wrong_ndim", "must be one-dimensional"),
        ("wrong_size", "invalid size"),
        ("wrong_dtype", "must be numeric"),
        ("non_finite", "NaN or infinite"),
    ],
)
def test_validate_feature_rejects_malformed_computer_output(mode: str, expected_message: str) -> None:
    cloud = _training_cloud()
    clf = MachineLearningClassifier(_FakeModel(), feature_names=["height"])
    clf._feature_manager._computers["height"] = _BrokenFeatureComputer(mode)

    with pytest.raises(ProcessingError, match=expected_message):
        clf._build_feature_matrix(cloud)


# ----------------------------------------------------------------------
# D. fit().
# ----------------------------------------------------------------------


def test_fit_rejects_empty_cloud() -> None:
    clf = MachineLearningClassifier(_FakeModel())
    with pytest.raises(ProcessingError, match="empty point cloud"):
        clf.fit(PointCloud(), np.array([], dtype=np.int64))


def test_fit_rejects_two_dimensional_labels() -> None:
    cloud = _training_cloud()
    clf = MachineLearningClassifier(_FakeModel())
    with pytest.raises(ProcessingError, match="one-dimensional"):
        clf.fit(cloud, np.zeros((50, 1), dtype=np.int64))


def test_fit_rejects_non_integer_labels() -> None:
    cloud = _training_cloud()
    clf = MachineLearningClassifier(_FakeModel())
    with pytest.raises(ProcessingError, match="integer values"):
        clf.fit(cloud, np.zeros(50, dtype=np.float64))


def test_fit_rejects_label_count_mismatch() -> None:
    cloud = _training_cloud(n=50)
    clf = MachineLearningClassifier(_FakeModel())
    with pytest.raises(ProcessingError, match="Expected 50 labels"):
        clf.fit(cloud, np.zeros(10, dtype=np.int64))


def test_fit_rejects_negative_labels() -> None:
    cloud = _training_cloud(n=50)
    labels = _binary_labels(n=50)
    labels[0] = -1
    clf = MachineLearningClassifier(_FakeModel())
    with pytest.raises(ProcessingError, match="non-negative"):
        clf.fit(cloud, labels)


def test_fit_rejects_fewer_than_two_classes() -> None:
    cloud = _training_cloud(n=50)
    clf = MachineLearningClassifier(_FakeModel())
    with pytest.raises(ProcessingError, match="at least two classes"):
        clf.fit(cloud, np.zeros(50, dtype=np.int64))


def test_fit_happy_path_marks_trained() -> None:
    cloud = _training_cloud(n=50)
    labels = _binary_labels(n=50)
    clf = MachineLearningClassifier(_FakeModel())

    result = clf.fit(cloud, labels)

    assert clf.is_trained() is True
    assert result is clf


def test_fit_wraps_model_exception() -> None:
    cloud = _training_cloud(n=50)
    labels = _binary_labels(n=50)
    clf = MachineLearningClassifier(_FakeModel(fit_raises=True))

    with pytest.raises(ProcessingError, match="Model training failed") as exc_info:
        clf.fit(cloud, labels)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


# ----------------------------------------------------------------------
# E. classify().
# ----------------------------------------------------------------------


def test_classify_rejects_untrained_classifier() -> None:
    cloud = _training_cloud()
    clf = MachineLearningClassifier(_FakeModel())
    with pytest.raises(ProcessingError, match="not been trained"):
        clf.classify(cloud)


def test_classify_rejects_empty_cloud() -> None:
    clf = MachineLearningClassifier(_FakeModel())
    clf.fit(_training_cloud(n=50), _binary_labels(n=50))
    with pytest.raises(ProcessingError, match="empty point cloud"):
        clf.classify(PointCloud())


def test_classify_happy_path_returns_labels_and_confidence() -> None:
    cloud = _training_cloud(n=50)
    labels = _binary_labels(n=50)
    clf = MachineLearningClassifier(_FakeModel())
    clf.fit(cloud, labels)

    result = clf.classify(cloud)

    assert len(result.labels) == 50
    assert result.has_confidence is True


def test_classify_wraps_predict_exception() -> None:
    cloud = _training_cloud(n=50)
    labels = _binary_labels(n=50)
    clf = MachineLearningClassifier(_FakeModel())
    clf.fit(cloud, labels)
    clf._model = _FakeModel(predict_raises=True)

    with pytest.raises(ProcessingError, match="Prediction failed") as exc_info:
        clf.classify(cloud)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_classify_falls_back_to_no_confidence_when_predict_proba_fails() -> None:
    """predict_proba() failing must not fail classify() overall -- confidence just becomes None."""
    cloud = _training_cloud(n=50)
    labels = _binary_labels(n=50)
    clf = MachineLearningClassifier(_FakeModel())
    clf.fit(cloud, labels)
    clf._model = _FakeModel(proba_raises=True)

    result = clf.classify(cloud)

    assert result.has_confidence is False


# ----------------------------------------------------------------------
# F. Radiometric features.
# ----------------------------------------------------------------------


def test_radiometric_feature_extracted_correctly() -> None:
    cloud = _training_cloud(n=50)
    chunk = next(iter(cloud))
    intensity_values = np.arange(50, dtype=np.float64)
    # Add INTENSITY by rebuilding the chunk with the extra attribute.
    new_chunk = Chunk(
        size=50,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
            PointAttribute.INTENSITY,
        ],
    )
    new_chunk[PointAttribute.X][:] = chunk[PointAttribute.X]
    new_chunk[PointAttribute.Y][:] = chunk[PointAttribute.Y]
    new_chunk[PointAttribute.Z][:] = chunk[PointAttribute.Z]
    new_chunk[PointAttribute.INTENSITY][:] = intensity_values
    cloud_with_intensity = PointCloud()
    cloud_with_intensity.add_chunk(new_chunk)

    clf = MachineLearningClassifier(_FakeModel(), feature_names=["intensity"])
    matrix = clf._build_feature_matrix(cloud_with_intensity)

    np.testing.assert_array_equal(matrix[:, 0], intensity_values)


def test_missing_radiometric_attribute_rejected() -> None:
    cloud = _training_cloud(n=50)  # no INTENSITY attribute
    clf = MachineLearningClassifier(_FakeModel(), feature_names=["intensity"])

    with pytest.raises(ProcessingError, match="requires attribute"):
        clf._build_feature_matrix(cloud)


# ----------------------------------------------------------------------
# G. name() -- confirmed reachable via ClassificationManager's own contract.
# ----------------------------------------------------------------------


def test_name_returns_machine_learning() -> None:
    clf = MachineLearningClassifier(_FakeModel())
    assert clf.name() == "machine_learning"


# ----------------------------------------------------------------------
# H. Persistence.
# ----------------------------------------------------------------------


def test_save_rejects_untrained_classifier(tmp_path: object) -> None:
    clf = MachineLearningClassifier(_FakeModel())
    with pytest.raises(ProcessingError, match="untrained"):
        clf.save(str(tmp_path) + "/model.joblib")  # type: ignore[operator]


def test_save_and_load_round_trip(tmp_path: object) -> None:
    cloud = _training_cloud(n=50)
    labels = _binary_labels(n=50)
    clf = MachineLearningClassifier(_FakeModel(), k=7, feature_names=["height"])
    clf.fit(cloud, labels)

    path = str(tmp_path) + "/model.joblib"  # type: ignore[operator]
    clf.save(path)
    restored = MachineLearningClassifier.load(path)

    assert restored.is_trained() is True
    assert restored._k == 7


def test_load_rejects_file_not_containing_expected_class(tmp_path: object) -> None:
    import joblib  # type: ignore[import-untyped]

    path = str(tmp_path) + "/not_a_classifier.joblib"  # type: ignore[operator]
    joblib.dump({"not": "a classifier"}, path)

    with pytest.raises(ClassificationError, match="does not contain"):
        MachineLearningClassifier.load(path)


def test_save_without_joblib_raises_classification_error(tmp_path: object) -> None:
    cloud = _training_cloud(n=50)
    labels = _binary_labels(n=50)
    clf = MachineLearningClassifier(_FakeModel())
    clf.fit(cloud, labels)

    with (
        patch.dict(sys.modules, {"joblib": None}),
        pytest.raises(ClassificationError, match="joblib is not installed"),
    ):
        clf.save(str(tmp_path) + "/model.joblib")  # type: ignore[operator]


def test_load_without_joblib_raises_classification_error(tmp_path: object) -> None:
    with (
        patch.dict(sys.modules, {"joblib": None}),
        pytest.raises(ClassificationError, match="joblib is not installed"),
    ):
        MachineLearningClassifier.load(str(tmp_path) + "/model.joblib")  # type: ignore[operator]
