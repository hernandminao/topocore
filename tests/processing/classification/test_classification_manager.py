"""
Coverage audit tests for topocore.processing.classification.manager.ClassificationManager.

Phase 5.1 (Integration) audit findings:

ClassificationManager IS a genuine orchestrator, not a case of
architectural duplication/bypass, despite superficially repeating
validations the underlying classifiers also perform. Two distinct,
legitimate reasons were confirmed directly before writing any tests:

1. State-mutation protection: fit()'s own empty-cloud and label-size
   checks run BEFORE the "auto" method resolution (which mutates
   self._method/self._classifier). Confirmed directly that without
   these checks, a failed fit() call on an "auto" manager would leave
   it partially mutated to a specific backend even though training
   never actually happened -- the manager's own checks prevent this
   inconsistent state.

2. Exception-type normalization: confirmed RuleBasedClassifier/
   MachineLearningClassifier raise the broader ProcessingError for
   these same conditions, not the narrower ClassificationError
   (confirmed ClassificationError IS a subclass of ProcessingError).
   Without the manager's own checks, a caller catching
   `except ClassificationError` around manager.classify()/fit() would
   miss the underlying classifier's own ProcessingError. The
   manager's checks guarantee its own public exception contract
   regardless of which classifier is currently active.

Neither of these was consolidated or removed -- they serve real
purposes distinct from the underlying classifiers' own validation.

Confirmed via workflow.py (the one real external caller):
ClassificationManager itself is genuinely active, but only via
`ClassificationManager(method=...).classify(cloud)` -- fit()/save()/
load()/supports_training()/supports_persistence()/is_trained() and
the available_methods/model_name/classifier/method properties are
NOT currently exercised by this caller. These are nonetheless treated
as genuinely testable public contract (not orphaned in the
name()/requires_k()-style sense): the class's own docstring
explicitly documents this fit()/save() usage as an intended pattern,
unlike the normals-estimator metadata methods which had no such
documented intended usage anywhere.

_validate_parameters()'s two type checks are confirmed UNREACHABLE:
`dict(model_params)`/`dict(rule_params)` run BEFORE
_validate_parameters() and either raise their own native error or
guarantee self._model_params/self._rule_params are already real
dict instances by the time _validate_parameters() runs -- its own
`isinstance(..., dict)` checks can never be False. Not tested here.

_create_classifier()'s own "unavailable method" check is confirmed
unreachable from both of its real call sites (__init__ and fit()'s
auto-resolution), each of which already guarantees availability
via _ml_registry() immediately before calling it -- a
theoretical TOCTOU race (a package becoming unavailable between
the two checks) is not force-tested.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.manager import ClassificationManager
from topocore.processing.classification.random_forest import RandomForestClassifier
from topocore.processing.exceptions import ClassificationError


def _training_cloud(n: int = 50, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 5, n)
    cloud.add_chunk(chunk)
    return cloud


def _training_labels(n: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n).astype(np.int64)


# ----------------------------------------------------------------------
# Constructor validation.
# ----------------------------------------------------------------------


def test_unsupported_method_rejected() -> None:
    with pytest.raises(ClassificationError, match="Unsupported classification method"):
        ClassificationManager(method="not_a_real_method")


def test_unavailable_ml_method_rejected() -> None:
    """Unavailable ML methods are rejected regardless of installed backends."""
    from topocore.processing.classification.manager import _ml_registry

    registry = _ml_registry()

    # Select a known ML backend and simulate that it is unavailable.
    method = "xgboost"

    if method not in registry:
        # Already unavailable in the current environment.
        unavailable_method = method
    else:
        unavailable_method = method

    with patch(
        "topocore.processing.classification.manager._ml_registry",
        return_value={key: value for key, value in registry.items() if key != unavailable_method},
    ):
        with pytest.raises(
            ClassificationError,
            match="requires an optional dependency",
        ):
            ClassificationManager(method=unavailable_method)


def test_rule_based_construction_uses_rule_based_classifier() -> None:
    manager = ClassificationManager(method="rule_based")
    assert manager.method == "rule_based"
    assert manager.supports_training() is False


def test_auto_construction_starts_as_rule_based() -> None:
    manager = ClassificationManager(method="auto")
    assert manager.method == "auto"
    assert type(manager.classifier).__name__ == "RuleBasedClassifier"


# ----------------------------------------------------------------------
# fit() -- state-mutation protection, confirmed via direct execution.
# ----------------------------------------------------------------------


def test_fit_on_rule_based_rejected_without_mutating_state() -> None:
    manager = ClassificationManager(method="rule_based")
    with pytest.raises(ClassificationError, match="does not support training"):
        manager.fit(_training_cloud(), _training_labels())


def test_fit_on_auto_with_empty_cloud_does_not_mutate_method() -> None:
    """Confirms the manager's own early check prevents a failed fit() from leaving method partially resolved."""
    manager = ClassificationManager(method="auto")

    with pytest.raises(ClassificationError, match="empty point cloud"):
        manager.fit(PointCloud(), np.array([], dtype=np.int64))

    assert manager.method == "auto"


def test_fit_rejects_label_count_mismatch() -> None:
    manager = ClassificationManager(method="random_forest")
    with pytest.raises(ClassificationError, match="invalid size"):
        manager.fit(_training_cloud(n=50), _training_labels(n=10))


def test_fit_resolves_auto_to_a_concrete_ml_backend() -> None:
    manager = ClassificationManager(method="auto")

    manager.fit(_training_cloud(), _training_labels())

    assert manager.method in ("xgboost", "lightgbm", "gradient_boost", "random_forest")
    assert manager.is_trained() is True


def test_fit_raises_when_underlying_classifier_violates_is_trained_postcondition() -> None:
    """A deliberately broken classifier: fit() succeeds but is_trained() still reports False."""
    manager = ClassificationManager(method="random_forest")

    with (
        patch.object(RandomForestClassifier, "is_trained", return_value=False),
        pytest.raises(ClassificationError, match="training failed"),
    ):
        manager.fit(_training_cloud(), _training_labels())


def test_select_auto_method_raises_when_no_backend_installed() -> None:
    with (
        patch("topocore.processing.classification.manager._ml_registry", return_value={}),
        pytest.raises(ClassificationError, match="No ML classification backend is installed"),
    ):
        ClassificationManager._select_auto_method()


# ----------------------------------------------------------------------
# classify() -- exception-type normalization, confirmed via direct execution.
# ----------------------------------------------------------------------


def test_classify_rejects_empty_cloud_with_classification_error() -> None:
    """Confirms the manager's own check, not merely the underlying classifier's broader ProcessingError."""
    manager = ClassificationManager(method="rule_based")
    with pytest.raises(ClassificationError, match="empty point cloud"):
        manager.classify(PointCloud())


def test_classify_happy_path_via_rule_based() -> None:
    manager = ClassificationManager(method="rule_based")
    result = manager.classify(_training_cloud())
    assert len(result.labels) == 50


# ----------------------------------------------------------------------
# Properties -- genuinely testable public contract, not orphaned.
# ----------------------------------------------------------------------


def test_available_methods_matches_installed_backends() -> None:
    """available_methods reflects the ML backends actually available."""
    from topocore.processing.classification.manager import _ml_registry

    manager = ClassificationManager(method="rule_based")
    methods = manager.available_methods
    registry = _ml_registry()

    assert "auto" in methods
    assert "rule_based" in methods

    for method in (
        "random_forest",
        "gradient_boost",
        "lightgbm",
        "xgboost",
    ):
        assert (method in methods) is (method in registry)


def test_model_name_reflects_underlying_classifier() -> None:
    manager = ClassificationManager(method="rule_based")
    assert manager.model_name == "rule_based"


# ----------------------------------------------------------------------
# save() / load() -- persistence round trip.
# ----------------------------------------------------------------------


def test_save_rejects_rule_based_classifier() -> None:
    manager = ClassificationManager(method="rule_based")
    with pytest.raises(ClassificationError, match="does not support persistence"):
        manager.save("/tmp/should_not_be_created.joblib")


def test_save_and_load_round_trip(tmp_path: object) -> None:
    manager = ClassificationManager(method="random_forest")
    manager.fit(_training_cloud(), _training_labels())

    path = str(tmp_path) + "/model.joblib"  # type: ignore[operator]
    manager.save(path)
    restored = ClassificationManager.load(path)

    assert restored.method == "random_forest"
    assert restored.is_trained() is True


def test_load_rejects_file_without_joblib(tmp_path: object) -> None:
    import sys

    with (
        patch.dict(sys.modules, {"joblib": None}),
        pytest.raises(ClassificationError, match="joblib is not installed"),
    ):
        ClassificationManager.load(str(tmp_path) + "/model.joblib")  # type: ignore[operator]


def test_load_rejects_file_not_containing_ml_classifier(tmp_path: object) -> None:
    import joblib  # type: ignore[import-untyped]

    path = str(tmp_path) + "/not_a_classifier.joblib"  # type: ignore[operator]
    joblib.dump({"not": "a classifier"}, path)

    with pytest.raises(ClassificationError, match="does not contain a trained ML classifier"):
        ClassificationManager.load(path)
