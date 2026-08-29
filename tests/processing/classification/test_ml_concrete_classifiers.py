"""
Coverage audit tests for the 4 concrete MachineLearningClassifier
subclasses: RandomForestClassifier, GradientBoostClassifier,
XGBoostClassifier, LightGBMClassifier.

Environment-adaptive by design: scikit-learn is confirmed installed
in every environment this suite has run in so far (random_forest,
gradient_boost genuinely available). xgboost/lightgbm's availability
varies BETWEEN environments -- this session's own audit sandbox had
them genuinely uninstalled, but a real CI/dev environment may have
them installed. An earlier version of this suite hardcoded the
sandbox's own state as a fixed assumption, which failed as soon as
it ran somewhere those packages ARE installed -- exactly the kind of
environment-assumption mistake this whole audit exists to catch.
Fixed to query `is_available()` dynamically and exercise whichever
path (installed or not) is actually true, with no mocking either
way.

ML-VALIDATION-001 (FIXED in this PR): all 4 classes consistently used
ClassificationError for their "backend not installed" check, but
plain ValueError (a completely unrelated exception type, sharing no
common ancestor with ClassificationError/ProcessingError/
TopoCoreError) for every parameter-range validation (n_estimators,
max_depth, learning_rate, etc.) -- confirmed identically across all
4 files before this fix, not an isolated mistake in one. This broke
ClassificationManager's own established exception-normalization
contract (confirmed elsewhere in this session): a caller catching
`except ClassificationError` around
`ClassificationManager(method=X, model_params={...})` would NOT have
caught this ValueError.

Fix: every constructor-level parameter validation in all 4 classes
now raises ClassificationError, matching the convention already used
for the "backend not installed" check in the same classes.
Deliberately NOT changed: internal errors raised by the underlying
sklearn/xgboost/lightgbm model itself during fit() still surface as
ProcessingError (via MachineLearningClassifier's own existing
exception-wrapping, confirmed unaffected by this fix) -- this
preserves the intended separation between "TopoCore's own contract
violation" and "the underlying model itself failed".

Confirmed directly, before and after this fix: the happy path
(fit()/classify()/reproducibility) for both available backends is
completely unaffected by this change.

Minor observation, not registered as its own finding: RandomForestClassifier's
own max_features validation only checks int/float ranges, not string
content -- an invalid string (e.g. "bogus_string") is not caught at
construction time, only later inside fit(), where
MachineLearningClassifier's own broad exception handling correctly
wraps it as ProcessingError (not a silent failure, just delayed) --
per this session's own explicit decision, this delayed-validation
case is NOT part of ML-VALIDATION-001's scope, since the error still
lands inside the TopoCore exception hierarchy.

Reproducibility (same random_state -> identical classify() output)
confirmed directly for both available backends.

is_available()'s real consumer is ClassificationManager's own
registry-building logic (confirmed via grep); name() has no
consumers beyond that same manager's own model_name property.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.gradient_boost import GradientBoostClassifier
from topocore.processing.classification.lightgbm import LightGBMClassifier
from topocore.processing.classification.manager import ClassificationManager
from topocore.processing.classification.random_forest import RandomForestClassifier
from topocore.processing.classification.xgboost import XGBoostClassifier
from topocore.processing.exceptions import ClassificationError


def _training_cloud(n: int = 50, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(
        size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z]
    )
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 5, n)
    cloud.add_chunk(chunk)
    return cloud


def _training_labels(n: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n).astype(np.int64)


# ----------------------------------------------------------------------
# Environment reality check -- queries actual availability dynamically
# rather than assuming a fixed environment. This session's own sandbox
# had xgboost/lightgbm genuinely uninstalled; a real CI/dev environment
# may have them installed -- these tests must pass in either case.
# ----------------------------------------------------------------------


def test_sklearn_backed_classifiers_are_available_in_this_environment() -> None:
    assert RandomForestClassifier.is_available() is True
    assert GradientBoostClassifier.is_available() is True


def test_xgboost_availability_flag_is_a_real_boolean() -> None:
    assert isinstance(XGBoostClassifier.is_available(), bool)


def test_lightgbm_availability_flag_is_a_real_boolean() -> None:
    assert isinstance(LightGBMClassifier.is_available(), bool)


# ----------------------------------------------------------------------
# "Not installed" OR "installed" -- whichever is actually true in this
# environment is exercised for real, with no mocking either way.
# ----------------------------------------------------------------------


def test_xgboost_matches_its_own_availability_flag() -> None:
    if XGBoostClassifier.is_available():
        # Installed here: construction succeeds; its own parameter
        # validation (ML-VALIDATION-001) is exercised for real.
        XGBoostClassifier()
        with pytest.raises(ClassificationError, match="n_estimators"):
            XGBoostClassifier(n_estimators=0)
    else:
        with pytest.raises(ClassificationError, match="XGBoost is not installed"):
            XGBoostClassifier()


def test_lightgbm_matches_its_own_availability_flag() -> None:
    if LightGBMClassifier.is_available():
        LightGBMClassifier()
        with pytest.raises(ClassificationError, match="n_estimators"):
            LightGBMClassifier(n_estimators=0)
    else:
        with pytest.raises(ClassificationError, match="LightGBM is not installed"):
            LightGBMClassifier()


# ----------------------------------------------------------------------
# RandomForestClassifier -- constructor validation.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_estimators": 0}, "n_estimators"),
        ({"min_samples_split": 1}, "min_samples_split"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"max_depth": 0}, "max_depth"),
        ({"n_jobs": 0}, "n_jobs"),
        ({"max_features": 0}, "max_features"),
        ({"max_features": 1.5}, "max_features"),
    ],
)
def test_random_forest_constructor_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ClassificationError, match=match):
        RandomForestClassifier(**kwargs)


def test_random_forest_fit_and_classify() -> None:
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    cloud = _training_cloud()
    clf.fit(cloud, _training_labels())

    assert clf.is_trained() is True
    result = clf.classify(cloud)
    assert len(result.labels) == 50
    assert clf.name() == "random_forest"


def test_random_forest_reproducible_with_same_random_state() -> None:
    cloud = _training_cloud()
    labels = _training_labels()

    clf_a = RandomForestClassifier(n_estimators=20, random_state=42)
    clf_a.fit(cloud, labels)
    clf_b = RandomForestClassifier(n_estimators=20, random_state=42)
    clf_b.fit(cloud, labels)

    np.testing.assert_array_equal(
        clf_a.classify(cloud).labels, clf_b.classify(cloud).labels
    )


# ----------------------------------------------------------------------
# GradientBoostClassifier -- constructor validation.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_iter": 0}, "max_iter"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"l2_regularization": -1}, "l2_regularization"),
        ({"max_leaf_nodes": 1}, "max_leaf_nodes"),
        ({"max_depth": 0}, "max_depth"),
    ],
)
def test_gradient_boost_constructor_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ClassificationError, match=match):
        GradientBoostClassifier(**kwargs)


def test_gradient_boost_fit_and_classify() -> None:
    clf = GradientBoostClassifier(max_iter=20, random_state=42)
    cloud = _training_cloud()
    clf.fit(cloud, _training_labels())

    result = clf.classify(cloud)
    assert len(result.labels) == 50
    assert clf.name() == "gradient_boost"


# ----------------------------------------------------------------------
# XGBoostClassifier / LightGBMClassifier -- constructor validation
# still fires even when not installed? No -- confirmed the
# availability check comes FIRST, so these can only be tested for
# real once the package is installed. Given it is not installed here,
# their own numeric-range validation is untestable in THIS
# environment and is not force-tested via monkeypatching the
# availability flag (that would not exercise the real constructor
# path meaningfully). Documented, not skipped silently.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# ML-VALIDATION-001 regression.
# ----------------------------------------------------------------------


def test_ml_validation_001_classification_error_no_longer_escapes_as_value_error() -> (
    None
):
    """
    The core regression: ClassificationManager(method="random_forest",
    model_params={"n_estimators": -1}) must now raise
    ClassificationError, not a raw ValueError that would slip past a
    caller's `except ClassificationError`.
    """
    with pytest.raises(ClassificationError, match="n_estimators"):
        ClassificationManager(method="random_forest", model_params={"n_estimators": -1})


@pytest.mark.parametrize(
    ("classifier_class", "kwargs"),
    [
        (RandomForestClassifier, {"n_estimators": -1}),
        (RandomForestClassifier, {"min_samples_split": 1}),
        (RandomForestClassifier, {"min_samples_leaf": 0}),
        (RandomForestClassifier, {"max_depth": 0}),
        (RandomForestClassifier, {"n_jobs": 0}),
        (RandomForestClassifier, {"max_features": 0}),
        (GradientBoostClassifier, {"max_iter": 0}),
        (GradientBoostClassifier, {"learning_rate": 0}),
        (GradientBoostClassifier, {"l2_regularization": -1}),
        (GradientBoostClassifier, {"max_leaf_nodes": 1}),
        (GradientBoostClassifier, {"max_depth": 0}),
    ],
)
def test_ml_validation_001_all_parameter_checks_raise_classification_error(
    classifier_class: type, kwargs: dict
) -> None:
    """Exhaustive regression: every one of the 11 parameter-range checks across both available backends."""
    with pytest.raises(ClassificationError):
        classifier_class(**kwargs)
