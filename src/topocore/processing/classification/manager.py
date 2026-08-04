"""
topocore.processing.classification.manager
============================================

Classification manager with automatic method selection.

This module provides a high-level manager that selects the
appropriate classification method based on user preferences and on
which optional ML dependencies (LightGBM, XGBoost) are installed. It
mirrors the design of ``topocore.processing.ground.GroundManager``.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import ClassificationError
from topocore.processing.types import IntArray1D

from .base import ClassificationResult, Classifier
from .gradient_boost import GradientBoostClassifier
from .ml import MachineLearningClassifier
from .random_forest import RandomForestClassifier
from .rules import RuleBasedClassifier

try:
    from .lightgbm import LightGBMClassifier
except ImportError:  # pragma: no cover
    LightGBMClassifier = None  # type: ignore[assignment, misc]

try:
    from .xgboost import XGBoostClassifier
except ImportError:  # pragma: no cover
    XGBoostClassifier = None  # type: ignore[assignment, misc]

_UNSUPPORTED_METHOD_ERROR = "Unsupported method: {method}. Supported: {supported}"
_NO_TRAINING_ERROR = "Method '{method}' does not support training."
_UNAVAILABLE_METHOD_ERROR = "Method '{method}' requires an optional dependency that is not installed. {hint}"

_INSTALL_HINTS: dict[str, str] = {
    "random_forest": "Install it with `pip install topocore[ml]` (or `pip install scikit-learn`).",
    "gradient_boost": "Install it with `pip install topocore[ml]` (or `pip install scikit-learn`).",
    "lightgbm": "Install it with `pip install lightgbm`.",
    "xgboost": "Install it with `pip install xgboost`.",
}

# Preference order used when method="auto" and `fit()` is called:
# prefer the fastest/most accurate backend that is actually installed.
_AUTO_PRIORITY: tuple[str, ...] = ("xgboost", "lightgbm", "gradient_boost", "random_forest")


def _build_ml_registry() -> dict[str, type[MachineLearningClassifier]]:
    """
    Build the registry of ML classifier backends that are actually
    usable in this environment (their optional dependency is
    installed).
    """
    registry: dict[str, type[MachineLearningClassifier]] = {}

    if RandomForestClassifier.is_available():
        registry["random_forest"] = RandomForestClassifier

    if GradientBoostClassifier.is_available():
        registry["gradient_boost"] = GradientBoostClassifier

    if LightGBMClassifier.is_available():
        registry["lightgbm"] = LightGBMClassifier

    if XGBoostClassifier.is_available():
        registry["xgboost"] = XGBoostClassifier

    return registry


def _ml_registry() -> dict[str, type[MachineLearningClassifier]]:
    """
    Return the registry of ML classifier backends that are actually
    usable in this environment.
    """
    return _build_ml_registry()


_ALL_ML_METHODS: tuple[str, ...] = ("random_forest", "gradient_boost", "lightgbm", "xgboost")


class ClassificationManager:
    """
    High-level manager for point cloud classification.

    This class provides a unified interface for point cloud
    classification with automatic method selection.

    Examples
    --------
    >>> manager = ClassificationManager(method="rule_based")
    >>> result = manager.classify(cloud)

    >>> manager = ClassificationManager(method="auto")
    >>> manager.fit(training_cloud, training_labels)  # picks the best installed backend
    >>> result = manager.classify(cloud)
    >>> manager.save("model.joblib")

    >>> manager = ClassificationManager(method="random_forest", model_params={"n_estimators": 200})
    >>> manager.fit(training_cloud, training_labels)
    """

    __slots__ = (
        "_method",
        "_model_params",
        "_rule_params",
        "_classifier",
    )

    def __init__(
        self,
        method: str = "auto",
        model_params: dict[str, Any] | None = None,
        rule_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the classification manager.

        Parameters
        ----------
        method
            Classification method. One of ``"rule_based"``,
            ``"auto"``, ``"random_forest"``, ``"gradient_boost"``,
            ``"lightgbm"``, or ``"xgboost"``. ``"auto"`` starts as
            rule-based (no training data required) and is promoted
            to the best available ML backend once :meth:`fit` is
            called.
        model_params
            Keyword arguments forwarded to the underlying ML
            classifier constructor (e.g. ``n_estimators``,
            ``feature_names``, ``k``).
        rule_params
            Keyword arguments forwarded to ``RuleBasedClassifier``.
        """
        supported = ("auto", "rule_based", *_ALL_ML_METHODS)

        if method not in supported:
            raise ClassificationError(
                f"Unsupported classification method '{method}'. Supported methods are: {list(supported)}"
            )

        self._method = method
        self._model_params = dict(model_params) if model_params else {}
        self._rule_params = dict(rule_params) if rule_params else {}

        self._validate_parameters()

        registry = _ml_registry()
        if method in _ALL_ML_METHODS and method not in registry:
            raise ClassificationError(
                _UNAVAILABLE_METHOD_ERROR.format(method=method, hint=_INSTALL_HINTS.get(method, ""))
            )

        self._classifier: Classifier = self._create_classifier(method)

    def _validate_parameters(self) -> None:
        """
        Validate constructor parameters.
        """
        if not isinstance(self._model_params, dict):
            raise TypeError("model_params must be a dictionary.")

        if not isinstance(self._rule_params, dict):
            raise TypeError("rule_params must be a dictionary.")

    def _create_classifier(self, method: str) -> Classifier:
        """
        Create a classifier instance for the specified method.
        """
        if method in ("rule_based", "auto"):
            return RuleBasedClassifier(**self._rule_params)

        registry = _ml_registry()
        if method not in registry:
            raise ClassificationError(
                _UNAVAILABLE_METHOD_ERROR.format(method=method, hint=_INSTALL_HINTS.get(method, ""))
            )

        return registry[method](**self._model_params)

    @property
    def method(self) -> str:
        """
        Get the current (possibly auto-resolved) method.
        """
        return self._method

    @property
    def available_methods(self) -> list[str]:
        """
        Return the classification methods usable in this environment.
        """
        methods = ["auto", "rule_based"]
        methods.extend(sorted(_ml_registry().keys()))
        return methods

    @property
    def classifier(self) -> Classifier:
        """Return the underlying classifier instance."""
        return self._classifier

    @property
    def model_name(self) -> str:
        """Return the actual name of the underlying classifier."""
        return self._classifier.name()

    def supports_training(self) -> bool:
        """Return whether the current classifier supports training."""
        return isinstance(self._classifier, MachineLearningClassifier)

    def supports_persistence(self) -> bool:
        """Return whether the current classifier can be saved/loaded."""
        # Currently, all ML classifiers support persistence via joblib
        return self.supports_training()

    def fit(
        self,
        cloud: PointCloud,
        labels: IntArray1D,
    ) -> ClassificationManager:
        """
        Train the manager on labeled data.

        Only meaningful for ML-based methods; ``RuleBasedClassifier``
        requires no training. When ``method="auto"``, the backend is
        resolved here, preferring XGBoost > LightGBM > Gradient
        Boost > Random Forest, based on what is installed.

        Parameters
        ----------
        cloud
            Training point cloud.
        labels
            Training labels for each point.

        Returns
        -------
        ClassificationManager
            Self, for method chaining.

        Raises
        ------
        ClassificationError
            If the resolved method does not support training.
        """
        if self._method == "rule_based":
            raise ClassificationError(_NO_TRAINING_ERROR.format(method=self._method))

        if cloud.is_empty:
            raise ClassificationError("Cannot train using an empty point cloud.")

        if labels.shape[0] != cloud.point_count:
            raise ClassificationError(
                f"Training labels have invalid size {labels.shape[0]}, expected {cloud.point_count}."
            )

        if self._method == "auto":
            resolved = self._select_auto_method()
            self._classifier = self._create_classifier(resolved)
            self._method = resolved

        classifier = self._classifier

        if not self.supports_training():
            raise ClassificationError(_NO_TRAINING_ERROR.format(method=self._method))

        assert isinstance(classifier, MachineLearningClassifier)
        classifier.fit(cloud, labels)

        if not classifier.is_trained():
            raise ClassificationError(f"Classifier '{self._method}' training failed.")

        return self

    @staticmethod
    def _select_auto_method() -> str:
        """
        Select the best available ML backend for ``method="auto"``.
        """
        registry = _ml_registry()
        for candidate in _AUTO_PRIORITY:
            if candidate in registry:
                return candidate

        raise ClassificationError(
            "No ML classification backend is installed. Install at least one of: "
            "scikit-learn (`pip install topocore[ml]`), lightgbm, or xgboost."
        )

    def classify(
        self,
        cloud: PointCloud,
    ) -> ClassificationResult:
        """
        Classify the point cloud using the current classifier.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        ClassificationResult
            Result containing labels and, when available, confidence
            scores.
        """
        if cloud.is_empty:
            raise ClassificationError("Cannot classify an empty point cloud.")

        return self._classifier.classify(cloud)

    def is_trained(self) -> bool:
        """
        Return whether the current classifier is ready to classify.

        Rule-based classifiers are always considered ready, since
        they require no training data.
        """
        classifier = self._classifier

        if isinstance(classifier, MachineLearningClassifier):
            return classifier.is_trained()

        return True

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Persist the trained ML classifier to disk.

        Parameters
        ----------
        path
            Destination file path.

        Raises
        ------
        ClassificationError
            If the current classifier is rule-based, or has not been
            trained yet.
        """
        if not self.supports_persistence():
            raise ClassificationError(
                f"Method '{self._method}' does not support persistence. Only trained ML classifiers can be saved."
            )

        assert isinstance(self._classifier, MachineLearningClassifier)
        self._classifier.save(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> ClassificationManager:
        """
        Load a previously trained ML classifier as a manager.

        The resulting manager's ``method`` is inferred from the
        loaded classifier itself.

        Parameters
        ----------
        path
            Source file path.

        Returns
        -------
        ClassificationManager
            A manager wrapping the restored classifier.

        Raises
        ------
        ClassificationError
            If ``joblib`` is not installed, or the file does not
            contain a trained ML classifier.
        """
        try:
            import joblib  # type: ignore
        except ImportError as exc:
            raise ClassificationError(
                "joblib is not installed. Install it with `pip install topocore[ml]` "
                "(or `pip install joblib`) to load classifiers."
            ) from exc

        classifier = joblib.load(Path(path))

        if classifier is None:
            raise ClassificationError(f"Failed to load classifier from {path}.")

        if not isinstance(classifier, MachineLearningClassifier):
            raise ClassificationError(f"File does not contain a trained ML classifier: {path}")

        manager = cls.__new__(cls)
        manager._method = classifier.name()
        manager._model_params = {}
        manager._rule_params = {}
        manager._classifier = classifier

        return manager


__all__ = [
    "ClassificationManager",
]
