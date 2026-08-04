"""
topocore.processing.classification.ml
=====================================

Base classes for machine learning classifiers.

This module provides the abstract implementation shared by all
machine learning classifiers used in TopoCore.

A MachineLearningClassifier is responsible for:

- Computing geometric and radiometric features.
- Validating feature matrices.
- Managing feature ordering.
- Delegating training and prediction to a scikit-learn compatible
  estimator.

Concrete algorithms (Random Forest, SVM, XGBoost, etc.) only need
to provide the underlying estimator.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol, override

import numpy as np
from numpy.typing import NDArray

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.classification import PointClassification
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.base import (
    ClassificationResult,
    Classifier,
)
from topocore.processing.exceptions import ClassificationError, ProcessingError
from topocore.processing.features import FeatureManager


class SklearnModel(Protocol):
    """
    Protocol implemented by scikit-learn classifiers.
    """

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> object: ...

    def predict(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.int64]: ...

    def predict_proba(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...


class MachineLearningClassifier(Classifier):
    """
    Base class for machine learning classifiers.
    """

    _DEFAULT_FEATURES: ClassVar[tuple[str, ...]] = (
        "height_above_ground",
        "curvature",
        "planarity",
        "linearity",
        "sphericity",
    )

    _SUPPORTED_FEATURES = frozenset(
        {
            "height",
            "height_above_ground",
            "curvature",
            "planarity",
            "linearity",
            "sphericity",
            "verticality",
            "density",
            "intensity",
            "return_number",
            "number_of_returns",
        }
    )

    __slots__ = (
        "_model",
        "_trained",
        "_feature_manager",
        "_feature_names",
        "_k",
        "_radius",
        "_ground_class",
    )

    def __init__(
        self,
        model: SklearnModel,
        *,
        feature_names: list[str] | None = None,
        k: int = 10,
        radius: float = 1.0,
        ground_class: PointClassification = PointClassification.GROUND,
    ) -> None:
        """
        Initialize the machine learning classifier.

        Parameters
        ----------
        model
            Underlying scikit-learn compatible estimator.
        feature_names
            Features used during training and prediction.
            If None, the default feature set is used.
        k
            Number of neighbors for neighborhood-based features.
        radius
            Search radius for density computation.
        ground_class
            Classification value representing ground points.
        """
        if k < 3:
            raise ValueError("k must be at least 3.")

        if radius <= 0.0:
            raise ValueError("radius must be greater than zero.")

        self._model = model
        self._trained = False

        self._k = k
        self._radius = radius
        self._ground_class = ground_class

        self._feature_names = list(feature_names) if feature_names is not None else list(self._DEFAULT_FEATURES)
        self._validate_features()

        self._feature_manager = FeatureManager()

    def _validate_features(self) -> None:
        """Validate configured feature names."""
        if not self._feature_names:
            raise ValueError("Feature names cannot be empty.")

        if len(self._feature_names) != len(set(self._feature_names)):
            raise ValueError("Duplicate feature names are not allowed.")

        unknown = set(self._feature_names) - self._SUPPORTED_FEATURES

        if unknown:
            raise ValueError(f"Unknown features: {sorted(unknown)}")

    def fit(
        self,
        cloud: PointCloud,
        labels: NDArray[np.int64],
    ) -> MachineLearningClassifier:
        """
                Train the classifier.

                Parameters
                ----------
                cloud
                    Training point cloud.
                labels
                    Classification labels.

                Returns
        -------
                MachineLearningClassifier
                    Self for method chaining.
        """

        if cloud.is_empty:
            raise ProcessingError("Cannot train on an empty point cloud.")

        if labels.ndim != 1:
            raise ProcessingError("Labels must be one-dimensional.")

        if not np.issubdtype(labels.dtype, np.integer):
            raise ProcessingError("Labels must contain integer values.")

        if labels.shape[0] != cloud.point_count:
            raise ProcessingError(f"Expected {cloud.point_count} labels, got {labels.shape[0]}.")

        if np.unique(labels).size < 2:
            raise ProcessingError("Training requires at least two classes.")

        if np.any(labels < 0):
            raise ProcessingError("Labels must be non-negative integers.")

        features = self._build_feature_matrix(cloud)

        if features.shape[0] != labels.shape[0]:
            raise ProcessingError("Feature matrix size does not match labels.")

        try:
            self._model.fit(features, labels)

        except Exception as exc:
            raise ProcessingError(f"Model training failed: {exc}") from exc

        else:
            self._trained = True

        return self

    @override
    def classify(
        self,
        cloud: PointCloud,
    ) -> ClassificationResult:
        """
        Classify a point cloud.
        """

        if not self._trained:
            raise ProcessingError("Classifier has not been trained.")

        if cloud.is_empty:
            raise ProcessingError("Cannot classify an empty point cloud.")

        features = self._build_feature_matrix(cloud)

        if features.shape[1] != len(self._feature_names):
            raise ProcessingError("Feature dimension mismatch.")

        try:
            labels = self._model.predict(features)

        except Exception as exc:
            raise ProcessingError(f"Prediction failed: {exc}") from exc

        confidence: NDArray[np.float64] | None = None

        if self.supports_confidence():
            try:
                probabilities = self._model.predict_proba(features)
                confidence = probabilities.max(axis=1)

            except Exception:
                confidence = None

        return ClassificationResult(
            labels=labels.astype(np.int64),
            cloud=cloud,
            confidence=confidence,
        )

    def _build_feature_matrix(
        self,
        cloud: PointCloud,
    ) -> NDArray[np.float64]:
        """
        Build the feature matrix required by the ML model.
        """

        feature_map = self._feature_manager.compute_all(
            cloud,
            self._feature_names,
        )

        self._add_radiometric_features(
            cloud,
            feature_map,
        )

        n_points = cloud.point_count
        n_features = len(self._feature_names)

        # Pre-allocate feature matrix to avoid unnecessary copies (np.column_stack)
        matrix = np.empty((n_points, n_features), dtype=np.float64)

        for i, name in enumerate(self._feature_names):
            if name not in feature_map:
                raise ProcessingError(f"Feature '{name}' was not computed.")

            feature = feature_map[name]

            self._validate_feature(
                name=name,
                feature=feature,
                expected_size=n_points,
            )

            matrix[:, i] = feature
            # Release reference to feature array to help GC
            feature_map[name] = None

        # Explicitly clear the feature map
        feature_map.clear()

        return matrix

    def _add_radiometric_features(
        self,
        cloud: PointCloud,
        feature_map: dict[str, NDArray[np.float64]],
    ) -> None:
        """
        Add radiometric attributes to the feature map.
        """

        attributes = {
            "intensity": PointAttribute.INTENSITY,
            "return_number": PointAttribute.RETURN_NUMBER,
            "number_of_returns": (PointAttribute.NUMBER_OF_RETURNS),
        }

        for name, attribute in attributes.items():
            if name not in self._feature_names:
                continue

            if attribute not in cloud.attributes:
                raise ProcessingError(f"Feature '{name}' requires attribute '{attribute.name}'.")

            feature_map[name] = self._extract_attribute(
                cloud,
                attribute,
            )

    def _extract_attribute(
        self,
        cloud: PointCloud,
        attribute: PointAttribute,
    ) -> NDArray[np.float64]:
        """
        Extract a point attribute as a contiguous float64 array.
        """

        values = [chunk[attribute].astype(np.float64) for chunk in cloud]

        if not values:
            raise ProcessingError(f"Point cloud does not contain attribute '{attribute.name}'.")

        return np.concatenate(values)

    def _validate_feature(
        self,
        *,
        name: str,
        feature: NDArray[np.float64],
        expected_size: int,
    ) -> None:
        """
        Validate a feature array.
        """

        if feature.ndim != 1:
            raise ProcessingError(f"Feature '{name}' must be one-dimensional.")

        if feature.shape[0] != expected_size:
            raise ProcessingError(f"Feature '{name}' has invalid size {feature.shape[0]}, expected {expected_size}.")

        if not np.issubdtype(feature.dtype, np.number):
            raise ProcessingError(f"Feature '{name}' must be numeric.")

        if not np.isfinite(feature).all():
            raise ProcessingError(f"Feature '{name}' contains NaN or infinite values.")

    @override
    def supports_confidence(self) -> bool:
        """
        Return whether the underlying model supports
        probability estimation.
        """

        predict_proba = getattr(
            self._model,
            "predict_proba",
            None,
        )

        return callable(predict_proba)

    def is_trained(self) -> bool:
        """
        Return whether the classifier has been trained.
        """

        return self._trained

    @override
    def name(self) -> str:
        return "machine_learning"

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Persist the trained classifier to disk.

        The full classifier (model, feature configuration, and
        neighborhood/ground parameters) is serialized via ``joblib``,
        so it can be reloaded and used for prediction without
        re-specifying any configuration.

        Parameters
        ----------
        path
            Destination file path.

        Raises
        ------
        ProcessingError
            If the classifier has not been trained yet.
        ClassificationError
            If ``joblib`` is not installed.
        """
        if not self._trained:
            raise ProcessingError("Cannot save an untrained classifier.")

        try:
            import joblib  # type: ignore
        except ImportError as exc:
            raise ClassificationError(
                "joblib is not installed. Install it with `pip install topocore[ml]` "
                "(or `pip install joblib`) to save classifiers."
            ) from exc

        joblib.dump(self, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> MachineLearningClassifier:
        """
        Load a previously trained classifier from disk.

        Parameters
        ----------
        path
            Source file path.

        Returns
        -------
        MachineLearningClassifier
            The restored classifier.

        Raises
        ------
        ClassificationError
            If ``joblib`` is not installed, or the file does not
            contain an instance of ``cls``.
        """
        try:
            import joblib
        except ImportError as exc:
            raise ClassificationError(
                "joblib is not installed. Install it with `pip install topocore[ml]` "
                "(or `pip install joblib`) to load classifiers."
            ) from exc

        obj = joblib.load(Path(path))

        if not isinstance(obj, cls):
            raise ClassificationError(f"File does not contain a '{cls.__name__}': {path}")

        return obj
