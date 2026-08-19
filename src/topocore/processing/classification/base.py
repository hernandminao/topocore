"""
topocore.processing.classification.base
=======================================

Abstract base classes for point cloud classification.

This module defines the interfaces for:
- Classifier: General classification interface
- RuleBasedClassifier: Rule-based classification interface
- MLClassifier: Machine learning classifier interface

Classification is the process of assigning semantic labels to
each point in a point cloud (ground, vegetation, building, etc.).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing._shared import build_cloud_from_mask, flatten_attributes
from topocore.processing.types import IntArray1D


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """
    Result of a classification operation.

    Attributes
    ----------
    labels
        Integer classification codes for each point (ASPRS codes).
    confidence
        Confidence scores for each classification (optional).
    cloud
        The classified point cloud (with labels added as an attribute).
    """

    labels: IntArray1D
    cloud: PointCloud
    confidence: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """
        Validate the classification result.
        """
        if self.labels.ndim != 1:
            raise ValueError("Labels must be a one-dimensional array.")

        if self.labels.shape[0] != self.cloud.point_count:
            raise ValueError(
                f"Label count ({self.labels.shape[0]}) does not match cloud point count ({self.cloud.point_count})."
            )

        if not np.issubdtype(self.labels.dtype, np.integer):
            raise TypeError(f"Labels must be integer-typed, got {self.labels.dtype}.")

        if self.confidence is not None:
            if self.confidence.ndim != 1:
                raise ValueError("Confidence must be a one-dimensional array.")

            if self.confidence.shape[0] != self.labels.shape[0]:
                raise ValueError(
                    f"Confidence count ({self.confidence.shape[0]}) does not match "
                    f"label count ({self.labels.shape[0]})."
                )

            if not np.issubdtype(self.confidence.dtype, np.floating):
                raise TypeError(f"Confidence must be float-typed, got {self.confidence.dtype}.")

    @property
    def has_confidence(self) -> bool:
        """Return whether confidence scores are available."""
        return self.confidence is not None

    def get_class_mask(
        self,
        class_code: int,
    ) -> NDArray[np.bool_]:
        """
        Get a boolean mask for a specific class.

        Parameters
        ----------
        class_code
            The ASPRS classification code.

        Returns
        -------
        NDArray[np.bool_]
            Boolean mask where True = points of the specified class.
        """
        return np.equal(self.labels, class_code)

    def extract_class(
        self,
        class_code: int,
    ) -> PointCloud:
        """
        Extract points of a specific class.

        Parameters
        ----------
        class_code
            The ASPRS classification code.

        Returns
        -------
        PointCloud
            A point cloud containing only points of the specified class.
        """
        mask = self.get_class_mask(class_code)
        flattened = flatten_attributes(self.cloud)

        return build_cloud_from_mask(flattened, mask)

    def extract_classes(
        self,
        class_codes: list[int],
    ) -> PointCloud:
        """
        Extract points of multiple classes.

        Parameters
        ----------
        class_codes
            List of ASPRS classification codes.

        Returns
        -------
        PointCloud
            A point cloud containing only points of the specified classes.
        """
        mask = np.isin(self.labels, class_codes)
        flattened = flatten_attributes(self.cloud)

        return build_cloud_from_mask(flattened, mask)


class Classifier(ABC):
    """
    Abstract interface for point cloud classifiers.

    Classifiers assign semantic labels to each point in a point cloud.

    Examples
    --------
    >>> classifier = RandomForestClassifier()
    >>> result = classifier.classify(cloud)
    >>> ground_mask = result.get_class_mask(2)  # ASPRS Ground = 2
    >>> ground_cloud = result.extract_class(2)
    """

    @abstractmethod
    def classify(
        self,
        cloud: PointCloud,
    ) -> ClassificationResult:
        """
        Classify the point cloud.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        ClassificationResult
            Result containing labels and confidence scores.

        Raises
        ------
        ProcessingError
            If the point cloud is empty or classification fails.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the classifier name."""
        ...

    @abstractmethod
    def supports_confidence(self) -> bool:
        """Return whether this classifier provides confidence scores."""
        ...

    def __call__(
        self,
        cloud: PointCloud,
    ) -> ClassificationResult:
        """
        Callable interface for classification.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        ClassificationResult
            Result containing labels and confidence scores.
        """
        return self.classify(cloud)


class RuleBasedClassifier(Classifier):
    """
    Abstract base class for rule-based classifiers.

    Rule-based classifiers use hand-crafted rules to assign labels
    based on geometric and radiometric properties.
    """

    @abstractmethod
    def rules(self) -> dict[str, str]:
        """
        Return the rules used by the classifier.

        Returns
        -------
        dict
            Dictionary mapping rule names to rule descriptions.
        """
        ...


class MLClassifier(Classifier):
    """
    Abstract base class for machine learning classifiers.

    ML classifiers learn to assign labels from labeled training data.
    """

    @abstractmethod
    def fit(
        self,
        cloud: PointCloud,
        labels: IntArray1D,
    ) -> MLClassifier:
        """
        Train the classifier on labeled data.

        Parameters
        ----------
        cloud
            Training point cloud.
        labels
            Training labels for each point.

        Returns
        -------
        MLClassifier
            Self, for method chaining.
        """
        ...

    @abstractmethod
    def is_trained(self) -> bool:
        """Return whether the classifier has been trained."""
        ...


__all__ = [
    "ClassificationResult",
    "Classifier",
    "MLClassifier",
    "RuleBasedClassifier",
]
