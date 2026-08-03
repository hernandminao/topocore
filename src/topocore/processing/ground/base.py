"""
topocore.processing.ground.base
===============================

Abstract base classes for ground classification and extraction.

This module defines the interfaces for:
- Ground classification (assigning ground/non-ground labels)
- Ground extraction (returning a ground point cloud)
- Ground elevation estimation (for relative height computation)

Ground classification is a fundamental operation for terrain modeling,
used in DTM generation, contouring, and earthwork calculations.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.types import BoolArray1D, FloatArray1D


class GroundClassifier(ABC):
    """
    Abstract interface for ground classification.

    Ground classifiers assign a boolean label to each point:
    True = ground, False = non-ground.

    Examples
    --------
    >>> classifier = GridGroundClassifier(cell_size=1.0)
    >>> mask = classifier.classify(cloud)
    >>> ground_cloud = cloud[mask]
    """

    @abstractmethod
    def classify(
        self,
        cloud: PointCloud,
    ) -> BoolArray1D:
        """
        Classify points as ground or non-ground.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        BoolArray1D
            Boolean mask where True = ground.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the classifier name."""
        ...


class GroundExtractor(ABC):
    """
    Abstract interface for ground point extraction.

    Ground extractors return a new PointCloud containing only
    the ground points.

    Examples
    --------
    >>> extractor = GridGroundExtractor(cell_size=1.0)
    >>> ground_cloud = extractor.extract(cloud)
    """

    @abstractmethod
    def extract(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """
        Extract ground points from the point cloud.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        PointCloud
            A new point cloud containing only ground points.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the extractor name."""
        ...


class GroundElevationEstimator(ABC):
    """
    Abstract interface for ground elevation estimation.

    Ground elevation estimators compute the ground elevation at
    each point location (i.e., the terrain height).

    Examples
    --------
    >>> estimator = GridGroundElevationEstimator(cell_size=1.0)
    >>> ground_z = estimator.estimate(cloud)
    """

    @abstractmethod
    def estimate(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        """
        Estimate ground elevation for each point.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        FloatArray1D
            Array of shape (N,) containing ground elevation estimates.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the estimator name."""
        ...


__all__ = [
    "GroundClassifier",
    "GroundExtractor",
    "GroundElevationEstimator",
]
