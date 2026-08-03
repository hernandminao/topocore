"""
topocore.processing.protocols
=============================

Protocols (structural typing) for the processing subsystem.

These protocols define the expected interfaces for processing components,
enabling flexible and decoupled implementations. They are used to enforce
contracts and facilitate dependency injection.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from topocore.pointcloud.pointcloud import PointCloud

# ============================================================================
# Transformer Protocols
# ============================================================================


class Transformer(Protocol):
    """
    A protocol for components that transform a point cloud.

    Any class that implements this protocol must provide a ``transform``
    method that takes a ``PointCloud`` and returns a new ``PointCloud``.
    The component should be callable as well, deferring to ``transform``.
    """

    def transform(self, cloud: PointCloud) -> PointCloud:
        """
        Apply the transformation to the point cloud.

        Parameters
        ----------
        cloud
            The input point cloud.

        Returns
        -------
        PointCloud
            The transformed point cloud.
        """
        ...

    def __call__(self, cloud: PointCloud) -> PointCloud: ...


# ============================================================================
# Masker Protocols
# ============================================================================


class Masker(Protocol):
    """
    A protocol for components that compute a boolean mask on a point cloud.

    The mask is typically used for filtering, segmentation, or
    classification. ``True`` values indicate points that should be kept
    or belong to a specific class.
    """

    def mask(self, cloud: PointCloud) -> NDArray[np.bool_]:
        """
        Compute a boolean mask for the point cloud.

        Parameters
        ----------
        cloud
            The input point cloud.

        Returns
        -------
        NDArray[np.bool_]
            A boolean array of the same length as the number of points.
        """
        ...

    def __call__(self, cloud: PointCloud) -> NDArray[np.bool_]: ...


# ============================================================================
# Estimator Protocols
# ============================================================================

T_co = TypeVar("T_co", covariant=True)


class Estimator(Protocol[T_co]):
    """
    A protocol for components that estimate properties of a point cloud.

    The estimation can be of any type (e.g., normals, features, ground
    elevation).
    """

    def estimate(self, cloud: PointCloud) -> T_co:
        """
        Estimate a property from the point cloud.

        Parameters
        ----------
        cloud
            The input point cloud.

        Returns
        -------
        object
            The estimated property (type depends on the estimator).
        """
        ...

    def __call__(self, cloud: PointCloud) -> T_co: ...


# ============================================================================
# Sampler Protocols
# ============================================================================


class Sampler(Protocol):
    """
    A protocol for components that sample/downsample a point cloud.

    The sample method must return a new, smaller ``PointCloud``.
    """

    def sample(self, cloud: PointCloud) -> PointCloud:
        """
        Sample the point cloud.

        Parameters
        ----------
        cloud
            The input point cloud.

        Returns
        -------
        PointCloud
            The sampled (downsampled) point cloud.
        """
        ...

    def __call__(self, cloud: PointCloud) -> PointCloud: ...


# ============================================================================
# Segmenter Protocols
# ============================================================================


class Segmenter(Protocol):
    """
    A protocol for components that segment a point cloud.

    The segment method must return a list of segments, where each segment
    is a list or array of point indices.
    """

    def segment(self, cloud: PointCloud) -> list[NDArray[np.int64]]:
        """
        Segment the point cloud into clusters.

        Parameters
        ----------
        cloud
            The input point cloud.

        Returns
        -------
        list[NDArray[np.int64]]
            A list of index arrays, one per segment.
        """
        ...

    def __call__(self, cloud: PointCloud) -> list[NDArray[np.int64]]: ...


# ============================================================================
# Classifier Protocols
# ============================================================================


class Classifier(Protocol):
    """
    A protocol for components that classify points in a point cloud.

    The classify method must return an array of integer labels
    (classification codes) for each point.
    """

    def classify(self, cloud: PointCloud) -> NDArray[np.int64]:
        """
        Classify each point in the cloud.

        Parameters
        ----------
        cloud
            The input point cloud.

        Returns
        -------
        NDArray[np.int64]
            An array of integer classification codes.
        """
        ...

    def __call__(self, cloud: PointCloud) -> NDArray[np.int64]: ...


__all__ = [
    "Transformer",
    "Masker",
    "Estimator",
    "Sampler",
    "Segmenter",
    "Classifier",
]
