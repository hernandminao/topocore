"""
topocore.processing.exceptions
==============================

Exceptions raised by the point cloud processing subsystem.

This module defines a clear exception hierarchy for the processing
module, ensuring that all error conditions are properly categorized
and can be handled gracefully by higher-level code.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.core.exceptions import TopoCoreError


class ProcessingError(TopoCoreError):
    """
    Base exception for the processing subsystem.

    All exceptions raised by the ``processing`` module inherit from
    this class.
    """


class NeighborError(ProcessingError):
    """
    Raised when a neighbor search operation fails.

    This can occur when the KD-tree index cannot be built, or when a
    query specifies invalid parameters (e.g., k > number of points).
    """


class PointDescriptorError(ProcessingError):
    """
    Raised when a point descriptor computation fails.

    This can occur when an attribute is missing from the point cloud,
    or when the computation encounters numerical issues (e.g., division
    by zero).
    """


class NormalError(ProcessingError):
    """
    Raised when normal estimation fails.

    This can occur when a neighborhood is degenerate (all points
    collinear/coincident), or when the fitting algorithm diverges.
    """


class GroundError(ProcessingError):
    """
    Raised when ground classification or extraction fails.

    This can occur when the point cloud lacks sufficient points for
    ground estimation, or when the chosen algorithm fails to converge.
    """


class FilterError(ProcessingError):
    """
    Raised when a filtering operation fails.

    This can occur when the filter parameters are invalid or when the
    point cloud is empty.
    """


class SamplingError(ProcessingError):
    """
    Raised when a sampling operation fails.

    This can occur when an invalid method or parameter combination is
    specified.
    """


class RegistrationError(ProcessingError):
    """
    Raised when a registration (ICP) operation fails.

    This can occur when the algorithm fails to converge, or when the
    source and target point clouds have insufficient overlap.
    """


class SegmentationError(ProcessingError):
    """
    Raised when a segmentation operation fails.

    This can occur when the algorithm parameters are invalid, or when
    the point cloud structure is unsuitable for the chosen algorithm.
    """


class ClassificationError(ProcessingError):
    """
    Raised when a classification operation fails.

    This can occur when an unsupported method is requested, when an
    optional dependency (LightGBM, XGBoost) is not installed, or when
    a classifier is used before being trained.
    """


__all__ = [
    "ClassificationError",
    "FilterError",
    "GroundError",
    "NeighborError",
    "NormalError",
    "PointDescriptorError",
    "ProcessingError",
    "RegistrationError",
    "SamplingError",
    "SegmentationError",
]
