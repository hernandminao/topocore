"""
topocore.analysis.exceptions
============================

Exceptions raised by the analysis subsystem.

This module defines a clear exception hierarchy for the analysis
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

from topocore.processing.exceptions import ProcessingError


class AnalysisError(ProcessingError):
    """
    Base exception for all analysis-related errors.

    All exceptions raised by the ``analysis`` module inherit from
    this class.
    """


class DistanceError(AnalysisError):
    """
    Raised when a distance calculation fails.

    This can occur when input coordinates are invalid, when the
    geodesic calculator cannot be initialized, or when the selected
    method does not support the given input dimensions.
    """


class VolumeError(AnalysisError):
    """
    Raised when a volume calculation fails.

    This can occur when the input surfaces are incompatible, when
    the bounding boxes do not overlap, or when the computation
    encounters numerical issues.
    """


class ProfileError(AnalysisError):
    """
    Raised when a profile generation fails.

    This can occur when the axis points are collinear, when the
    TIN does not cover the profile extent, or when the interpolation
    fails along the profile line.
    """


class VisibilityError(AnalysisError):
    """
    Raised when a visibility analysis fails.

    This can occur when the observer or target point is outside the
    terrain surface, when the TIN cannot be traversed for the LOS
    computation, or when the viewshed grid exceeds memory limits.
    """


class StatisticsError(AnalysisError):
    """
    Raised when a statistical computation fails.

    This can occur when the input data is empty, when the requested
    statistic is not applicable to the input type, or when
    numerical overflow / underflow occurs.
    """


class QualityError(AnalysisError):
    """
    Raised when a quality assessment fails.

    This can occur when control points do not match measured points,
    when the cloud-to-cloud matching fails, or when the quality
    metric parameters are invalid.
    """


__all__ = [
    "AnalysisError",
    "DistanceError",
    "VolumeError",
    "ProfileError",
    "VisibilityError",
    "StatisticsError",
    "QualityError",
]
