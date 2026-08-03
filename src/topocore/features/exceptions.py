"""
topocore.features.exceptions
=============================

Exception hierarchy for the feature extraction module.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.core.exceptions import TopoCoreError


class FeatureError(TopoCoreError):
    """Base exception for all feature extraction errors."""


class GeometryError(FeatureError):
    """
    Raised when a feature geometry is invalid.

    This includes wrong vertex counts for the declared geometry
    type (e.g. a polygon with fewer than 3 vertices), non-finite
    coordinates, or shape mismatches.
    """


class DetectionError(FeatureError):
    """
    Raised when a detector fails to run.

    This includes missing required inputs in the
    ``DetectionContext``, or a failure internal to the detector's
    algorithm.
    """


__all__ = ["FeatureError", "GeometryError", "DetectionError"]
