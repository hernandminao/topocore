"""
topocore.features.utilities.signs
====================================

Sign detection from classified point-cloud clusters.

This detector specializes `PoleDetector` for the shorter vertical
extent and compact footprint typically associated with sign posts.
It reuses the same clustering, geometric filtering,
representative-point generation, metadata, and source-point
traceability provided by the generic pole detector.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import override

from topocore.features.detector import DetectorRegistry
from topocore.features.models import FeatureType
from topocore.features.utilities.poles import PoleDetector
from topocore.pointcloud.classification import PointClassification


class SignDetector(PoleDetector):
    """
    Detect signs as short, thin, isolated vertical clusters.

    This detector specializes `PoleDetector` with dimensions more
    appropriate for sign posts while preserving the same configurable
    classification bucket.

    Parameters
    ----------
    classification_code
        Classification bucket searched for candidate sign points.
        Defaults to `PointClassification.UNCLASSIFIED`.
    eps
        Maximum horizontal distance between points for clustering.
    min_points
        Minimum number of points required for a candidate cluster.
    min_height
        Minimum vertical extent, in meters, required for a cluster.
    max_height
        Maximum vertical extent, in meters, allowed for a cluster.
    max_horizontal_extent
        Maximum horizontal footprint extent, in meters, allowed for
        a cluster.
    """

    feature_type = FeatureType.SIGN
    version = "1.0"

    def __init__(
        self,
        classification_code: PointClassification = PointClassification.UNCLASSIFIED,
        eps: float = 0.25,
        min_points: int = 5,
        min_height: float = 0.8,
        max_height: float = 3.0,
        max_horizontal_extent: float = 0.8,
    ) -> None:
        super().__init__(
            classification_code=classification_code,
            eps=eps,
            min_points=min_points,
            min_height=min_height,
            max_height=max_height,
            max_horizontal_extent=max_horizontal_extent,
        )

    @override
    def name(self) -> str:
        """Return the detector registry name."""
        return "signs"


DetectorRegistry.register(SignDetector)

__all__ = ["SignDetector"]
