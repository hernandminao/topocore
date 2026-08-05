"""
topocore.features.utilities.poles
====================================

Pole detection from classified point-cloud clusters.

Poles are modeled as thin, vertically extended clusters and reported
as representative XYZ points.

ASPRS LAS classification codes do not define a dedicated generic
pole class. The classification bucket searched by this detector is
therefore configurable per instance so it can match the classification
scheme used by the processing pipeline.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import override

from topocore.features._shared import (
    ClusterDetectorBase,
    ClusterFilterConfig,
)
from topocore.features.detector import DetectorRegistry
from topocore.features.models import (
    FeatureCategory,
    FeatureType,
    GeometryType,
)
from topocore.pointcloud.classification import PointClassification


class PoleDetector(ClusterDetectorBase):
    """
    Detect poles as thin, vertically extended point-cloud clusters.

    Candidate points are clustered horizontally and filtered by
    vertical and horizontal extent. Accepted clusters are represented
    by a single XYZ point using the lowest-Z cluster point as an
    approximate pole-base position.

    Parameters
    ----------
    classification_code
        Classification bucket searched for candidate pole points.
        Defaults to `PointClassification.UNCLASSIFIED`.
    eps
        Maximum horizontal distance between points for clustering.
    min_points
        Minimum number of points required for a pole candidate.
    min_height
        Minimum vertical extent, in meters, required for a cluster.
    max_height
        Maximum vertical extent, in meters, allowed for a cluster.
    max_horizontal_extent
        Maximum horizontal footprint extent, in meters, allowed for
        a cluster.
    """

    category = FeatureCategory.UTILITY
    feature_type = FeatureType.POLE
    geometry_type = GeometryType.POINT

    point_strategy = "min_z"
    version = "1.0"

    __slots__ = ("_classification_code",)

    def __init__(
        self,
        classification_code: PointClassification = PointClassification.UNCLASSIFIED,
        eps: float = 0.3,
        min_points: int = 10,
        min_height: float = 2.0,
        max_height: float = 12.0,
        max_horizontal_extent: float = 0.6,
    ) -> None:
        self._classification_code = classification_code

        super().__init__(
            ClusterFilterConfig(
                eps=eps,
                min_points=min_points,
                min_height=min_height,
                max_height=max_height,
                max_horizontal_extent=max_horizontal_extent,
            )
        )

    @override
    def name(self) -> str:
        """Return the detector registry name."""
        return "poles"

    @override
    def _classification_codes(
        self,
    ) -> tuple[PointClassification, ...]:
        """
        Return the classification bucket configured for this instance.

        Generic poles do not have a dedicated ASPRS LAS
        classification code, so the candidate bucket is selected
        when the detector instance is created.
        """
        return (self._classification_code,)


DetectorRegistry.register(PoleDetector)

__all__ = ["PoleDetector"]
