"""
topocore.features.drainage.inspection_chambers
==================================================

Inspection chamber detection from classified point-cloud clusters.

Inspection chambers use the same compact-cluster detection strategy
as manholes, but with larger default footprint and clustering
thresholds to represent their typically greater physical extent.

As with `ManholeDetector`, ASPRS LAS classification codes do not
define a dedicated inspection-chamber class. The classification
bucket is therefore configurable per detector instance.

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
from topocore.features.drainage.manholes import ManholeDetector
from topocore.features.models import FeatureType
from topocore.pointcloud.classification import PointClassification


class InspectionChamberDetector(ManholeDetector):
    """
    Detect inspection chambers as relatively large, compact clusters.

    This detector specializes `ManholeDetector`, preserving its
    configurable classification-code handling and POINT geometry
    generation while using size thresholds more appropriate for
    inspection chambers.

    Parameters
    ----------
    classification_code
        Classification bucket searched for candidate inspection
        chamber points. Defaults to
        `PointClassification.UNCLASSIFIED`.
    eps
        Maximum horizontal distance between points for clustering.
    min_points
        Minimum number of points required for a candidate cluster.
    min_horizontal_extent
        Minimum expected horizontal extent in meters.
    max_horizontal_extent
        Maximum expected horizontal extent in meters.
    max_elongation
        Maximum XY elongation ratio allowed for a candidate cluster.
    """

    feature_type = FeatureType.INSPECTION_CHAMBER
    version = "1.0"

    def __init__(
        self,
        classification_code: PointClassification = PointClassification.UNCLASSIFIED,
        eps: float = 0.5,
        min_points: int = 8,
        min_horizontal_extent: float = 0.8,
        max_horizontal_extent: float = 2.5,
        max_elongation: float = 1.6,
    ) -> None:
        super().__init__(
            classification_code=classification_code,
            eps=eps,
            min_points=min_points,
            min_horizontal_extent=min_horizontal_extent,
            max_horizontal_extent=max_horizontal_extent,
            max_elongation=max_elongation,
        )

    @override
    def name(self) -> str:
        """Return the detector registry name."""
        return "inspection_chambers"


DetectorRegistry.register(InspectionChamberDetector)

__all__ = ["InspectionChamberDetector"]
