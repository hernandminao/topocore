"""
topocore.features.infrastructure.roads
=========================================

Road surface detection.

ROAD_SURFACE-classified points are spatially clustered and filtered
by horizontal elongation. Road surfaces are expected to form
substantially longer-than-wide clusters, which helps distinguish
them from compact paved surfaces such as parking areas.

The common classification, XYZ extraction, clustering, geometric
filtering, footprint construction, provenance metadata, and source
point traceability are provided by `ClusterDetectorBase`.

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


class RoadDetector(ClusterDetectorBase):
    """
    Detect road-surface footprints from classified point clouds.

    Candidate points classified as ``ROAD_SURFACE`` are spatially
    clustered in XY. Clusters are then filtered by elongation so
    compact paved regions are less likely to be reported as roads.

    Parameters
    ----------
    eps
        Maximum XY distance between neighboring points during
        spatial clustering.
    min_points
        Minimum number of points required for a cluster.
    min_elongation
        Minimum horizontal elongation ratio required for a cluster
        to be considered a road surface.

    Notes
    -----
    Detection is implemented by `ClusterDetectorBase`. This class
    only defines the semantic classification, geometry type, and
    road-specific filtering configuration.
    """

    category = FeatureCategory.INFRASTRUCTURE
    feature_type = FeatureType.ROAD
    geometry_type = GeometryType.POLYGON
    classification_codes = (PointClassification.ROAD_SURFACE,)
    version = "1.0"

    def __init__(
        self,
        eps: float = 1.5,
        min_points: int = 50,
        min_elongation: float = 2.0,
    ) -> None:
        super().__init__(
            ClusterFilterConfig(
                eps=eps,
                min_points=min_points,
                min_elongation=min_elongation,
            )
        )

    @override
    def name(self) -> str:
        return "roads"


DetectorRegistry.register(RoadDetector)

__all__ = ["RoadDetector"]
