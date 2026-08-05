"""
topocore.features.buildings.buildings
========================================

Building footprint detection: clusters BUILDING-classified points
and reports each cluster's 2D convex-hull footprint.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import override

from topocore.features._shared import ClusterDetectorBase, ClusterFilterConfig
from topocore.features.detector import DetectorRegistry
from topocore.features.models import FeatureCategory, FeatureType, GeometryType
from topocore.pointcloud.classification import PointClassification


class BuildingDetector(ClusterDetectorBase):
    """
    Detects building footprints from classified BUILDING points.

    Parameters
    ----------
    eps
        Max distance (meters) between points to link them into the
        same cluster.
    min_points
        Minimum points for a cluster to be considered a building.
    min_horizontal_extent / max_horizontal_extent
        Footprint diagonal bounds (meters), to reject noise-sized or
        implausibly large clusters.
    """

    category = FeatureCategory.BUILDING
    feature_type = FeatureType.BUILDING
    geometry_type = GeometryType.POLYGON
    classification_codes = (PointClassification.BUILDING,)
    version = "1.0"

    def __init__(
        self,
        eps: float = 1.0,
        min_points: int = 20,
        min_horizontal_extent: float = 2.0,
        max_horizontal_extent: float = 200.0,
    ) -> None:
        super().__init__(
            ClusterFilterConfig(
                eps=eps,
                min_points=min_points,
                min_horizontal_extent=min_horizontal_extent,
                max_horizontal_extent=max_horizontal_extent,
            )
        )

    @override
    def name(self) -> str:
        return "buildings"


DetectorRegistry.register(BuildingDetector)

__all__ = ["BuildingDetector"]
