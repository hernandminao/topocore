"""
topocore.features.infrastructure.parking
===========================================

Parking area detection.

ROAD_SURFACE-classified points are spatially clustered and filtered
for compact horizontal footprints. This complements `RoadDetector`,
which operates on the same ASPRS classification but favors elongated
road-like clusters.

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


class ParkingDetector(ClusterDetectorBase):
    """
    Detect parking-area footprints from classified point clouds.

    Candidate points classified as ``ROAD_SURFACE`` are spatially
    clustered in XY. Clusters are filtered by maximum horizontal
    elongation so compact paved regions are favored over long,
    narrow road surfaces.

    Parameters
    ----------
    eps
        Maximum XY distance between neighboring points during
        spatial clustering.
    min_points
        Minimum number of points required for a cluster.
    max_elongation
        Maximum horizontal elongation ratio accepted for a parking
        area. Clusters exceeding this value are considered too
        road-like.

    Notes
    -----
    Detection is implemented by `ClusterDetectorBase`. This class
    only defines the semantic classification, geometry type, and
    parking-specific filtering configuration.

    `RoadDetector` and `ParkingDetector` intentionally operate on
    the same ``ROAD_SURFACE`` classification. Their geometric
    elongation filters provide the semantic separation between
    elongated road surfaces and compact parking areas.
    """

    category = FeatureCategory.INFRASTRUCTURE
    feature_type = FeatureType.PARKING
    geometry_type = GeometryType.POLYGON
    classification_codes = (PointClassification.ROAD_SURFACE,)
    version = "1.0"

    def __init__(
        self,
        eps: float = 1.5,
        min_points: int = 80,
        max_elongation: float = 2.0,
    ) -> None:
        super().__init__(
            ClusterFilterConfig(
                eps=eps,
                min_points=min_points,
                max_elongation=max_elongation,
            )
        )

    @override
    def name(self) -> str:
        return "parking"


DetectorRegistry.register(ParkingDetector)

__all__ = ["ParkingDetector"]
