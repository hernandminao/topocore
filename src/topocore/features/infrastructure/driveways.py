"""
topocore.features.infrastructure.driveways
=============================================

Driveway footprint detection.

ROAD_SURFACE-classified points are spatially clustered and filtered
for the smaller, moderately elongated footprints characteristic of
private or local vehicle access ways.

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


class DrivewayDetector(ClusterDetectorBase):
    """
    Detect driveway footprints from classified point clouds.

    Candidate points classified as ``ROAD_SURFACE`` are spatially
    clustered in XY and filtered by horizontal size and elongation.

    Driveways are modeled as features that are generally more
    elongated than compact paved areas while remaining substantially
    smaller than road-scale surfaces.

    Parameters
    ----------
    eps
        Maximum XY distance between neighboring points during
        spatial clustering.
    min_points
        Minimum number of points required for a cluster.
    min_elongation
        Minimum horizontal elongation ratio accepted for a driveway.
    max_elongation
        Maximum horizontal elongation ratio accepted for a driveway.
    min_horizontal_extent
        Minimum footprint diagonal, in meters.
    max_horizontal_extent
        Maximum footprint diagonal, in meters.

    Notes
    -----
    Detection is implemented by `ClusterDetectorBase`. This class
    defines only the semantic classification and driveway-specific
    geometric filtering configuration.

    ``ROAD_SURFACE`` is shared with road and parking detectors.
    Consequently, these thresholds provide geometric candidate
    separation rather than a definitive semantic classification.
    """

    category = FeatureCategory.INFRASTRUCTURE
    feature_type = FeatureType.DRIVEWAY
    geometry_type = GeometryType.POLYGON
    classification_codes = (PointClassification.ROAD_SURFACE,)
    version = "1.0"

    def __init__(
        self,
        eps: float = 1.0,
        min_points: int = 20,
        min_elongation: float = 1.3,
        max_elongation: float = 4.0,
        min_horizontal_extent: float = 2.0,
        max_horizontal_extent: float = 40.0,
    ) -> None:
        super().__init__(
            ClusterFilterConfig(
                eps=eps,
                min_points=min_points,
                min_horizontal_extent=min_horizontal_extent,
                max_horizontal_extent=max_horizontal_extent,
                min_elongation=min_elongation,
                max_elongation=max_elongation,
            )
        )

    @override
    def name(self) -> str:
        return "driveways"


DetectorRegistry.register(DrivewayDetector)

__all__ = ["DrivewayDetector"]
