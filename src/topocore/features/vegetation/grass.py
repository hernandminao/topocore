"""
topocore.features.vegetation.grass
=====================================

Grass-area detection from LOW_VEGETATION-classified point-cloud
clusters.

Candidate points are clustered horizontally and filtered by vertical
and horizontal extent. Broad clusters with limited vertical extent
are reported as polygon footprints representing grass-covered areas.

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


class GrassDetector(ClusterDetectorBase):
    """
    Detect grass-covered areas from LOW_VEGETATION points.

    Candidate points are clustered in XY space. Clusters are filtered
    by vertical extent and minimum horizontal extent so that broad,
    low vegetation regions are retained while small or vertically
    developed vegetation clusters are rejected.

    Accepted clusters are represented by polygon footprints.

    Parameters
    ----------
    eps
        Maximum horizontal distance between points for clustering.
    min_points
        Minimum number of points required for a grass-area candidate.
    max_height
        Maximum vertical extent, in meters, allowed for a cluster.
    min_horizontal_extent
        Minimum horizontal footprint extent, in meters, required for
        a cluster to represent a meaningful grass-covered area.
    """

    category = FeatureCategory.VEGETATION
    feature_type = FeatureType.GRASS
    geometry_type = GeometryType.POLYGON

    classification_codes = (PointClassification.LOW_VEGETATION,)

    version = "1.0"

    __slots__ = ()

    def __init__(
        self,
        eps: float = 2.0,
        min_points: int = 100,
        max_height: float = 0.3,
        min_horizontal_extent: float = 3.0,
    ) -> None:
        super().__init__(
            ClusterFilterConfig(
                eps=eps,
                min_points=min_points,
                max_height=max_height,
                min_horizontal_extent=min_horizontal_extent,
            )
        )

    @override
    def name(self) -> str:
        """Return the detector registry name."""
        return "grass"


DetectorRegistry.register(GrassDetector)

__all__ = ["GrassDetector"]
