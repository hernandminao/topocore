"""
topocore.features.vegetation.shrubs
======================================

Shrub detection from LOW_VEGETATION and MEDIUM_VEGETATION
classified point-cloud clusters.

Candidate vegetation points are clustered horizontally and filtered
by vertical extent to distinguish shrub-scale vegetation from
ground-level vegetation and taller tree-like structures.

Each accepted cluster is represented by a single XYZ point using
the lowest-Z cluster point as an approximate shrub position. The
reported point is therefore a representative location derived from
the classified vegetation cluster, not an exact stem position.

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


class ShrubDetector(ClusterDetectorBase):
    """
    Detect shrubs from LOW_VEGETATION and MEDIUM_VEGETATION points.

    Candidate points are clustered in XY space. Clusters are then
    filtered by vertical extent so only vegetation within the
    configured shrub-height range is reported.

    Parameters
    ----------
    eps
        Maximum horizontal distance between points for clustering.
    min_points
        Minimum number of points required for a shrub candidate.
    min_height
        Minimum vertical extent, in meters, required for a cluster.
    max_height
        Maximum vertical extent, in meters, allowed for a cluster.
    """

    category = FeatureCategory.VEGETATION
    feature_type = FeatureType.SHRUB
    geometry_type = GeometryType.POINT

    classification_codes = (
        PointClassification.LOW_VEGETATION,
        PointClassification.MEDIUM_VEGETATION,
    )

    point_strategy = "min_z"
    version = "1.0"

    __slots__ = ()

    def __init__(
        self,
        eps: float = 0.8,
        min_points: int = 8,
        min_height: float = 0.3,
        max_height: float = 2.0,
    ) -> None:
        super().__init__(
            ClusterFilterConfig(
                eps=eps,
                min_points=min_points,
                min_height=min_height,
                max_height=max_height,
            )
        )

    @override
    def name(self) -> str:
        """Return the detector registry name."""
        return "shrubs"


DetectorRegistry.register(ShrubDetector)

__all__ = ["ShrubDetector"]
