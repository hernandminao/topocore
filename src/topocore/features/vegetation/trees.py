"""
topocore.features.vegetation.trees
=====================================

Tree detection from HIGH_VEGETATION-classified point-cloud clusters.

Candidate vegetation points are clustered horizontally to separate
individual tree canopies. Each accepted cluster is represented by a
single XYZ point using the lowest-Z cluster point as an approximate
tree position.

The reported point should be interpreted as a representative tree
location derived from the classified canopy, not as an exact
reconstruction of the physical trunk position.

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


class TreeDetector(ClusterDetectorBase):
    """
    Detect individual trees from HIGH_VEGETATION points.

    Candidate points are clustered in XY space. Clusters that satisfy
    the configured minimum point count and vertical extent are
    reported as individual tree features.

    Parameters
    ----------
    eps
        Maximum horizontal distance between points for clustering.
    min_points
        Minimum number of points required for a tree candidate.
    min_height
        Minimum vertical extent, in meters, required for a cluster
        to be considered a tree.
    """

    category = FeatureCategory.VEGETATION
    feature_type = FeatureType.TREE
    geometry_type = GeometryType.POINT

    classification_codes = (PointClassification.HIGH_VEGETATION,)

    point_strategy = "min_z"
    version = "1.0"

    __slots__ = ()

    def __init__(
        self,
        eps: float = 1.5,
        min_points: int = 10,
        min_height: float = 2.0,
    ) -> None:
        super().__init__(
            ClusterFilterConfig(
                eps=eps,
                min_points=min_points,
                min_height=min_height,
            )
        )

    @override
    def name(self) -> str:
        """Return the detector registry name."""
        return "trees"


DetectorRegistry.register(TreeDetector)

__all__ = ["TreeDetector"]
