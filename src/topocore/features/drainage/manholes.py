"""
topocore.features.drainage.manholes
======================================

Manhole detection from classified point-cloud clusters.

ASPRS LAS classification codes do not define a dedicated manhole
class. Therefore, the classification bucket searched by this
detector is configurable per instance. The selected code is exposed
to the shared `ClusterDetectorBase` pipeline through
`_classification_codes()`, avoiding mutation of the class-level
`classification_codes` contract.

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


class ManholeDetector(ClusterDetectorBase):
    """
    Detect manhole covers as small, compact point clusters.

    Manholes do not have a dedicated ASPRS LAS classification code,
    so the classification bucket to inspect is configurable for each
    detector instance.

    Parameters
    ----------
    classification_code
        Classification bucket searched for candidate manhole points.
        Defaults to `PointClassification.UNCLASSIFIED`.
    eps
        Maximum horizontal distance between points for clustering.
    min_points
        Minimum number of points required for a candidate cluster.
    min_horizontal_extent
        Minimum expected horizontal extent in meters.
    max_horizontal_extent
        Maximum expected horizontal extent in meters.
    max_elongation
        Maximum XY elongation ratio. Compact clusters are preferred
        because manhole covers are expected to be approximately
        circular or square in plan view.
    """

    category = FeatureCategory.DRAINAGE
    feature_type = FeatureType.MANHOLE
    geometry_type = GeometryType.POINT
    version = "1.0"

    __slots__ = ("_classification_code",)

    def __init__(
        self,
        classification_code: PointClassification = PointClassification.UNCLASSIFIED,
        eps: float = 0.3,
        min_points: int = 5,
        min_horizontal_extent: float = 0.3,
        max_horizontal_extent: float = 1.2,
        max_elongation: float = 1.5,
    ) -> None:
        self._classification_code = classification_code

        super().__init__(
            ClusterFilterConfig(
                eps=eps,
                min_points=min_points,
                min_horizontal_extent=min_horizontal_extent,
                max_horizontal_extent=max_horizontal_extent,
                max_elongation=max_elongation,
            )
        )

    @override
    def name(self) -> str:
        """Return the detector registry name."""
        return "manholes"

    @override
    def _classification_codes(
        self,
    ) -> tuple[PointClassification, ...]:
        """
        Return the classification bucket configured for this instance.

        Manholes do not have a dedicated ASPRS classification code,
        so unlike most cluster-based detectors this value is selected
        when the detector instance is created.
        """
        return (self._classification_code,)


DetectorRegistry.register(ManholeDetector)

__all__ = ["ManholeDetector"]
