"""
topocore.features.infrastructure.curbs
=========================================

Curb detection from a TIN.

Curbs are detected as linear surface discontinuities using the same
dihedral-angle edge analysis employed by terrain breakline detection,
but with thresholds tuned for the smaller-scale discontinuities
typical of curb geometry.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import replace
from typing import override

from topocore.features.detector import DetectorRegistry
from topocore.features.models import (
    FeatureCategory,
    FeatureCollection,
    FeatureType,
)
from topocore.features.protocols import DetectionContext
from topocore.features.terrain.breaklines import BreaklineDetector


class CurbDetector(BreaklineDetector):
    """
    Detect curbs as shallow-angle TIN edge discontinuities.

    This detector specializes `BreaklineDetector` because curbs and
    terrain breaklines share the same geometric primitive: an edge
    shared by two triangles whose surface normals differ by a
    significant angle.

    Curbs generally occur at a smaller scale than natural terrain
    breaklines, so this detector uses a lower default angular
    threshold and minimum reported length.

    Parameters
    ----------
    angle_threshold_deg
        Minimum dihedral angle, in degrees, between adjacent
        triangles for their shared edge to be considered a curb
        candidate.
    min_length
        Minimum chained curb-polyline length, in meters, required
        for the feature to be reported.
    """

    category = FeatureCategory.INFRASTRUCTURE
    feature_type = FeatureType.CURB
    version = "1.0"

    def __init__(
        self,
        angle_threshold_deg: float = 8.0,
        min_length: float = 0.3,
    ) -> None:
        super().__init__(
            angle_threshold_deg=angle_threshold_deg,
            min_length=min_length,
        )

    @override
    def name(self) -> str:
        return "curbs"

    @override
    def _detect(
        self,
        context: DetectionContext,
    ) -> FeatureCollection:
        collection = super()._detect(context)

        collection.features = [
            replace(
                feature,
                category=self.category,
                feature_type=self.feature_type,
            )
            for feature in collection.features
        ]

        return collection


DetectorRegistry.register(CurbDetector)

__all__ = ["CurbDetector"]
