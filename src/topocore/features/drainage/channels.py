"""
topocore.features.drainage.channels
======================================

Channel detection from a TIN.

Channels are detected as deeper and more sustained valley lines than
general surface drainage. The detector specializes
`DrainageDetector`, reusing its TIN valley-edge detection and edge
chaining while applying stricter default depth and length thresholds.

Typical targets include drainage ditches, natural channels, and
stream-like terrain depressions represented by the TIN.

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
from topocore.features.drainage.drainage import DrainageDetector
from topocore.features.models import FeatureCategory, FeatureType


class ChannelDetector(DrainageDetector):
    """
    Detect channels as deep, sustained valley lines.

    This detector specializes `DrainageDetector`. It uses the same
    valley-depth criterion and polyline chaining algorithm, but with
    stricter default thresholds appropriate for more pronounced
    drainage structures.

    Parameters
    ----------
    min_depth
        Minimum vertical difference, in meters, between each
        adjacent triangle's opposite vertex and the shared-edge mean
        elevation.
    min_length
        Minimum 3D length, in meters, required for a chained channel
        polyline to be reported.
    """

    category = FeatureCategory.DRAINAGE
    feature_type = FeatureType.CHANNEL
    version = "1.0"

    def __init__(
        self,
        min_depth: float = 0.3,
        min_length: float = 3.0,
    ) -> None:
        super().__init__(
            min_depth=min_depth,
            min_length=min_length,
        )

    @override
    def name(self) -> str:
        return "channels"


DetectorRegistry.register(ChannelDetector)

__all__ = ["ChannelDetector"]
