"""
topocore.features.terrain.contours
=====================================

Contour line detection — a thin adapter over
`topocore.terrain.contours.ContourGenerator`, which already
implements marching-triangles contour extraction with vertex
welding and open/closed polyline stitching.

This module deliberately does not reimplement contour generation.
It converts each resulting `ContourLine` into TopoCore's common
`Feature` representation for downstream CAD/GIS consumers.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math
from typing import override

import numpy as np

from topocore.features.base import BaseFeatureDetector
from topocore.features.detector import DetectorRegistry
from topocore.features.exceptions import DetectionError
from topocore.features.models import (
    ContextField,
    Feature,
    FeatureCategory,
    FeatureCollection,
    FeatureGeometry,
    FeatureType,
    GeometryType,
)
from topocore.features.protocols import DetectionContext
from topocore.terrain.constants import DEFAULT_CONTOUR_INTERVAL
from topocore.terrain.contours import ContourGenerator
from topocore.terrain.exceptions import ContourError


class ContourDetector(BaseFeatureDetector):
    """
    Detect contour lines from a TIN via `ContourGenerator`.

    Parameters
    ----------
    interval
        Vertical spacing between consecutive contour levels.
        Must be finite and strictly positive.
    base
        Elevation offset used to align contour levels.
    min_length
        Minimum 3D polyline length, in meters, required for a
        generated contour to be reported. ``0.0`` keeps every
        valid contour.
    """

    category = FeatureCategory.TERRAIN
    feature_type = FeatureType.CONTOUR
    version = "1.0"
    required_inputs = frozenset({ContextField.TIN})

    __slots__ = ("_interval", "_base", "_min_length")

    def __init__(
        self,
        interval: float = DEFAULT_CONTOUR_INTERVAL,
        base: float = 0.0,
        min_length: float = 0.0,
    ) -> None:
        interval = float(interval)
        base = float(base)
        min_length = float(min_length)

        if not math.isfinite(interval) or interval <= 0.0:
            raise DetectionError(f"interval must be finite and > 0; got {interval}.")

        if not math.isfinite(base):
            raise DetectionError(f"base must be finite; got {base}.")

        if not math.isfinite(min_length) or min_length < 0.0:
            raise DetectionError(f"min_length must be finite and non-negative; got {min_length}.")

        self._interval = interval
        self._base = base
        self._min_length = min_length

    @override
    def name(self) -> str:
        """Return the detector registry name."""
        return "contours"

    @override
    def _detect(self, context: DetectionContext) -> FeatureCollection:
        tin = context.tin
        assert tin is not None

        try:
            contour_lines = ContourGenerator(tin).generate(
                self._interval,
                base=self._base,
            )
        except ContourError as exc:
            raise DetectionError(f"Contour generation failed: {exc}") from exc

        result = FeatureCollection()
        local_id = 0

        for line in contour_lines:
            vertices = np.asarray(
                [(point.x, point.y, point.z) for point in line.points],
                dtype=np.float64,
            )

            # FeatureGeometry.POLYLINE requires at least two
            # vertices. ContourGenerator should normally guarantee
            # this, but the adapter keeps its own output contract
            # protected.
            if vertices.shape[0] < 2:
                continue

            length = float(
                np.linalg.norm(
                    np.diff(vertices, axis=0),
                    axis=1,
                ).sum()
            )

            if length < self._min_length:
                continue

            local_id += 1

            result.add(
                Feature(
                    feature_id=local_id,
                    category=self.category,
                    feature_type=self.feature_type,
                    geometry=FeatureGeometry(
                        geometry_type=GeometryType.POLYLINE,
                        vertices=vertices,
                        closed=bool(line.closed),
                    ),
                    confidence=1.0,
                    metadata=self._metadata(
                        inputs_used=frozenset({ContextField.TIN}),
                        interval=self._interval,
                        base=self._base,
                        min_length=self._min_length,
                        length=length,
                    ),
                    attributes={
                        "elevation": float(line.elevation),
                        "closed": bool(line.closed),
                    },
                )
            )

        return result


DetectorRegistry.register(ContourDetector)

__all__ = ["ContourDetector"]
