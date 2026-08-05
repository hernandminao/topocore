"""
topocore.features.terrain.embankments
========================================

Embankment (cut/fill slope) detection from a TIN: triangles whose
slope exceeds a threshold are grouped into connected regions (an
embankment face), and each region's boundary (crest + toe lines) is
extracted as the reported polyline. Distinct from `slope_changes.py`,
which flags a single discontinuity edge; this flags a whole
sustained-steep-slope band.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import NDArray

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
from topocore.features.terrain._mesh_utils import (
    TINMesh,
    build_edge_adjacency,
    chain_edges,
    triangle_normals,
    triangle_slope_deg,
)


class EmbankmentDetector(BaseFeatureDetector):
    """
    Detects embankments (cut/fill slopes) as connected regions of
    steep triangles, reporting each region's boundary polyline(s).

    Parameters
    ----------
    min_slope_deg
        Minimum triangle slope (degrees) to be considered part of an
        embankment face.
    min_triangle_count
        Minimum connected triangles for a region to be reported.
    min_length
        Minimum boundary-polyline length (meters) to report.
    """

    category = FeatureCategory.TERRAIN
    version = "1.0"
    required_inputs = frozenset({ContextField.TIN})

    __slots__ = (
        "_min_slope_deg",
        "_min_triangle_count",
        "_min_length",
    )

    def __init__(
        self,
        min_slope_deg: float = 35.0,
        min_triangle_count: int = 4,
        min_length: float = 2.0,
    ) -> None:
        if not 0.0 < min_slope_deg < 90.0:
            raise DetectionError(f"min_slope_deg must be in (0, 90); got {min_slope_deg}.")

        if min_triangle_count < 1:
            raise DetectionError(f"min_triangle_count must be >= 1; got {min_triangle_count}.")

        if min_length < 0.0:
            raise DetectionError(f"min_length must be non-negative; got {min_length}.")

        self._min_slope_deg = float(min_slope_deg)
        self._min_triangle_count = int(min_triangle_count)
        self._min_length = float(min_length)

    @override
    def name(self) -> str:
        return "embankments"

    @override
    def _detect(
        self,
        context: DetectionContext,
    ) -> FeatureCollection:
        tin = context.tin
        assert tin is not None

        if not isinstance(tin, TINMesh):
            raise DetectionError("EmbankmentDetector requires a TIN exposing `vertex_array()` and `.simplices`.")

        vertices = np.asarray(
            tin.vertex_array(),
            dtype=np.float64,
        )

        triangles = np.asarray(
            tin.simplices,
            dtype=np.int64,
        )

        if triangles.size == 0:
            return FeatureCollection()

        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise DetectionError(f"Triangles must have shape (T, 3); got {triangles.shape}.")

        if triangles.min() < 0:
            raise DetectionError("TIN contains negative vertex indices.")

        if triangles.max() >= len(vertices):
            raise DetectionError("TIN triangle references missing vertex.")

        normals = triangle_normals(vertices, triangles)
        slopes = triangle_slope_deg(normals)

        steep_mask = slopes >= self._min_slope_deg

        regions = self._connected_steep_regions(
            triangles,
            steep_mask,
        )

        result = FeatureCollection()
        local_id = 0

        for region in regions:
            if len(region) < self._min_triangle_count:
                continue

            boundary_edges = self._region_boundary_edges(
                triangles,
                set(region),
            )

            chains = chain_edges(boundary_edges)

            for chain in chains:
                chain_vertices = vertices[chain]

                length = float(
                    np.sum(
                        np.linalg.norm(
                            np.diff(chain_vertices, axis=0),
                            axis=1,
                        )
                    )
                )

                if length < self._min_length:
                    continue

                local_id += 1

                result.add(
                    Feature(
                        feature_id=local_id,
                        category=self.category,
                        feature_type=FeatureType.EMBANKMENT,
                        geometry=FeatureGeometry(
                            geometry_type=GeometryType.POLYLINE,
                            vertices=chain_vertices,
                            closed=chain[0] == chain[-1],
                        ),
                        confidence=1.0,
                        metadata=self._metadata(
                            inputs_used=frozenset({ContextField.TIN}),
                            min_slope_deg=self._min_slope_deg,
                            region_triangle_count=len(region),
                            length=length,
                        ),
                        attributes={
                            "mean_slope_deg": float(np.mean(slopes[region])),
                        },
                    )
                )

        return result

    @staticmethod
    def _connected_steep_regions(
        triangles: NDArray[np.int64],
        steep_mask: NDArray[np.bool_],
    ) -> list[list[int]]:
        """
        Compute connected components of steep triangles.

        Two triangles belong to the same region when they both
        satisfy ``steep_mask`` and share an interior edge.
        """
        edge_map = build_edge_adjacency(triangles)

        adjacency: dict[int, set[int]] = {}

        for triangle_indices in edge_map.values():
            if len(triangle_indices) == 2 and steep_mask[triangle_indices[0]] and steep_mask[triangle_indices[1]]:
                adjacency.setdefault(
                    triangle_indices[0],
                    set(),
                ).add(triangle_indices[1])

                adjacency.setdefault(
                    triangle_indices[1],
                    set(),
                ).add(triangle_indices[0])

        visited: set[int] = set()
        regions: list[list[int]] = []

        for triangle_index in np.flatnonzero(steep_mask):
            tri = int(triangle_index)

            if tri in visited:
                continue

            stack = [tri]
            region: list[int] = []

            while stack:
                current = stack.pop()

                if current in visited:
                    continue

                visited.add(current)
                region.append(current)

                stack.extend(adjacency.get(current, ()))

            regions.append(region)

        return regions

    @staticmethod
    def _region_boundary_edges(
        triangles: NDArray[np.int64],
        region: set[int],
    ) -> list[tuple[int, int]]:
        """
        Return the outer boundary edges of a connected triangle
        region.

        An edge belongs to the boundary when it appears exactly
        once among all triangles of the region.
        """
        counts: dict[tuple[int, int], int] = {}

        for triangle_index in region:
            a, b, c = triangles[triangle_index]

            for u, v in (
                (int(a), int(b)),
                (int(b), int(c)),
                (int(c), int(a)),
            ):
                edge = (u, v) if u < v else (v, u)
                counts[edge] = counts.get(edge, 0) + 1

        return [edge for edge, count in counts.items() if count == 1]


DetectorRegistry.register(EmbankmentDetector)

__all__ = ["EmbankmentDetector"]
