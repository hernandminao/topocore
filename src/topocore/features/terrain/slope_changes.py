"""
topocore.features.terrain.slope_changes
=========================================

Slope-change detection from a TIN: edges between adjacent triangles
whose slope magnitude (not full 3D dihedral angle) differs by more
than a threshold. Complements `breaklines.py`, which reacts to any
sharp fold regardless of whether it's a slope increase or a twist in
aspect; this reacts specifically to slope magnitude discontinuities.

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
    interior_edges,
    triangle_normals,
    triangle_slope_deg,
)


class SlopeChangeDetector(BaseFeatureDetector):
    """
    Detects slope-magnitude discontinuities from a TIN.

    Parameters
    ----------
    slope_delta_threshold_deg
        Minimum difference in slope angle (degrees) between two
        adjacent triangles for their shared edge to be flagged.
    min_length
        Minimum total chained-polyline length (meters) to report.
    """

    category = FeatureCategory.TERRAIN
    version = "1.0"
    required_inputs = frozenset({ContextField.TIN})

    __slots__ = ("_slope_delta_threshold_deg", "_min_length")

    def __init__(self, slope_delta_threshold_deg: float = 10.0, min_length: float = 1.0) -> None:
        if not 0.0 < slope_delta_threshold_deg < 90.0:
            raise DetectionError(f"slope_delta_threshold_deg must be in (0, 90); got {slope_delta_threshold_deg}.")
        if min_length < 0.0:
            raise DetectionError(f"min_length must be non-negative; got {min_length}.")
        self._slope_delta_threshold_deg = float(slope_delta_threshold_deg)
        self._min_length = float(min_length)

    @override
    def name(self) -> str:
        return "slope_changes"

    @override
    def _detect(self, context: DetectionContext) -> FeatureCollection:
        tin = context.tin
        assert tin is not None

        if not isinstance(tin, TINMesh):
            raise DetectionError("SlopeChangeDetector requires a TIN exposing `vertex_array()` and `.simplices`.")

        vertices = np.asarray(tin.vertex_array(), dtype=np.float64)
        triangles = np.asarray(tin.simplices, dtype=np.int64)

        if triangles.shape[0] == 0:
            return FeatureCollection()

        change_edges = self._detect_change_edges(vertices, triangles)
        chains = chain_edges(change_edges)

        result = FeatureCollection()
        for local_id, chain in enumerate(chains, start=1):
            chain_vertices = vertices[chain]
            length = float(np.sum(np.linalg.norm(np.diff(chain_vertices, axis=0), axis=1)))
            if length < self._min_length:
                continue

            result.add(
                Feature(
                    feature_id=local_id,
                    category=self.category,
                    feature_type=FeatureType.SLOPE_CHANGE,
                    geometry=FeatureGeometry(
                        geometry_type=GeometryType.POLYLINE,
                        vertices=chain_vertices,
                        closed=chain[0] == chain[-1],
                    ),
                    confidence=1.0,
                    metadata=self._metadata(
                        inputs_used=frozenset({ContextField.TIN}),
                        slope_delta_threshold_deg=self._slope_delta_threshold_deg,
                        length=length,
                    ),
                )
            )
        return result

    def _detect_change_edges(
        self,
        vertices: NDArray[np.float64],
        triangles: NDArray[np.int64],
    ) -> list[tuple[int, int]]:
        """
        Detect interior edges whose adjacent triangles exhibit a slope
        change greater than or equal to the configured threshold.

        Parameters
        ----------
        vertices
            Vertex coordinates as an ``(V, 3)`` array.
        triangles
            Triangle connectivity as an ``(T, 3)`` array of vertex
            indices.

        Returns
        -------
        list[tuple[int, int]]
            Undirected vertex-index pairs representing slope-change
            edges.
        """
        if triangles.size == 0:
            return []

        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise DetectionError(f"Triangles must have shape (T, 3); got {triangles.shape}.")

        if triangles.min() < 0:
            raise DetectionError("TIN contains negative vertex indices.")

        if triangles.max() >= len(vertices):
            raise DetectionError("TIN triangle references missing vertex.")

        normals = triangle_normals(vertices, triangles)
        slopes = triangle_slope_deg(normals)

        edge_map = build_edge_adjacency(triangles)
        edges_with_tris = interior_edges(edge_map)

        if not edges_with_tris:
            return []

        edges = [edge for edge, _ in edges_with_tris]

        tri_pairs = np.asarray(
            [pair for _, pair in edges_with_tris],
            dtype=np.int64,
        )

        slope_delta = np.abs(slopes[tri_pairs[:, 0]] - slopes[tri_pairs[:, 1]])

        keep = slope_delta >= self._slope_delta_threshold_deg

        return [edge for edge, is_change in zip(edges, keep.tolist(), strict=True) if is_change]


DetectorRegistry.register(SlopeChangeDetector)

__all__ = ["SlopeChangeDetector"]
