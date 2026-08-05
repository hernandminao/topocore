"""
topocore.features.drainage.drainage
======================================

Drainage line detection from a TIN.

Candidate drainage lines are extracted from interior TIN edges whose
opposite triangle vertices lie sufficiently above the shared edge.
This valley-depth criterion identifies local downward terrain folds
that may represent water-collecting linear features.

Detected valley edges are chained into 3D XYZ polylines and filtered
by minimum length.

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
    opposite_vertex,
)


class DrainageDetector(BaseFeatureDetector):
    """
    Detect drainage valley lines from a TIN.

    Interior TIN edges are evaluated using the elevations of the
    opposite vertices of their two adjacent triangles. An edge is
    considered a valley candidate when both opposite vertices lie
    at least ``min_depth`` above the mean elevation of the shared
    edge.

    Connected candidate edges are chained into XYZ polylines.

    Parameters
    ----------
    min_depth
        Minimum vertical difference, in meters, between each
        opposite vertex and the shared-edge mean elevation.
    min_length
        Minimum 3D length, in meters, required for a chained
        drainage polyline to be reported.
    """

    category = FeatureCategory.DRAINAGE
    feature_type = FeatureType.DRAINAGE
    version = "1.0"
    required_inputs = frozenset({ContextField.TIN})

    __slots__ = ("_min_depth", "_min_length")

    def __init__(
        self,
        min_depth: float = 0.05,
        min_length: float = 1.0,
    ) -> None:
        if min_depth < 0.0:
            raise DetectionError(f"min_depth must be non-negative; got {min_depth}.")

        if min_length < 0.0:
            raise DetectionError(f"min_length must be non-negative; got {min_length}.")

        self._min_depth = float(min_depth)
        self._min_length = float(min_length)

    @override
    def name(self) -> str:
        return "drainage"

    @override
    def _detect(
        self,
        context: DetectionContext,
    ) -> FeatureCollection:
        tin = context.tin
        assert tin is not None

        if not isinstance(tin, TINMesh):
            raise DetectionError("DrainageDetector requires a TIN exposing `vertex_array()` and `.simplices`.")

        vertices = np.asarray(
            tin.vertex_array(),
            dtype=np.float64,
        )
        triangles = np.asarray(
            tin.simplices,
            dtype=np.int64,
        )

        self._validate_mesh(
            vertices,
            triangles,
        )

        if triangles.shape[0] == 0:
            return FeatureCollection()

        valley_edges = self._detect_valley_edges(
            vertices,
            triangles,
        )

        if not valley_edges:
            return FeatureCollection()

        chains = chain_edges(valley_edges)

        result = FeatureCollection()
        local_id = 0

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
                    feature_type=self.feature_type,
                    geometry=FeatureGeometry(
                        geometry_type=GeometryType.POLYLINE,
                        vertices=chain_vertices,
                        closed=chain[0] == chain[-1],
                    ),
                    confidence=1.0,
                    metadata=self._metadata(
                        inputs_used=frozenset(
                            {
                                ContextField.TIN,
                            }
                        ),
                        min_depth=self._min_depth,
                        min_length=self._min_length,
                        length=length,
                        edge_count=len(chain) - 1,
                        closed=chain[0] == chain[-1],
                    ),
                    attributes={
                        "length_m": length,
                        "edge_count": len(chain) - 1,
                    },
                )
            )

        return result

    @staticmethod
    def _validate_mesh(
        vertices: NDArray[np.float64],
        triangles: NDArray[np.int64],
    ) -> None:
        """
        Validate arrays required by drainage detection.

        Raises
        ------
        DetectionError
            If vertex or triangle arrays have invalid shapes,
            contain non-finite coordinates, or reference invalid
            vertex indices.
        """
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise DetectionError(f"TIN vertices must have shape (n, 3); got {vertices.shape}.")

        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise DetectionError(f"TIN simplices must have shape (m, 3); got {triangles.shape}.")

        if not np.all(np.isfinite(vertices)):
            raise DetectionError("TIN vertices must contain only finite coordinates.")

        if triangles.shape[0] == 0:
            return

        if vertices.shape[0] == 0:
            raise DetectionError("TIN contains triangles but no vertices.")

        if int(triangles.min()) < 0:
            raise DetectionError("TIN contains negative vertex indices.")

        if int(triangles.max()) >= vertices.shape[0]:
            raise DetectionError("TIN triangle references missing vertex.")

    def _detect_valley_edges(
        self,
        vertices: NDArray[np.float64],
        triangles: NDArray[np.int64],
    ) -> list[tuple[int, int]]:
        """
        Detect interior edges satisfying the valley-depth criterion.

        For each interior edge, the elevation at the edge is
        approximated by the mean Z of its two endpoints. The
        opposite vertex of each adjacent triangle must lie at least
        ``min_depth`` above that elevation.

        Parameters
        ----------
        vertices
            XYZ vertex array with shape ``(n, 3)``.
        triangles
            Triangle connectivity array with shape ``(m, 3)``.

        Returns
        -------
        list[tuple[int, int]]
            Canonical vertex-index pairs representing detected
            valley edges.
        """
        edge_map = build_edge_adjacency(triangles)
        edges_with_tris = interior_edges(edge_map)

        if not edges_with_tris:
            return []

        valley_edges: list[tuple[int, int]] = []

        for edge, (tri_a, tri_b) in edges_with_tris:
            edge_z = float((vertices[edge[0], 2] + vertices[edge[1], 2]) * 0.5)

            triangle_a = (
                int(triangles[tri_a, 0]),
                int(triangles[tri_a, 1]),
                int(triangles[tri_a, 2]),
            )
            triangle_b = (
                int(triangles[tri_b, 0]),
                int(triangles[tri_b, 1]),
                int(triangles[tri_b, 2]),
            )

            opposite_a = opposite_vertex(
                triangle_a,
                edge,
            )
            opposite_b = opposite_vertex(
                triangle_b,
                edge,
            )

            depth_a = float(vertices[opposite_a, 2] - edge_z)
            depth_b = float(vertices[opposite_b, 2] - edge_z)

            if depth_a >= self._min_depth and depth_b >= self._min_depth:
                valley_edges.append(edge)

        return valley_edges


DetectorRegistry.register(DrainageDetector)

__all__ = ["DrainageDetector"]
