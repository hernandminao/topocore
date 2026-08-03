"""
topocore.features.terrain.breaklines
======================================

Breakline detection from a TIN.

A breakline is a linear terrain feature marking an abrupt change in
slope (ridges, valleys, edges) — critical for accurate TIN/DTM
generation and for CAD/GIS deliverables. This detector finds them by
computing the dihedral angle between every pair of triangles sharing
an interior edge: an edge whose two triangles fold sharply relative
to each other is a break edge. Connected break edges are then
chained into polylines.

The edge->triangle adjacency is built directly from the TIN's own
`vertex_array()` / `simplices`, rather than relying on
`TIN.neighbors`/`neighbors_of()`. That keeps this detector
independent of whichever Delaunay backend produced the TIN, works
for any triangular mesh (not just Delaunay ones), and keeps it
testable against small synthetic meshes.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Protocol, override, runtime_checkable

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


@runtime_checkable
class _TINMesh(Protocol):
    """
    Minimal structural need from a TIN for edge-adjacency analysis.
    """

    def vertex_array(self) -> NDArray[np.float64]: ...  # (V, 3)

    @property
    def simplices(self) -> NDArray[np.int32]: ...  # (T, 3), vertex indices


class BreaklineDetector(BaseFeatureDetector):
    """
    Detects breaklines from a TIN via dihedral-angle edge analysis.

    Parameters
    ----------
    angle_threshold_deg
        Minimum dihedral angle (degrees) between two adjacent
        triangles' normals for their shared edge to be considered a
        break edge. Typical range 15-45 degrees; lower values catch
        gentler slope changes at the cost of more noise on rough
        (e.g. vegetation-contaminated) TINs.
    min_length
        Minimum total length (meters) of a chained polyline for it
        to be reported. Filters out short, likely-spurious segments
        from local noise.
    """

    category = FeatureCategory.TERRAIN
    version = "1.0"
    required_inputs = frozenset({ContextField.TIN})

    __slots__ = ("_angle_threshold_deg", "_min_length")

    def __init__(
        self,
        angle_threshold_deg: float = 25.0,
        min_length: float = 1.0,
    ) -> None:
        if not 0.0 < angle_threshold_deg < 180.0:
            raise DetectionError(f"angle_threshold_deg must be in (0, 180); got {angle_threshold_deg}.")
        if min_length < 0.0:
            raise DetectionError(f"min_length must be non-negative; got {min_length}.")

        self._angle_threshold_deg = float(angle_threshold_deg)
        self._min_length = float(min_length)

    @override
    def name(self) -> str:
        return "breaklines"

    @override
    def _detect(self, context: DetectionContext) -> FeatureCollection:
        tin = context.tin
        assert tin is not None  # guaranteed by required_inputs validation in detect()

        if not isinstance(tin, _TINMesh):
            raise DetectionError(
                "BreaklineDetector requires a TIN exposing `vertex_array()` "
                "((V,3) float array) and `.simplices` ((T,3) int array)."
            )

        vertices = np.asarray(tin.vertex_array(), dtype=np.float64)
        triangles = np.asarray(tin.simplices, dtype=np.int64)

        if triangles.shape[0] == 0:
            return FeatureCollection()

        # Validación de forma
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise DetectionError("TIN vertex array must have shape (V,3).")

        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise DetectionError("TIN simplices must have shape (T,3).")

        # Validación de índices
        if triangles.size > 0:
            min_index = int(triangles.min())
            max_index = int(triangles.max())

            if min_index < 0:
                raise DetectionError("TIN contains negative vertex indices.")

            if max_index >= vertices.shape[0]:
                raise DetectionError("TIN triangle references a vertex outside the vertex array.")

        break_edges = self._detect_break_edges(vertices, triangles)
        chains = self._chain_edges(break_edges)

        result = FeatureCollection()

        for local_id, chain in enumerate(chains, start=1):
            chain_vertices = vertices[chain]
            length = float(np.sum(np.linalg.norm(np.diff(chain_vertices, axis=0), axis=1)))

            if length < self._min_length:
                continue

            geometry = FeatureGeometry(
                geometry_type=GeometryType.POLYLINE,
                vertices=chain_vertices,
                closed=chain[0] == chain[-1],
            )

            result.add(
                Feature(
                    feature_id=local_id,
                    category=self.category,
                    feature_type=FeatureType.BREAKLINE,
                    geometry=geometry,
                    confidence=1.0,
                    metadata=self._metadata(
                        inputs_used=frozenset({ContextField.TIN}),
                        angle_threshold_deg=self._angle_threshold_deg,
                        min_length=self._min_length,
                        length=length,
                    ),
                )
            )

        return result

    def _detect_break_edges(
        self,
        vertices: NDArray[np.float64],
        triangles: NDArray[np.int64],
    ) -> list[tuple[int, int]]:
        """
        Find edges whose two adjacent triangles fold at an angle
        greater than or equal to `angle_threshold_deg`.
        """
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]

        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1)
        safe_norms = np.where(norms > 1e-15, norms, 1.0)
        normals = normals / safe_norms[:, None]

        # Single O(T) pass to build edge -> [triangle indices]. This
        # is inherent to reading topology from a per-triangle index
        # array; it happens once per TIN, not once per point.
        edge_map: dict[tuple[int, int], list[int]] = {}
        for tri_idx, (a, b, c) in enumerate(triangles.tolist()):
            for u, w in ((a, b), (b, c), (c, a)):
                key = (u, w) if u < w else (w, u)
                edge_map.setdefault(key, []).append(tri_idx)

        interior_edges = [(edge, tris) for edge, tris in edge_map.items() if len(tris) == 2]

        if not interior_edges:
            return []

        edges = [edge for edge, _ in interior_edges]
        tri_pairs = np.array([tris for _, tris in interior_edges], dtype=np.int64)

        n1 = normals[tri_pairs[:, 0]]
        n2 = normals[tri_pairs[:, 1]]
        cos_angle = np.clip(np.einsum("ed,ed->e", n1, n2), -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(cos_angle))

        threshold_mask = angles_deg >= self._angle_threshold_deg

        return [edge for edge, keep in zip(edges, threshold_mask.tolist(), strict=True) if keep]

    @staticmethod
    def _chain_edges(edges: list[tuple[int, int]]) -> list[list[int]]:
        """
        Chain undirected edges into polylines (open) or loops (closed).

        Open chains are traced starting from degree-1 vertices first;
        whatever remains after that consists entirely of closed
        loops, traced separately.
        """
        adjacency: dict[int, set[int]] = {}
        for a, b in edges:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

        def edge_key(u: int, v: int) -> tuple[int, int]:
            return (u, v) if u < v else (v, u)

        visited: set[tuple[int, int]] = set()
        chains: list[list[int]] = []

        def trace(start: int) -> list[int]:
            chain = [start]
            current = start
            while True:
                next_v = next(nb for nb in sorted(adjacency[current]) if edge_key(current, nb) not in visited)
                if next_v is None:
                    break
                visited.add(edge_key(current, next_v))
                chain.append(next_v)
                current = next_v
                if current == start:
                    break
            return chain

        endpoints = [v for v, nbs in adjacency.items() if len(nbs) == 1]
        for start in endpoints:
            if all(edge_key(start, nb) in visited for nb in adjacency[start]):
                continue
            chain = trace(start)
            if len(chain) >= 2:
                chains.append(chain)

        remaining = [v for v, nbs in adjacency.items() if any(edge_key(v, nb) not in visited for nb in nbs)]
        for start in remaining:
            if all(edge_key(start, nb) in visited for nb in adjacency[start]):
                continue
            chain = trace(start)
            if len(chain) >= 2:
                chains.append(chain)

        return chains


DetectorRegistry.register(BreaklineDetector)

__all__ = ["BreaklineDetector"]
