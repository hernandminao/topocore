"""
topocore.features.terrain._mesh_utils
=======================================

Shared TIN mesh utilities for terrain detectors: triangle normals,
edge->triangle adjacency, and edge chaining into polylines. Used by
`breaklines.py`, `slope_changes.py`, and `embankments.py` so the
edge-adjacency construction (an O(T) pass over `simplices`) isn't
duplicated three times.

Private to `features.terrain`.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class TINMesh(Protocol):
    def vertex_array(self) -> NDArray[np.float64]: ...
    @property
    def simplices(self) -> NDArray[np.int32]: ...


def triangle_normals(
    vertices: NDArray[np.float64],
    triangles: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Unit outward normal per triangle, shape (T, 3)."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1)
    safe = np.where(norms > 1e-15, norms, 1.0)
    return normals / safe[:, None]


def triangle_slope_deg(normals: NDArray[np.float64]) -> NDArray[np.float64]:
    """Slope angle (degrees from horizontal) implied by each normal."""
    cos_from_vertical = np.clip(np.abs(normals[:, 2]), 0.0, 1.0)
    return np.degrees(np.arccos(cos_from_vertical))


def build_edge_adjacency(triangles: NDArray[np.int64]) -> dict[tuple[int, int], list[int]]:
    """
    Map each undirected edge (sorted vertex-index pair) to the list
    of triangle indices sharing it (length 1 = boundary, 2 = interior).
    """
    edge_map: dict[tuple[int, int], list[int]] = {}
    for tri_idx, (a, b, c) in enumerate(triangles.tolist()):
        for u, w in ((a, b), (b, c), (c, a)):
            key = (u, w) if u < w else (w, u)
            edge_map.setdefault(key, []).append(tri_idx)
    return edge_map


def interior_edges(
    edge_map: dict[tuple[int, int], list[int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Edges shared by exactly two triangles, as ``(edge, (tri_a, tri_b))``."""
    return [(edge, (tris[0], tris[1])) for edge, tris in edge_map.items() if len(tris) == 2]


def opposite_vertex(triangle: tuple[int, int, int], edge: tuple[int, int]) -> int:
    """The one vertex of `triangle` not part of `edge`."""
    return next(v for v in triangle if v not in edge)


def chain_edges(edges: list[tuple[int, int]]) -> list[list[int]]:
    """
    Chain undirected edges into polylines (open) or loops (closed).

    Open chains are traced from degree-1 vertices first; whatever
    remains afterward consists entirely of closed loops.
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
            next_v = next(
                (nb for nb in adjacency[current] if edge_key(current, nb) not in visited),
                None,
            )
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


__all__ = [
    "TINMesh",
    "triangle_normals",
    "triangle_slope_deg",
    "build_edge_adjacency",
    "interior_edges",
    "opposite_vertex",
    "chain_edges",
]
