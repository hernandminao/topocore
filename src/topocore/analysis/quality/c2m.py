"""
topocore.analysis.quality.c2m
==============================

Cloud-to-Mesh (C2M) distance analysis.

Computes the point-to-mesh distance between a point cloud and a
triangulated surface (TIN). For each point, the shortest distance
to the mesh is computed by finding the closest point on the nearest
triangle face.

Performance
-----------
Instead of testing every point against every triangle (O(N*T) in
pure Python), this module builds a uniform spatial grid over the
mesh's triangle bounding boxes once per TIN (cached by `id(tin)`).
Points sharing a grid cell are grouped and their distances to
nearby triangles are computed together as a single vectorized NumPy
operation, with the candidate search expanding outward in rings
only as far as a proven distance bound requires.

``TriangulatedSurface`` only exposes ``triangle_count``,
``triangle_vertices(j)``, ``contains(x, y)`` and ``bounds``, so
building the index still requires one Python-level pass over all
triangles (unavoidable given that API), but this now happens once
per mesh instead of once per point.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from topocore.analysis.exceptions import QualityError
from topocore.analysis.protocols import TriangulatedSurface
from topocore.analysis.types import CloudToMeshResult

# NumPy einsum signatures
_EINSUM_DOT_CD = "cd,cd->c"
_EINSUM_POINT_CD = "mcd,cd->mc"


@dataclass(slots=True)
class _TriangleSpatialIndex:
    """
    Uniform spatial grid over a mesh's triangles, keyed by 2D cell.

    A triangle is registered under every cell its XY bounding box
    overlaps, so lookups by cell are conservative (never miss a
    triangle) at the cost of occasionally listing a triangle in more
    than one cell.
    """

    v0: NDArray[np.float64]
    v1: NDArray[np.float64]
    v2: NDArray[np.float64]
    min_x: float
    min_y: float
    cell_size: float
    n_cols: int
    n_rows: int
    cells: dict[int, NDArray[np.int64]]


def _ring_cells(
    row: int,
    col: int,
    radius: int,
    n_rows: int,
    n_cols: int,
) -> list[tuple[int, int]]:
    """
    Return grid cells at exact Chebyshev distance `radius` from
    (row, col), clipped to the grid bounds.
    """
    if radius == 0:
        if 0 <= row < n_rows and 0 <= col < n_cols:
            return [(row, col)]
        return []

    cells: list[tuple[int, int]] = []
    r0, r1 = row - radius, row + radius
    c0, c1 = col - radius, col + radius

    for c in range(max(c0, 0), min(c1, n_cols - 1) + 1):
        if 0 <= r0 < n_rows:
            cells.append((r0, c))
        if 0 <= r1 < n_rows and r1 != r0:
            cells.append((r1, c))

    for r in range(max(r0 + 1, 0), min(r1 - 1, n_rows - 1) + 1):
        if 0 <= c0 < n_cols:
            cells.append((r, c0))
        if 0 <= c1 < n_cols and c1 != c0:
            cells.append((r, c1))

    return cells


class CloudToMeshDistance:
    """
    Computes Cloud-to-Mesh distances.

    Parameters
    ----------
    max_distance
        Maximum search distance (meters). Zero means unlimited.
    """

    __slots__ = ("_cache_index", "_cache_tin_id", "_max_distance")

    def __init__(self, max_distance: float = 0.0) -> None:
        if not np.isfinite(max_distance):
            raise QualityError("Max distance must be finite.")
        if max_distance < 0:
            raise QualityError("Max distance cannot be negative.")
        self._max_distance = float(max_distance)
        self._cache_tin_id: int | None = None
        self._cache_index: _TriangleSpatialIndex | None = None

    @property
    def max_distance(self) -> float:
        return self._max_distance

    def compute(
        self,
        pointcloud: NDArray[np.float64],
        tin: TriangulatedSurface,
    ) -> CloudToMeshResult:
        """
        Compute C2M distances from a point cloud to a TIN.

        Parameters
        ----------
        pointcloud
            Point cloud as ``(n, 3)`` array.
        tin
            A ``TIN`` instance from ``topocore.terrain``.

        Returns
        -------
        CloudToMeshResult
            Distance statistics and per-point distances.

        Raises
        ------
        QualityError
            If the point cloud is empty or the TIN has no triangles.
        """
        pointcloud = np.asarray(pointcloud, dtype=np.float64)

        if pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
            raise QualityError("Point cloud must have shape (n, 3).")
        if pointcloud.shape[0] == 0:
            raise QualityError("Point cloud must not be empty.")
        if not np.isfinite(pointcloud).all():
            raise QualityError("Point cloud contains NaN or infinite values.")
        if tin.triangle_count == 0:
            raise QualityError("TIN has no triangles.")

        index = self._get_index(tin)
        distances = self._query_distances(pointcloud, index)

        if self._max_distance > 0:
            distances = np.where(
                distances <= self._max_distance,
                distances,
                np.inf,
            )

        unmatched = ~np.isfinite(distances)

        if np.any(unmatched):
            unmatched_count = int(np.count_nonzero(unmatched))
            raise QualityError(
                "Cloud-to-Mesh correspondence incomplete: "
                f"{unmatched_count} of {distances.size} point-cloud points "
                "have no mesh distance within max_distance."
            )

        # Every input point must participate in the statistics.  Silently
        # dropping points outside max_distance would make the quality result
        # look better than the actual coverage of the correspondence.
        std = float(np.std(distances, ddof=1)) if distances.size > 1 else 0.0

        return CloudToMeshResult(
            mean=float(np.mean(distances)),
            std=std,
            minimum=float(np.min(distances)),
            maximum=float(np.max(distances)),
            median=float(np.median(distances)),
            distances=distances,
        )

    def __call__(
        self,
        pointcloud: NDArray[np.float64],
        tin: TriangulatedSurface,
    ) -> CloudToMeshResult:
        return self.compute(pointcloud, tin)

    # ------------------------------------------------------------------
    # Spatial index construction (once per distinct TIN, then cached)
    # ------------------------------------------------------------------

    def _get_index(self, tin: TriangulatedSurface) -> _TriangleSpatialIndex:
        """
        Return the cached spatial index for this TIN, building it if
        this is a new (or first-seen) TIN instance.

        Notes
        -----
        The cache key is ``id(tin)``. This is a lightweight identity
        cache, not a content hash: if a TIN object is garbage
        collected and a new, unrelated object happens to reuse the
        same id, the cache would incorrectly hit. In practice TINs
        are long-lived objects held by the caller for the duration
        of an analysis, so this is an acceptable, documented
        trade-off in exchange for avoiding an expensive per-triangle
        hash of the mesh on every call.
        """
        tin_id = id(tin)

        if self._cache_index is None or self._cache_tin_id != tin_id:
            self._cache_index = self._build_index(tin)
            self._cache_tin_id = tin_id

        return self._cache_index

    @staticmethod
    def _build_index(
        tin: TriangulatedSurface,
    ) -> _TriangleSpatialIndex:
        """
        Build a uniform spatial grid over the mesh's triangles.

        This requires one Python-level pass over all triangles
        (``TriangulatedSurface`` does not expose a vectorized way to
        read them), but it happens once per mesh, not once per point.
        """
        triangle_count = tin.triangle_count

        v0 = np.empty((triangle_count, 3), dtype=np.float64)
        v1 = np.empty((triangle_count, 3), dtype=np.float64)
        v2 = np.empty((triangle_count, 3), dtype=np.float64)

        for j in range(triangle_count):
            p1, p2, p3 = tin.triangle_vertices(j)

            v0[j] = (p1.x, p1.y, p1.z)
            v1[j] = (p2.x, p2.y, p2.z)
            v2[j] = (p3.x, p3.y, p3.z)

        tri_min_x = np.minimum(
            np.minimum(v0[:, 0], v1[:, 0]),
            v2[:, 0],
        )
        tri_max_x = np.maximum(
            np.maximum(v0[:, 0], v1[:, 0]),
            v2[:, 0],
        )

        tri_min_y = np.minimum(
            np.minimum(v0[:, 1], v1[:, 1]),
            v2[:, 1],
        )
        tri_max_y = np.maximum(
            np.maximum(v0[:, 1], v1[:, 1]),
            v2[:, 1],
        )

        min_x = float(np.min(tri_min_x))
        min_y = float(np.min(tri_min_y))
        max_x = float(np.max(tri_max_x))
        max_y = float(np.max(tri_max_y))

        extent_x = max(max_x - min_x, 1e-9)
        extent_y = max(max_y - min_y, 1e-9)

        # Aim for roughly one triangle per cell on average, which
        # keeps ring expansion rare for reasonably uniform meshes.
        target_cells = max(triangle_count, 1)

        cell_size = max(
            math.sqrt((extent_x * extent_y) / target_cells),
            1e-6,
        )

        n_cols = max(math.ceil(extent_x / cell_size), 1)
        n_rows = max(math.ceil(extent_y / cell_size), 1)

        cell_min_x = np.clip(
            np.floor((tri_min_x - min_x) / cell_size).astype(np.int64),
            0,
            n_cols - 1,
        )
        cell_max_x = np.clip(
            np.floor((tri_max_x - min_x) / cell_size).astype(np.int64),
            0,
            n_cols - 1,
        )
        cell_min_y = np.clip(
            np.floor((tri_min_y - min_y) / cell_size).astype(np.int64),
            0,
            n_rows - 1,
        )
        cell_max_y = np.clip(
            np.floor((tri_max_y - min_y) / cell_size).astype(np.int64),
            0,
            n_rows - 1,
        )

        cells: dict[int, list[int]] = {}

        for j in range(triangle_count):
            for cy in range(
                int(cell_min_y[j]),
                int(cell_max_y[j]) + 1,
            ):
                row_offset = cy * n_cols

                for cx in range(
                    int(cell_min_x[j]),
                    int(cell_max_x[j]) + 1,
                ):
                    cells.setdefault(
                        row_offset + cx,
                        [],
                    ).append(j)

        cells_arr = {key: np.asarray(indices, dtype=np.int64) for key, indices in cells.items()}

        return _TriangleSpatialIndex(
            v0=v0,
            v1=v1,
            v2=v2,
            min_x=min_x,
            min_y=min_y,
            cell_size=cell_size,
            n_cols=n_cols,
            n_rows=n_rows,
            cells=cells_arr,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _query_distances(
        self,
        points: NDArray[np.float64],
        index: _TriangleSpatialIndex,
    ) -> NDArray[np.float64]:
        """
        Compute the nearest-triangle distance for every point.

        Points are grouped by grid cell so that each group is
        checked against candidate triangles with a single vectorized
        NumPy computation instead of a per-point Python loop.
        """
        n = points.shape[0]
        distances = np.full(n, np.inf, dtype=np.float64)

        col = np.clip(
            np.floor((points[:, 0] - index.min_x) / index.cell_size).astype(np.int64),
            0,
            index.n_cols - 1,
        )
        row = np.clip(
            np.floor((points[:, 1] - index.min_y) / index.cell_size).astype(np.int64),
            0,
            index.n_rows - 1,
        )
        cell_keys = row * index.n_cols + col

        order = np.argsort(cell_keys, kind="stable")
        sorted_keys = cell_keys[order]
        group_starts = np.flatnonzero(np.diff(sorted_keys, prepend=sorted_keys[0] - 1))
        group_starts = np.append(group_starts, n)

        max_radius = max(index.n_rows, index.n_cols)

        for i in range(len(group_starts) - 1):
            start, end = group_starts[i], group_starts[i + 1]
            point_idx = order[start:end]
            key = int(sorted_keys[start])
            cell_row, cell_col = divmod(key, index.n_cols)

            group_points = points[point_idx]
            group_best = self._nearest_distance_expanding_rings(
                group_points,
                cell_row,
                cell_col,
                index,
                max_radius,
            )

            distances[point_idx] = group_best

        return distances

    @staticmethod
    def _nearest_distance_expanding_rings(
        group_points: NDArray[np.float64],
        cell_row: int,
        cell_col: int,
        index: _TriangleSpatialIndex,
        max_radius: int,
    ) -> NDArray[np.float64]:
        """
        Find the nearest-triangle distance for a group of points that
        all fall in the same grid cell, expanding the search outward
        in rings only as far as needed.

        Correctness
        -----------
        After fully searching all cells within Chebyshev radius `r`
        of the query cell, any unexplored cell is at least
        ``r * cell_size`` away from a point located anywhere inside
        the query cell. So once the group's worst-case current best
        distance is within that bound, no closer triangle can exist
        outside the searched cells, and expansion can stop safely.
        """
        group_best = np.full(group_points.shape[0], np.inf, dtype=np.float64)
        radius = 0

        while True:
            ring = _ring_cells(cell_row, cell_col, radius, index.n_rows, index.n_cols)
            ring_triangle_groups = [
                arr for (r, c) in ring if (arr := index.cells.get(r * index.n_cols + c)) is not None and arr.size > 0
            ]

            if ring_triangle_groups:
                candidates = np.unique(np.concatenate(ring_triangle_groups))

                ring_distances = CloudToMeshDistance._batch_point_to_triangle_distance(
                    group_points,
                    index.v0[candidates],
                    index.v1[candidates],
                    index.v2[candidates],
                )
                group_best = np.minimum(group_best, ring_distances.min(axis=1))

            bound = radius * index.cell_size

            if np.isfinite(group_best).all() and group_best.max() <= bound:
                break

            radius += 1

            if radius > max_radius:
                break

        return group_best

    @staticmethod
    def _batch_point_to_triangle_distance(
        points: NDArray[np.float64],
        v0: NDArray[np.float64],
        v1: NDArray[np.float64],
        v2: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Vectorized point-to-triangle distance for M points against C
        triangles at once, returning an (M, C) distance matrix.

        This computes the same result as
        :meth:`_point_to_triangle_distance` applied to every
        (point, triangle) pair, but as batched NumPy operations
        instead of a Python-level double loop. Degenerate triangles
        (near-zero area) and degenerate edges are handled via
        ``np.where`` masks instead of Python branches.
        """
        edge0 = v1 - v0  # (C,3)
        edge1 = v2 - v0  # (C,3)
        normal = np.cross(edge0, edge1)  # (C,3)
        normal_norm = np.linalg.norm(normal, axis=1)  # (C,)
        degenerate = normal_norm < 1e-15

        normal_safe = normal / np.where(normal_norm[:, None] > 0, normal_norm[:, None], 1.0)

        diff = points[:, None, :] - v0[None, :, :]  # (M,C,3)
        signed = np.einsum(_EINSUM_POINT_CD, diff, normal_safe)  # (M,C)
        plane_dist = np.abs(signed)

        projection = points[:, None, :] - normal_safe[None, :, :] * signed[:, :, None]
        diff_proj = projection - v0[None, :, :]

        dot00 = np.einsum(_EINSUM_DOT_CD, edge0, edge0)
        dot01 = np.einsum(_EINSUM_DOT_CD, edge0, edge1)
        dot02 = np.einsum(_EINSUM_POINT_CD, diff_proj, edge0)
        dot11 = np.einsum(_EINSUM_DOT_CD, edge1, edge1)
        dot12 = np.einsum(_EINSUM_POINT_CD, diff_proj, edge1)

        denom = dot00 * dot11 - dot01 * dot01  # (C,)
        denom_degenerate = np.abs(denom) < 1e-15
        denom_safe = np.where(denom_degenerate, 1.0, denom)
        inv_denom = 1.0 / denom_safe

        u = (dot11[None, :] * dot02 - dot01[None, :] * dot12) * inv_denom[None, :]
        v = (dot00[None, :] * dot12 - dot01[None, :] * dot02) * inv_denom[None, :]

        inside = (u >= 0) & (v >= 0) & (u + v <= 1) & ~denom_degenerate[None, :]

        d1 = CloudToMeshDistance._batch_closest_on_segment(points, v0, v1)
        d2 = CloudToMeshDistance._batch_closest_on_segment(points, v1, v2)
        d3 = CloudToMeshDistance._batch_closest_on_segment(points, v2, v0)
        edge_min = np.minimum(np.minimum(d1, d2), d3)

        d_v0 = np.linalg.norm(points[:, None, :] - v0[None, :, :], axis=2)
        d_v1 = np.linalg.norm(points[:, None, :] - v1[None, :, :], axis=2)
        d_v2 = np.linalg.norm(points[:, None, :] - v2[None, :, :], axis=2)
        degenerate_dist = np.minimum(np.minimum(d_v0, d_v1), d_v2)

        return np.where(
            degenerate[None, :],
            degenerate_dist,
            np.where(inside, plane_dist, edge_min),
        )

    @staticmethod
    def _batch_closest_on_segment(
        points: NDArray[np.float64],
        a: NDArray[np.float64],
        b: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Distance from M points to the closest point on each of C
        segments AB, returning an (M, C) matrix.
        """

        ab: NDArray[np.float64] = b - a

        ap: NDArray[np.float64] = points[:, None, :] - a[None, :, :]

        denom: NDArray[np.float64] = np.einsum(
            "cd,cd->c",
            ab,
            ab,
        )

        denom_safe: NDArray[np.float64] = np.where(
            denom < 1e-15,
            1.0,
            denom,
        )

        t: NDArray[np.float64] = (
            np.einsum(
                "mcd,cd->mc",
                ap,
                ab,
            )
            / denom_safe[None, :]
        )

        t = np.clip(
            t,
            0.0,
            1.0,
        )

        closest: NDArray[np.float64] = a[None, :, :] + t[:, :, None] * ab[None, :, :]

        distances: NDArray[np.float64] = np.linalg.norm(
            points[:, None, :] - closest,
            axis=2,
        )

        return distances.astype(
            np.float64,
            copy=False,
        )

    # ------------------------------------------------------------------
    # Original scalar reference implementation
    #
    # Kept unchanged (algorithm and signature) as a documented,
    # independently-testable reference. `_batch_point_to_triangle_distance`
    # above computes the same result for a whole (M, C) batch at once
    # and is what `compute()` actually uses.
    # ------------------------------------------------------------------

    @staticmethod
    def _point_to_triangle_distance(
        point: NDArray[np.float64],
        v0: NDArray[np.float64],
        v1: NDArray[np.float64],
        v2: NDArray[np.float64],
    ) -> float:
        """
        Compute the shortest distance from a point to a triangle.

        Uses the algorithm from "Distance Between a Point and a
        Triangle in 3D" (David Eberly).
        """
        edge0 = v1 - v0
        edge1 = v2 - v0
        normal = np.cross(edge0, edge1)
        normal_norm = float(np.linalg.norm(normal))

        if normal_norm < 1e-15:
            # Degenerate triangle
            d1 = float(np.linalg.norm(point - v0))
            d2 = float(np.linalg.norm(point - v1))
            d3 = float(np.linalg.norm(point - v2))
            return min(d1, d2, d3)

        normal = normal / normal_norm

        # Signed distance to the triangle plane
        diff = point - v0
        plane_dist = abs(float(np.dot(diff, normal)))

        # Check if the projection falls inside the triangle
        # by testing barycentric coordinates
        projection = point - normal * np.dot(diff, normal)
        diff_proj = projection - v0

        dot00 = float(np.dot(edge0, edge0))
        dot01 = float(np.dot(edge0, edge1))
        dot02 = float(np.dot(edge0, diff_proj))
        dot11 = float(np.dot(edge1, edge1))
        dot12 = float(np.dot(edge1, diff_proj))

        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-15:
            # Degenerate case
            return plane_dist

        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        if u >= 0 and v >= 0 and u + v <= 1:
            # Projection is inside the triangle
            return plane_dist

        # Find the closest point on the triangle boundary
        closest1 = CloudToMeshDistance._closest_point_on_segment(
            point,
            v0,
            v1,
        )
        d1 = float(np.linalg.norm(point - closest1))

        closest2 = CloudToMeshDistance._closest_point_on_segment(
            point,
            v1,
            v2,
        )
        d2 = float(np.linalg.norm(point - closest2))

        closest3 = CloudToMeshDistance._closest_point_on_segment(
            point,
            v2,
            v0,
        )
        d3 = float(np.linalg.norm(point - closest3))

        return min(d1, d2, d3)

    @staticmethod
    def _closest_point_on_segment(
        point: NDArray[np.float64],
        a: NDArray[np.float64],
        b: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Find the closest point on segment AB to the given point."""
        ab = b - a
        ap = point - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-15:
            return a
        t = float(np.dot(ap, ab)) / denom
        t = float(np.clip(t, 0.0, 1.0))
        return a + t * ab


__all__ = ["CloudToMeshDistance"]
