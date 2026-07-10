"""
topocore.terrain.filters
==========================

Vertex-level filters for a Triangulated Irregular Network (TIN).

Two independent, composable operations:

- **Laplacian smoothing**: reduces high-frequency noise by nudging
  each vertex's elevation toward the average of its triangulation
  neighbors. Only Z is modified; X/Y (planimetry) are left untouched,
  so surveyed horizontal accuracy is never altered by smoothing.
- **Spike detection / removal**: flags vertices whose elevation
  deviates from their neighbors' average by more than a threshold --
  typical of LiDAR blunders or misclassified points -- without
  touching any other vertex.

Both operate on the same vertex-adjacency graph derived from
``tin.simplices``, built once per call.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from topocore.geometry.point3d import Point3D
from topocore.terrain.exceptions import TerrainError
from topocore.terrain.exceptions import TerrainValidationError
from topocore.terrain.tin import TIN


def _vertex_adjacency(tin: TIN) -> list[set[int]]:
    """
    Build a vertex-index adjacency graph from the TIN triangulation.

    Returns
    -------
    list
        One set of neighbor indices per vertex, indexed the same
        way as ``tin.vertices``.
    """
    adjacency: list[set[int]] = [set() for _ in range(tin.vertex_count)]

    for simplex in tin.simplices:
        i0 = int(simplex[0])
        i1 = int(simplex[1])
        i2 = int(simplex[2])

        adjacency[i0].update((i1, i2))
        adjacency[i1].update((i0, i2))
        adjacency[i2].update((i0, i1))

    return adjacency


def _neighbor_mean_z(
    vertices: tuple[Point3D, ...],
    neighbors: set[int],
) -> float:
    return sum(vertices[index].z for index in neighbors) / len(neighbors)


def _validate_iterations(iterations: int) -> None:
    if iterations < 1:
        raise TerrainValidationError(
            f"iterations must be >= 1, got {iterations}."
        )


def _validate_factor(factor: float) -> None:
    if not 0.0 < factor <= 1.0:
        raise TerrainValidationError(
            f"factor must be in the range (0, 1], got {factor}."
        )


def _validate_threshold(threshold: float) -> None:
    if threshold <= 0.0:
        raise TerrainValidationError(
            f"threshold must be greater than zero, got {threshold}."
        )


# =====================================================================
# Laplacian smoothing
# =====================================================================


def _smooth_pass(
    vertices: tuple[Point3D, ...],
    adjacency: list[set[int]],
    factor: float,
) -> tuple[Point3D, ...]:
    """
    Apply a single Laplacian smoothing pass over every vertex.
    """
    smoothed: list[Point3D] = []

    for index, point in enumerate(vertices):
        neighbors = adjacency[index]

        if not neighbors:
            smoothed.append(point)
            continue

        mean_z = _neighbor_mean_z(vertices, neighbors)
        new_z = point.z + factor * (mean_z - point.z)

        smoothed.append(Point3D(point.x, point.y, new_z))

    return tuple(smoothed)


def laplacian_smooth(
    tin: TIN,
    *,
    iterations: int = 1,
    factor: float = 0.5,
) -> TIN:
    """
    Smooth terrain elevations toward their triangulation neighbors.

    Parameters
    ----------
    iterations
        Number of smoothing passes. Higher values smooth more
        aggressively (and flatten more real terrain detail).
    factor
        Blend factor per pass, in (0, 1]. ``1.0`` snaps a vertex
        directly to its neighbor average; lower values blend
        partially, which is gentler and less prone to overshoot
        near sharp terrain features.

    Returns
    -------
    TIN
        A new TIN with smoothed elevations. X/Y are unchanged, so
        this re-triangulates over identical planimetry -- the
        resulting connectivity is the same as the input TIN's.

    Raises
    ------
    TerrainValidationError
        If ``iterations`` or ``factor`` are out of range.
    """
    _validate_iterations(iterations)
    _validate_factor(factor)

    adjacency = _vertex_adjacency(tin)
    vertices = tin.vertices

    for _ in range(iterations):
        vertices = _smooth_pass(vertices, adjacency, factor)

    return TIN.from_points(vertices)


class LaplacianSmoother:
    """
    Applies Laplacian elevation smoothing to a TIN.
    """

    __slots__ = (
        "_tin",
        "_iterations",
        "_factor",
    )

    def __init__(
        self,
        tin: TIN,
        *,
        iterations: int = 1,
        factor: float = 0.5,
    ) -> None:
        _validate_iterations(iterations)
        _validate_factor(factor)

        self._tin = tin
        self._iterations = iterations
        self._factor = factor

    @property
    def tin(self) -> TIN:
        return self._tin

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def factor(self) -> float:
        return self._factor

    def apply(self) -> TIN:
        """
        Run the configured smoothing and return a new TIN.
        """
        return laplacian_smooth(
            self._tin,
            iterations=self._iterations,
            factor=self._factor,
        )

    def __call__(self) -> TIN:
        return self.apply()


# =====================================================================
# Spike detection / removal
# =====================================================================


def _spike_mask(
    vertices: tuple[Point3D, ...],
    adjacency: list[set[int]],
    threshold: float,
) -> NDArray[np.bool_]:
    mask = np.zeros(len(vertices), dtype=np.bool_)

    for index, point in enumerate(vertices):
        neighbors = adjacency[index]

        if not neighbors:
            continue

        mean_z = _neighbor_mean_z(vertices, neighbors)
        mask[index] = abs(point.z - mean_z) > threshold

    return mask


def detect_spikes(
    tin: TIN,
    *,
    threshold: float,
) -> NDArray[np.bool_]:
    """
    Flag vertices whose elevation deviates from their neighbors.

    Parameters
    ----------
    threshold
        Maximum allowed absolute difference, in Z units, between a
        vertex and the average elevation of its triangulation
        neighbors. There is no universal default: pick a value
        based on the expected noise level of the source survey
        (LiDAR blunders vs. total-station precision differ by
        orders of magnitude).

    Returns
    -------
    ndarray
        Boolean mask, one entry per vertex, in the same order as
        ``tin.vertices``. ``True`` marks a flagged vertex.

    Raises
    ------
    TerrainValidationError
        If ``threshold`` is not greater than zero.
    """
    _validate_threshold(threshold)

    adjacency = _vertex_adjacency(tin)

    return _spike_mask(tin.vertices, adjacency, threshold)


def remove_spikes(
    tin: TIN,
    *,
    threshold: float,
) -> TIN:
    """
    Remove flagged spike vertices and return a new TIN.

    Unlike ``laplacian_smooth``, this does not move any surviving
    vertex -- it only removes the ones flagged by ``detect_spikes``.

    Raises
    ------
    TerrainError
        If removing the flagged vertices would leave fewer than 3
        vertices to triangulate.
    """
    mask = detect_spikes(tin, threshold=threshold)

    kept = tuple(
        point
        for point, flagged in zip(tin.vertices, mask)
        if not flagged
    )

    if len(kept) < 3:
        raise TerrainError(
            "Spike removal would leave fewer than 3 vertices; "
            "raise the threshold."
        )

    return TIN.from_points(kept)


class SpikeDetector:
    """
    Detects and removes elevation spikes from a TIN.
    """

    __slots__ = (
        "_tin",
        "_threshold",
    )

    def __init__(
        self,
        tin: TIN,
        *,
        threshold: float,
    ) -> None:
        _validate_threshold(threshold)

        self._tin = tin
        self._threshold = threshold

    @property
    def tin(self) -> TIN:
        return self._tin

    @property
    def threshold(self) -> float:
        return self._threshold

    def detect(self) -> NDArray[np.bool_]:
        """
        Return the spike mask, without modifying the TIN.
        """
        return detect_spikes(self._tin, threshold=self._threshold)

    def remove(self) -> TIN:
        """
        Return a new TIN with flagged vertices removed.
        """
        return remove_spikes(self._tin, threshold=self._threshold)

    def __call__(self) -> NDArray[np.bool_]:
        return self.detect()


__all__ = [
    "LaplacianSmoother",
    "SpikeDetector",
    "laplacian_smooth",
    "detect_spikes",
    "remove_spikes",
]
