"""
topocore.terrain.breaklines
==============================

Enforce breaklines as constraints on a TIN.

This is the public entry point for the constrained triangulation
backend in ``terrain.algorithms.constrained_delaunay``. It handles
the bookkeeping a raw CDT call doesn't know about: merging breakline
vertices into the existing point set (without duplicating points that
already exist in the TIN), building the constraint-edge list, and
honoring the distinction between ``BreaklineType.HARD`` (the edges
themselves are forced into the triangulation -- retaining walls,
curbs, grade breaks) and ``BreaklineType.SOFT`` (only the vertices
are added, so elevation is captured precisely, but the natural
Delaunay connectivity around them is left alone).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Sequence

from topocore.geometry.point3d import Point3D
from topocore.terrain.algorithms.constrained_delaunay import (
    ConstrainedDelaunayTriangulator,
)
from topocore.terrain.enums import BreaklineType
from topocore.terrain.exceptions import BreaklineError
from topocore.terrain.models import Breakline
from topocore.terrain.tin import TIN

_XYKey = tuple[float, float]


def _merge_points(
    tin: TIN,
    breaklines: Sequence[Breakline],
) -> tuple[list[Point3D], dict[_XYKey, int]]:
    """
    Combine the TIN's vertices with every breakline vertex.

    A breakline point whose XY coincides with an existing point (be
    it an original TIN vertex or one already added by an earlier
    breakline) is not duplicated -- the earlier point's Z is kept,
    and the breakline reuses its index.
    """
    points = list(tin.vertices)
    index_by_xy: dict[_XYKey, int] = {(point.x, point.y): index for index, point in enumerate(points)}

    for breakline in breaklines:
        for point in breakline.points:
            key = (point.x, point.y)

            if key not in index_by_xy:
                index_by_xy[key] = len(points)
                points.append(point)

    return points, index_by_xy


def _validate_breakline(breakline: Breakline) -> None:
    if breakline.vertex_count < 2:
        raise BreaklineError(f"Breakline '{breakline.id}' needs at least 2 points, got {breakline.vertex_count}.")


def _build_constraints(
    breaklines: Sequence[Breakline],
    index_by_xy: dict[_XYKey, int],
) -> list[tuple[int, int]]:
    """
    Build the constraint-edge list for every HARD breakline.

    SOFT breaklines contribute their points to the triangulation (via
    ``_merge_points``) but not constraint edges: their vertices are
    captured for elevation accuracy, without forcing a rigid edge
    that would override the natural Delaunay connectivity around
    them.
    """
    constraints: list[tuple[int, int]] = []

    for breakline in breaklines:
        _validate_breakline(breakline)

        if breakline.type != BreaklineType.HARD:
            continue

        indices = [index_by_xy[(point.x, point.y)] for point in breakline.points]

        for a, b in zip(indices[:-1], indices[1:]):
            constraints.append((a, b))

    return constraints


def apply_breaklines(
    tin: TIN,
    breaklines: Sequence[Breakline],
) -> TIN:
    """
    Return a new TIN with the given breaklines enforced.

    Parameters
    ----------
    tin
        Source TIN.
    breaklines
        Breaklines to enforce. HARD breaklines force their edges into
        the triangulation; SOFT breaklines only add their vertices.

    Returns
    -------
    TIN
        A new TIN. ``tin`` itself is left unmodified.

    Raises
    ------
    BreaklineError
        If a breakline has fewer than 2 points, if two HARD
        breaklines cross each other, or if a breakline cannot be
        recovered (see ``ConstrainedDelaunayTriangulator``).
    """
    if not breaklines:
        return tin

    points, index_by_xy = _merge_points(tin, breaklines)
    constraints = _build_constraints(breaklines, index_by_xy)

    result = ConstrainedDelaunayTriangulator.triangulate(points, constraints)

    return TIN(result)


class BreaklineEnforcer:
    """
    Applies a fixed set of breaklines to a TIN.
    """

    __slots__ = (
        "_tin",
        "_breaklines",
    )

    def __init__(
        self,
        tin: TIN,
        breaklines: Sequence[Breakline],
    ) -> None:
        self._tin = tin
        self._breaklines = tuple(breaklines)

    @property
    def tin(self) -> TIN:
        return self._tin

    @property
    def breaklines(self) -> tuple[Breakline, ...]:
        return self._breaklines

    def apply(self) -> TIN:
        """
        Return a new TIN with the configured breaklines enforced.
        """
        return apply_breaklines(self._tin, self._breaklines)

    def __call__(self) -> TIN:
        return self.apply()


__all__ = [
    "BreaklineEnforcer",
    "apply_breaklines",
]
