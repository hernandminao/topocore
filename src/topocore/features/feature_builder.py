"""
topocore.features.feature_builder
====================================

Turns a ``SurveyPointSet`` into geometry, driven by field codes.

Grouping rule (v1)
-------------------
Trailing digits are stripped from each point's code to get its
*base code* (``"CERCA1"`` -> ``"CERCA"``, ``"CERCA2"`` -> ``"CERCA"``).
Consecutive points, in survey order, sharing the same base code are
grouped into one run:

* ``POINT`` / ``SYMBOL`` codes -> every point in the run becomes its
  own independent feature (a run of "ARBOL" points is N separate
  trees, not one connected line of trees).
* ``LINE`` / ``POLYGON`` codes -> the run becomes one polyline, in
  survey order. A run with fewer than 2 points can't form a line and
  is reported in ``FeatureSet.unmatched`` instead of silently
  dropped.

A code change breaks the run, even if the same code reappears later
in the file -- two separate "CERCA" fences surveyed at different
times become two separate line features, which is the correct
behavior for real fieldwork.

This is deliberately a simple, well-defined v1 rule, not a general
field-code grammar (no explicit start/end markers, no join-across-
gaps). Extending it is future work, not attempted here.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from topocore.features.feature_codes import (
    FeatureCodeDefinition,
    FeatureCodeRegistry,
    FeatureGeometryType,
)
from topocore.survey.models import SurveyPoint, SurveyPointSet

_POINT_LIKE = (FeatureGeometryType.POINT, FeatureGeometryType.SYMBOL)
_LINE_LIKE = (FeatureGeometryType.LINE, FeatureGeometryType.POLYGON)


@dataclass(frozen=True, slots=True)
class PointFeature:
    """
    A single point-like feature (POINT or SYMBOL code).
    """

    code: str
    definition: FeatureCodeDefinition
    point: SurveyPoint


@dataclass(frozen=True, slots=True)
class LineFeature:
    """
    A polyline built from a run of consecutive same-code points.
    """

    code: str
    definition: FeatureCodeDefinition
    points: tuple[SurveyPoint, ...]
    closed: bool


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """
    The geometries built from a ``SurveyPointSet``.
    """

    points: tuple[PointFeature, ...]
    lines: tuple[LineFeature, ...]
    unmatched: tuple[SurveyPoint, ...]


def _base_code(code: str) -> str:
    stripped = code.rstrip("0123456789")
    return stripped if stripped else code


def _group_runs(
    points: Sequence[SurveyPoint],
) -> list[tuple[str | None, list[SurveyPoint]]]:
    """
    Group consecutive points sharing the same base code.

    An uncoded point always starts (and is) its own single-point run.
    """
    runs: list[tuple[str | None, list[SurveyPoint]]] = []

    for point in points:
        base = _base_code(point.code) if point.code is not None else None

        if base is not None and runs and runs[-1][0] == base:
            runs[-1][1].append(point)
        else:
            runs.append((base, [point]))

    return runs


def _is_closed_run(
    definition: FeatureCodeDefinition,
    run: Sequence[SurveyPoint],
) -> bool:
    if definition.closed:
        return True

    first, last = run[0], run[-1]

    return first.x == last.x and first.y == last.y


def _build_line_or_polygon(
    base_code: str,
    definition: FeatureCodeDefinition,
    run: list[SurveyPoint],
) -> LineFeature | None:
    if len(run) < 2:
        return None

    return LineFeature(
        code=base_code,
        definition=definition,
        points=tuple(run),
        closed=_is_closed_run(definition, run),
    )


def build_features(
    points: SurveyPointSet,
    registry: FeatureCodeRegistry,
) -> FeatureSet:
    """
    Build point and line features from a ``SurveyPointSet``.

    Points whose code is unregistered, or whose run is too short to
    form a line, end up in ``FeatureSet.unmatched`` -- never silently
    dropped.
    """
    point_features: list[PointFeature] = []
    line_features: list[LineFeature] = []
    unmatched: list[SurveyPoint] = []

    for base_code, run in _group_runs(points.points):
        if base_code is None:
            unmatched.extend(run)
            continue

        definition = registry.get(base_code)

        if definition is None:
            unmatched.extend(run)
            continue

        if definition.geometry_type in _POINT_LIKE:
            point_features.extend(
                PointFeature(code=base_code, definition=definition, point=p) for p in run
            )
            continue

        line = _build_line_or_polygon(base_code, definition, run)

        if line is None:
            unmatched.extend(run)
            continue

        line_features.append(line)

    return FeatureSet(
        points=tuple(point_features),
        lines=tuple(line_features),
        unmatched=tuple(unmatched),
    )


class FeatureBuilder:
    """
    Builds a ``FeatureSet`` from survey points using a fixed
    ``FeatureCodeRegistry``.
    """

    __slots__ = ("_registry",)

    def __init__(
        self,
        registry: FeatureCodeRegistry | None = None,
    ) -> None:
        self._registry = registry or FeatureCodeRegistry.default()

    @property
    def registry(self) -> FeatureCodeRegistry:
        return self._registry

    def build(self, points: SurveyPointSet) -> FeatureSet:
        return build_features(points, self._registry)

    def __call__(self, points: SurveyPointSet) -> FeatureSet:
        return self.build(points)


__all__ = [
    "PointFeature",
    "LineFeature",
    "FeatureSet",
    "FeatureBuilder",
    "build_features",
]
