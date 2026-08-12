"""
topocore.features.feature_builder
====================================

Turns a ``SurveyPointSet`` into ``Feature``/``FeatureCollection`` --
the same model PR15's automatic detectors produce, so a DXF/GPKG
exporter never needs to know whether a feature came from LiDAR or
from a total station.

Grouping rule (v1)
-------------------
Trailing digits are stripped from each point's code to get its
*base code* (``"CERCA1"`` -> ``"CERCA"``, ``"CERCA2"`` -> ``"CERCA"``).
Consecutive points, in survey order, sharing the same base code are
grouped into one run:

* ``POINT`` / ``SYMBOL`` codes -> every point in the run becomes its
  own independent feature (a run of "ARBOL" points is N separate
  trees, not one connected line of trees).
* ``LINE`` / ``POLYGON`` codes -> the run becomes one polyline/polygon,
  in survey order. A run with fewer than 2 points can't form a line
  and is reported as ``unmatched`` + a diagnostic instead of being
  silently dropped.
* ``GROUND`` codes -> the run's points feed TIN/DTM construction
  directly and never produce a `Feature` at all -- not reported as a
  problem either, since a ground shot is fully expected.

A code change breaks the run, even if the same code reappears later
in the file -- two separate "CERCA" fences surveyed at different
times become two separate line features, which is the correct
behavior for real fieldwork.

This is deliberately a simple, well-defined v1 rule, not a general
field-code grammar (no explicit start/end markers, no join-across-
gaps). Extending it is future work, not attempted here.

Determinism
-----------
``feature_id``s are assigned sequentially, in production order, only
to features that were actually built -- a GROUND run, an unregistered
code, or an insufficient-length line never consumes an id. The same
``SurveyPointSet`` + the same ``FeatureCodeRegistry`` + the same
``closure_tolerance`` therefore always produces exactly the same ids.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from topocore.features._code_utils import base_code as _base_code
from topocore.features.feature_codes import (
    CATALOG_TO_MODEL_GEOMETRY,
    FeatureCodeDefinition,
    FeatureCodeRegistry,
    FeatureGeometryType,
)
from topocore.features.grammar import AssembledDiagnosticReason, assemble
from topocore.features.models import (
    Feature,
    FeatureCollection,
    FeatureGeometry,
    FeatureMetadata,
    GeometryType,
)
from topocore.survey.models import SurveyPoint, SurveyPointSet

_POINT_LIKE = (FeatureGeometryType.POINT, FeatureGeometryType.SYMBOL)

_PRODUCER_NAME = "feature_builder"


class BuildDiagnosticReason(StrEnum):
    """Why a run of survey points didn't become a Feature."""

    MISSING_CODE = "missing_code"
    UNREGISTERED_CODE = "unregistered_code"
    INSUFFICIENT_POINTS = "insufficient_points"

    #: Grammar-mode only (see `topocore.features.grammar`) -- a
    #: figure-syntax code couldn't be parsed, or its START/CONTINUE/
    #: END/CLOSE sequence was invalid.
    MALFORMED_GRAMMAR_CODE = "malformed_grammar_code"
    CONTINUE_WITHOUT_START = "continue_without_start"
    END_WITHOUT_START = "end_without_start"
    DUPLICATE_START = "duplicate_start"
    UNCLOSED_FIGURE = "unclosed_figure"


_GRAMMAR_REASON_MAP: dict[AssembledDiagnosticReason, BuildDiagnosticReason] = {
    AssembledDiagnosticReason.MALFORMED_CODE: BuildDiagnosticReason.MALFORMED_GRAMMAR_CODE,
    AssembledDiagnosticReason.CONTINUE_WITHOUT_START: BuildDiagnosticReason.CONTINUE_WITHOUT_START,
    AssembledDiagnosticReason.END_WITHOUT_START: BuildDiagnosticReason.END_WITHOUT_START,
    AssembledDiagnosticReason.DUPLICATE_START: BuildDiagnosticReason.DUPLICATE_START,
    AssembledDiagnosticReason.UNCLOSED_FIGURE: BuildDiagnosticReason.UNCLOSED_FIGURE,
}


@dataclass(frozen=True, slots=True)
class BuildDiagnostic:
    """
    Explains why one run of points didn't produce a Feature.

    Parameters
    ----------
    code
        Base code of the run, or ``None`` for uncoded points
        (``BuildDiagnosticReason.MISSING_CODE``).
    reason
        Why the run didn't become a Feature.
    point_count
        Number of points in the run.
    point_ids
        ``SurveyPoint.id`` of every point in the run, in survey order.
    """

    code: str | None
    reason: BuildDiagnosticReason
    point_count: int
    point_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FeatureBuildResult:
    """
    Result of building Features from a ``SurveyPointSet``.

    Parameters
    ----------
    features
        Every Feature successfully built, with normalized ``1..N``
        ``feature_id``s (see ``FeatureCollection.normalize_ids``).
    ground
        Points whose code is recognized as
        ``FeatureGeometryType.GROUND`` -- never a problem, these feed
        TIN/DTM construction directly.
    unmatched
        Every point that didn't end up in ``features`` or ``ground``,
        flat list, in survey order. Never silently dropped.
    diagnostics
        Structured explanation of *why* each run in ``unmatched``
        didn't produce a Feature. ``unmatched`` and ``diagnostics``
        coexist deliberately: ``unmatched`` answers "which points",
        ``diagnostics`` answers "why".
    """

    features: FeatureCollection
    ground: tuple[SurveyPoint, ...]
    unmatched: tuple[SurveyPoint, ...]
    diagnostics: tuple[BuildDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class _PointFeature:
    """Internal: a single point-like run (POINT or SYMBOL code)."""

    code: str
    definition: FeatureCodeDefinition
    point: SurveyPoint


@dataclass(frozen=True, slots=True)
class _LineFeature:
    """Internal: a polyline/polygon built from a run of points."""

    code: str
    definition: FeatureCodeDefinition
    points: tuple[SurveyPoint, ...]
    closed: bool


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


def _distance_xy(a: SurveyPoint, b: SurveyPoint) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _is_closed_run(
    definition: FeatureCodeDefinition,
    run: Sequence[SurveyPoint],
    closure_tolerance: float,
) -> bool:
    if definition.closed:
        return True

    first, last = run[0], run[-1]
    return _distance_xy(first, last) <= closure_tolerance


def _is_closed_figure(
    definition: FeatureCodeDefinition,
    points: Sequence[SurveyPoint],
    *,
    explicit_close: bool,
    closure_tolerance: float,
) -> bool:
    """
    Same as `_is_closed_run`, plus the grammar-mode CLOSE (X) command
    taking priority over everything else: an explicit instruction
    from the survey overrides both `definition.closed` and geometric
    inference via `closure_tolerance`.
    """
    if explicit_close:
        return True
    return _is_closed_run(definition, points, closure_tolerance)


def _build_attributes(definition: FeatureCodeDefinition, run: Sequence[SurveyPoint]) -> dict[str, object]:
    return {
        "survey_code": definition.code,
        "survey_name": definition.name,
        "cad_layer": definition.layer,
        "survey_point_ids": tuple(p.id for p in run),
    }


def _point_feature_to_feature(pf: _PointFeature, feature_id: int) -> Feature:
    assert pf.definition.feature_type is not None  # POINT/SYMBOL is never GROUND

    vertices = np.array([[pf.point.x, pf.point.y, pf.point.z]], dtype=np.float64)
    geometry = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=vertices)

    return Feature(
        feature_id=feature_id,
        category=pf.definition.category,
        feature_type=pf.definition.feature_type,
        geometry=geometry,
        metadata=FeatureMetadata(detector=_PRODUCER_NAME, extra={"source": "field_survey"}),
        attributes=_build_attributes(pf.definition, (pf.point,)),
    )


def _line_feature_to_feature(lf: _LineFeature, feature_id: int) -> Feature:
    assert lf.definition.feature_type is not None  # LINE/POLYGON is never GROUND

    vertices = np.array([[p.x, p.y, p.z] for p in lf.points], dtype=np.float64)
    model_geometry_type = CATALOG_TO_MODEL_GEOMETRY[lf.definition.geometry_type]
    geometry = FeatureGeometry(geometry_type=model_geometry_type, vertices=vertices, closed=lf.closed)

    return Feature(
        feature_id=feature_id,
        category=lf.definition.category,
        feature_type=lf.definition.feature_type,
        geometry=geometry,
        metadata=FeatureMetadata(detector=_PRODUCER_NAME, extra={"source": "field_survey"}),
        attributes=_build_attributes(lf.definition, lf.points),
    )


def _build_legacy(
    points: SurveyPointSet,
    registry: FeatureCodeRegistry,
    *,
    closure_tolerance: float = 0.0,
) -> FeatureBuildResult:
    """The default, unchanged v1 grouping rule -- see module docstring."""
    ground: list[SurveyPoint] = []
    unmatched: list[SurveyPoint] = []
    diagnostics: list[BuildDiagnostic] = []
    collection = FeatureCollection()
    next_id = 1

    for base_code, run in _group_runs(points.points):
        if base_code is None:
            unmatched.extend(run)
            diagnostics.append(
                BuildDiagnostic(
                    code=None,
                    reason=BuildDiagnosticReason.MISSING_CODE,
                    point_count=len(run),
                    point_ids=tuple(p.id for p in run),
                )
            )
            continue

        definition = registry.get(base_code)

        if definition is None:
            unmatched.extend(run)
            diagnostics.append(
                BuildDiagnostic(
                    code=base_code,
                    reason=BuildDiagnosticReason.UNREGISTERED_CODE,
                    point_count=len(run),
                    point_ids=tuple(p.id for p in run),
                )
            )
            continue

        if definition.geometry_type is FeatureGeometryType.GROUND:
            ground.extend(run)
            continue

        if definition.geometry_type in _POINT_LIKE:
            for point in run:
                point_feature = _PointFeature(code=base_code, definition=definition, point=point)
                collection.add(_point_feature_to_feature(point_feature, next_id))
                next_id += 1
            continue

        # LINE / POLYGON
        if len(run) < 2:
            unmatched.extend(run)
            diagnostics.append(
                BuildDiagnostic(
                    code=base_code,
                    reason=BuildDiagnosticReason.INSUFFICIENT_POINTS,
                    point_count=len(run),
                    point_ids=tuple(p.id for p in run),
                )
            )
            continue

        closed = _is_closed_run(definition, run, closure_tolerance)
        line_feature = _LineFeature(code=base_code, definition=definition, points=tuple(run), closed=closed)
        collection.add(_line_feature_to_feature(line_feature, next_id))
        next_id += 1

    return FeatureBuildResult(
        features=collection,
        ground=tuple(ground),
        unmatched=tuple(unmatched),
        diagnostics=tuple(diagnostics),
    )


def _build_with_grammar(
    points: SurveyPointSet,
    registry: FeatureCodeRegistry,
    *,
    closure_tolerance: float = 0.0,
) -> FeatureBuildResult:
    """
    Opt-in grammar mode: assembles figures via
    `topocore.features.grammar.assemble` (identity ``(base_code,
    figure_id)``, simultaneous open figures, explicit CLOSE) instead
    of `_group_runs`'s consecutive-only rule. Codes without the
    grammar separator still fall through to the exact same
    consecutive-run grouping as legacy mode -- see `assemble`'s
    docstring.
    """
    assembled = assemble(points.points)

    ground: list[SurveyPoint] = []
    unmatched: list[SurveyPoint] = list(assembled.unmatched)
    diagnostics: list[BuildDiagnostic] = [
        BuildDiagnostic(
            code=(f"{d.base_code}.{d.figure_id}" if d.figure_id is not None else d.base_code),
            reason=_GRAMMAR_REASON_MAP[d.reason],
            point_count=len(d.point_ids),
            point_ids=d.point_ids,
        )
        for d in assembled.diagnostics
    ]
    collection = FeatureCollection()
    next_id = 1

    for figure in assembled.figures:
        definition = registry.get(figure.base_code)

        if definition is None:
            unmatched.extend(figure.points)
            diagnostics.append(
                BuildDiagnostic(
                    code=figure.base_code,
                    reason=BuildDiagnosticReason.UNREGISTERED_CODE,
                    point_count=len(figure.points),
                    point_ids=tuple(p.id for p in figure.points),
                )
            )
            continue

        if definition.geometry_type is FeatureGeometryType.GROUND:
            # A figure whose code resolves to GROUND has no
            # meaningful figure structure -- same treatment as
            # legacy GROUND handling: every point feeds TIN/DTM
            # directly, command structure is simply irrelevant here.
            ground.extend(figure.points)
            continue

        if definition.geometry_type in _POINT_LIKE:
            for point in figure.points:
                point_feature = _PointFeature(code=figure.base_code, definition=definition, point=point)
                collection.add(_point_feature_to_feature(point_feature, next_id))
                next_id += 1
            continue

        # LINE / POLYGON
        if len(figure.points) < 2:
            unmatched.extend(figure.points)
            diagnostics.append(
                BuildDiagnostic(
                    code=figure.base_code,
                    reason=BuildDiagnosticReason.INSUFFICIENT_POINTS,
                    point_count=len(figure.points),
                    point_ids=tuple(p.id for p in figure.points),
                )
            )
            continue

        closed = _is_closed_figure(
            definition,
            figure.points,
            explicit_close=figure.explicit_close,
            closure_tolerance=closure_tolerance,
        )
        line_feature = _LineFeature(
            code=figure.base_code,
            definition=definition,
            points=figure.points,
            closed=closed,
        )
        collection.add(_line_feature_to_feature(line_feature, next_id))
        next_id += 1

    return FeatureBuildResult(
        features=collection,
        ground=tuple(ground),
        unmatched=tuple(unmatched),
        diagnostics=tuple(diagnostics),
    )


def build_features(
    points: SurveyPointSet,
    registry: FeatureCodeRegistry,
    *,
    closure_tolerance: float = 0.0,
    use_field_code_grammar: bool = False,
) -> FeatureBuildResult:
    """
    Build a ``FeatureBuildResult`` from a ``SurveyPointSet``.

    Parameters
    ----------
    points
        Points in survey order.
    registry
        Resolves base codes to ``FeatureCodeDefinition``.
    closure_tolerance
        Max XY distance (meters) between a run's first and last point
        for it to be considered a closed polygon/polyline, when
        ``definition.closed`` doesn't already force it (and, in
        grammar mode, when the figure wasn't closed via the explicit
        CLOSE/X command either). Must be non-negative.
    use_field_code_grammar
        If ``False`` (default), uses the v1 consecutive-run grouping
        described in the module docstring -- completely unchanged
        behavior. If ``True``, opts into
        ``topocore.features.grammar``: figures identified by
        ``BASE.FIGURE[.S|E|X]`` syntax can be assembled even when
        interleaved with other figures in the file; codes without
        the grammar separator still use the same consecutive-run
        rule as legacy mode.
    """
    if use_field_code_grammar:
        return _build_with_grammar(points, registry, closure_tolerance=closure_tolerance)
    return _build_legacy(points, registry, closure_tolerance=closure_tolerance)


class FeatureBuilder:
    """
    Builds Features from survey points using a fixed
    ``FeatureCodeRegistry`` and closure tolerance.
    """

    __slots__ = ("_closure_tolerance", "_registry", "_use_field_code_grammar")

    def __init__(
        self,
        registry: FeatureCodeRegistry | None = None,
        *,
        closure_tolerance: float = 0.0,
        use_field_code_grammar: bool = False,
    ) -> None:
        if closure_tolerance < 0:
            raise ValueError(f"closure_tolerance must be non-negative; got {closure_tolerance}.")

        self._registry = registry or FeatureCodeRegistry.default()
        self._closure_tolerance = closure_tolerance
        self._use_field_code_grammar = use_field_code_grammar

    @property
    def registry(self) -> FeatureCodeRegistry:
        return self._registry

    @property
    def closure_tolerance(self) -> float:
        return self._closure_tolerance

    @property
    def use_field_code_grammar(self) -> bool:
        return self._use_field_code_grammar

    def build(self, points: SurveyPointSet) -> FeatureBuildResult:
        return build_features(
            points,
            self._registry,
            closure_tolerance=self._closure_tolerance,
            use_field_code_grammar=self._use_field_code_grammar,
        )

    def __call__(self, points: SurveyPointSet) -> FeatureBuildResult:
        return self.build(points)


__all__ = [
    "BuildDiagnostic",
    "BuildDiagnosticReason",
    "FeatureBuildResult",
    "FeatureBuilder",
    "build_features",
]
