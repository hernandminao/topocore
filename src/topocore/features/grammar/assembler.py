"""
topocore.features.grammar.assembler
=======================================

Assembles a stream of `SurveyPoint`s into complete figures, tracking
multiple simultaneously-open figures by identity
``(base_code, figure_id)`` -- never `figure_id` alone, since
``MURO.1`` and ``CERCA.1`` must never be confused with each other.

Points using the legacy convention (no separator in the code) are
NOT rejected here -- they fall through to the exact same
consecutive-run grouping `feature_builder._group_runs` already uses,
so a single survey file can freely mix ``ARBOL`` (legacy point code,
no figure needed) with ``MURO.1.S`` (explicit figure) without any
special configuration. This module has no dependency on
`feature_builder.py` (avoiding a circular import, since
`feature_builder.py` depends on this package for its opt-in grammar
mode).

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
from enum import StrEnum

from topocore.features._code_utils import base_code as legacy_base_code
from topocore.features.grammar.commands import FieldCodeCommand
from topocore.features.grammar.exceptions import FieldCodeGrammarError
from topocore.features.grammar.parser import FieldCodeParser, is_grammar_code
from topocore.survey.models import SurveyPoint


class AssembledDiagnosticReason(StrEnum):
    """Why one grammar-syntax point didn't join a figure."""

    #: The code itself couldn't be parsed (bad segment count, unknown
    #: command token, empty base/figure segment).
    MALFORMED_CODE = "malformed_code"

    #: CONTINUE (or END/CLOSE) for a (base_code, figure_id) that was
    #: never opened with START.
    CONTINUE_WITHOUT_START = "continue_without_start"
    END_WITHOUT_START = "end_without_start"

    #: START for a (base_code, figure_id) that's already open.
    DUPLICATE_START = "duplicate_start"

    #: A figure was opened with START but never closed (no END/CLOSE
    #: before the point stream ended).
    UNCLOSED_FIGURE = "unclosed_figure"


@dataclass(frozen=True, slots=True)
class AssembledDiagnostic:
    """
    Parameters
    ----------
    base_code
        For every reason except MALFORMED_CODE, the parsed base
        code. For MALFORMED_CODE, the raw unparsed code string
        (parsing failed, so there is no base/figure to report).
    figure_id
        The figure identifier, or ``None`` for MALFORMED_CODE.
    """

    base_code: str
    figure_id: str | None
    reason: AssembledDiagnosticReason
    point_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssembledFigure:
    """
    One completed figure or legacy run, ready for geometry
    construction -- deliberately doesn't know about `FeatureType`,
    `FeatureCodeDefinition`, or geometry; that's `FeatureBuilder`'s
    job, using `base_code` to look up the definition exactly like
    the legacy path does.
    """

    base_code: str
    points: tuple[SurveyPoint, ...]

    #: True only if this figure was closed via the explicit CLOSE (X)
    #: command -- legacy runs and grammar figures closed via END (E)
    #: are always False here; `definition.closed`/`closure_tolerance`
    #: still apply downstream when this is False.
    explicit_close: bool = False


@dataclass(frozen=True, slots=True)
class AssembleResult:
    figures: tuple[AssembledFigure, ...]
    unmatched: tuple[SurveyPoint, ...]
    diagnostics: tuple[AssembledDiagnostic, ...]


def assemble(points: Sequence[SurveyPoint]) -> AssembleResult:
    """
    Process `points` in order, producing completed figures.

    A code with no separator falls through to legacy consecutive-run
    grouping. A code with a separator is parsed as grammar syntax;
    multiple such figures can be open simultaneously, identified by
    ``(base_code, figure_id)``, so interleaved figures (e.g. two
    walls surveyed alternately) assemble correctly regardless of
    point order in the file.
    """
    open_figures: dict[tuple[str, str], list[SurveyPoint]] = {}
    completed: list[AssembledFigure] = []
    unmatched: list[SurveyPoint] = []
    diagnostics: list[AssembledDiagnostic] = []

    current_legacy_key: str | None = None
    current_legacy_run: list[SurveyPoint] = []

    def flush_legacy() -> None:
        nonlocal current_legacy_key, current_legacy_run
        if current_legacy_run:
            completed.append(
                AssembledFigure(
                    base_code=current_legacy_key or "",
                    points=tuple(current_legacy_run),
                    explicit_close=False,
                )
            )
        current_legacy_key = None
        current_legacy_run = []

    for point in points:
        if point.code is None:
            flush_legacy()
            unmatched.append(point)
            continue

        if not is_grammar_code(point.code):
            base = legacy_base_code(point.code)
            if current_legacy_key == base and current_legacy_run:
                current_legacy_run.append(point)
            else:
                flush_legacy()
                current_legacy_key = base
                current_legacy_run = [point]
            continue

        # Grammar-style code: any pending legacy run ends here, since
        # it's a different track.
        flush_legacy()

        try:
            parsed = FieldCodeParser.parse(point.code)
        except FieldCodeGrammarError:
            unmatched.append(point)
            diagnostics.append(
                AssembledDiagnostic(
                    base_code=point.code,
                    figure_id=None,
                    reason=AssembledDiagnosticReason.MALFORMED_CODE,
                    point_ids=(point.id,),
                )
            )
            continue

        assert parsed.figure_id is not None  # the parser never returns None for a successful parse
        key = (parsed.base_code, parsed.figure_id)

        if parsed.command == FieldCodeCommand.START:
            if key in open_figures:
                diagnostics.append(
                    AssembledDiagnostic(
                        base_code=parsed.base_code,
                        figure_id=parsed.figure_id,
                        reason=AssembledDiagnosticReason.DUPLICATE_START,
                        point_ids=(point.id,),
                    )
                )
                unmatched.append(point)
                continue
            open_figures[key] = [point]

        elif parsed.command == FieldCodeCommand.CONTINUE:
            if key not in open_figures:
                diagnostics.append(
                    AssembledDiagnostic(
                        base_code=parsed.base_code,
                        figure_id=parsed.figure_id,
                        reason=AssembledDiagnosticReason.CONTINUE_WITHOUT_START,
                        point_ids=(point.id,),
                    )
                )
                unmatched.append(point)
                continue
            open_figures[key].append(point)

        else:  # END or CLOSE
            if key not in open_figures:
                diagnostics.append(
                    AssembledDiagnostic(
                        base_code=parsed.base_code,
                        figure_id=parsed.figure_id,
                        reason=AssembledDiagnosticReason.END_WITHOUT_START,
                        point_ids=(point.id,),
                    )
                )
                unmatched.append(point)
                continue

            figure_points = open_figures.pop(key)
            figure_points.append(point)
            completed.append(
                AssembledFigure(
                    base_code=parsed.base_code,
                    points=tuple(figure_points),
                    explicit_close=(parsed.command == FieldCodeCommand.CLOSE),
                )
            )

    flush_legacy()

    for (base, figure_id), pts in open_figures.items():
        diagnostics.append(
            AssembledDiagnostic(
                base_code=base,
                figure_id=figure_id,
                reason=AssembledDiagnosticReason.UNCLOSED_FIGURE,
                point_ids=tuple(p.id for p in pts),
            )
        )
        unmatched.extend(pts)

    return AssembleResult(
        figures=tuple(completed),
        unmatched=tuple(unmatched),
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "AssembleResult",
    "AssembledDiagnostic",
    "AssembledDiagnosticReason",
    "AssembledFigure",
    "assemble",
]
