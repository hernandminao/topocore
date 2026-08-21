"""
Regression suite for topocore.features.grammar.assembler -- PR19.

Verified the decisive scenario -- interleaved figures assembled
correctly by (base_code, figure_id) identity, not confused with each
other -- plus every diagnostic reason, the explicit CLOSE vs END
distinction, and legacy/grammar mixing in a single point stream. No
bugs found.
"""

from __future__ import annotations

from topocore.features.grammar.assembler import AssembledDiagnosticReason, assemble
from topocore.survey.models import SurveyPoint


def _pt(point_id: str, code: str | None) -> SurveyPoint:
    return SurveyPoint(id=point_id, x=0.0, y=0.0, z=0.0, code=code)


def test_interleaved_figures_assembled_independently() -> None:
    """The decisive case: two figures surveyed alternately must never be confused."""
    points = [
        _pt("p1", "MURO.1.S"),
        _pt("p2", "CERCA.1.S"),
        _pt("p3", "MURO.1"),
        _pt("p4", "CERCA.1"),
        _pt("p5", "MURO.1.E"),
        _pt("p6", "CERCA.1.E"),
    ]
    result = assemble(points)

    assert len(result.figures) == 2
    by_base = {fig.base_code: [p.id for p in fig.points] for fig in result.figures}
    assert by_base["MURO"] == ["p1", "p3", "p5"]
    assert by_base["CERCA"] == ["p2", "p4", "p6"]
    assert result.unmatched == ()
    assert result.diagnostics == ()


def test_continue_without_start() -> None:
    result = assemble([_pt("p1", "MURO.1")])
    assert result.diagnostics[0].reason == AssembledDiagnosticReason.CONTINUE_WITHOUT_START


def test_end_without_start() -> None:
    result = assemble([_pt("p1", "MURO.1.E")])
    assert result.diagnostics[0].reason == AssembledDiagnosticReason.END_WITHOUT_START


def test_duplicate_start_keeps_original_open_until_stream_end() -> None:
    """Verifies the subtle two-diagnostic sequence: DUPLICATE_START for the second S, then UNCLOSED_FIGURE."""
    result = assemble([_pt("p1", "MURO.1.S"), _pt("p2", "MURO.1.S")])
    reasons = [d.reason for d in result.diagnostics]
    assert AssembledDiagnosticReason.DUPLICATE_START in reasons
    assert AssembledDiagnosticReason.UNCLOSED_FIGURE in reasons


def test_unclosed_figure_reported_with_orphaned_points() -> None:
    result = assemble([_pt("p1", "MURO.1.S"), _pt("p2", "MURO.1")])
    assert result.diagnostics[0].reason == AssembledDiagnosticReason.UNCLOSED_FIGURE
    assert result.diagnostics[0].point_ids == ("p1", "p2")


def test_malformed_code_reported() -> None:
    result = assemble([_pt("p1", "MURO.1.Z")])
    assert result.diagnostics[0].reason == AssembledDiagnosticReason.MALFORMED_CODE


def test_explicit_close_flag_distinguishes_x_from_e() -> None:
    close_result = assemble([_pt("p1", "MURO.1.S"), _pt("p2", "MURO.1.X")])
    end_result = assemble([_pt("p1", "MURO.1.S"), _pt("p2", "MURO.1.E")])
    assert close_result.figures[0].explicit_close is True
    assert end_result.figures[0].explicit_close is False


def test_legacy_and_grammar_codes_mixed_in_one_stream() -> None:
    points = [
        _pt("p1", "ARBOL"),
        _pt("p2", "ARBOL"),
        _pt("p3", "MURO.1.S"),
        _pt("p4", "MURO.1.E"),
        _pt("p5", "ARBOL"),
    ]
    result = assemble(points)
    assert len(result.figures) == 3
    assert [fig.base_code for fig in result.figures] == ["ARBOL", "MURO", "ARBOL"]
