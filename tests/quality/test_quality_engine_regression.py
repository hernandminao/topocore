"""
Regression suite for topocore.quality (prototype, not yet a stable
public API -- see docs/24-usage/quality-engine-vision.md).

These tests protect BEHAVIOR, not just line coverage -- in
particular, the 2 real regressions found during this module's own
Point 4 (error-path) review:

1. QualityAnalyzer.analyze_density() used to let a raw OverflowError
   from StatisticsAnalysis.density() escape for non-finite input,
   instead of a diagnostic. Now validated at this module's own
   boundary before calling the (already-closed, untouched) PR22
   component.
2. pdf_renderer.py used field names from before the Point 2 contract
   change (source_crs/target_crs/transformation) and would raise
   AttributeError on any real render call. Now reconciled with
   models.py's real, current field names.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.quality.analyzer import QualityAnalyzer
from topocore.quality.models import DiagnosticSeverity, QualityResult
from topocore.quality.pdf_renderer import PDFRenderError, render_pdf

# ---------------------------------------------------------------------
# QualityAnalyzer.analyze_accuracy
# ---------------------------------------------------------------------


def test_accuracy_with_two_control_points_computes_rmse() -> None:
    analyzer = QualityAnalyzer("test")
    reference = np.array([[0.0, 0.0, 10.0], [10.0, 10.0, 12.0]])
    observed = np.array([[0.0, 0.0, 10.1], [10.0, 10.0, 12.0]])

    section = analyzer.analyze_accuracy(reference, observed)

    assert section is not None
    assert section.control_point_count == 2
    assert section.rmse_vertical == pytest.approx(0.07071068, abs=1e-6)


def test_accuracy_with_zero_control_points_returns_none_with_diagnostic() -> None:
    analyzer = QualityAnalyzer("test")

    section = analyzer.analyze_accuracy(np.empty((0, 3)), np.empty((0, 3)))

    assert section is None
    assert any(d.code == "ACCURACY_INSUFFICIENT_CONTROL_POINTS" for d in analyzer._diagnostics)


def test_accuracy_with_one_control_point_returns_none_with_diagnostic() -> None:
    analyzer = QualityAnalyzer("test")

    section = analyzer.analyze_accuracy(np.array([[0.0, 0.0, 10.0]]), np.array([[0.0, 0.0, 10.1]]))

    assert section is None
    diag = analyzer._diagnostics[-1]
    assert diag.code == "ACCURACY_INSUFFICIENT_CONTROL_POINTS"
    assert diag.severity == DiagnosticSeverity.WARNING


def test_accuracy_with_nan_returns_none_with_diagnostic_not_crash() -> None:
    analyzer = QualityAnalyzer("test")
    reference = np.array([[0.0, 0.0, 10.0], [10.0, 10.0, float("nan")]])
    observed = np.array([[0.0, 0.0, 10.1], [10.0, 10.0, 12.0]])

    section = analyzer.analyze_accuracy(reference, observed)

    assert section is None
    assert analyzer._diagnostics[-1].code == "ACCURACY_COMPUTE_FAILED"


# ---------------------------------------------------------------------
# QualityAnalyzer.analyze_density -- the real OverflowError regression
# ---------------------------------------------------------------------


def test_density_with_infinite_value_returns_none_not_a_crash() -> None:
    """
    Regression: this exact input used to raise a raw OverflowError
    from StatisticsAnalysis.density() (confirmed, real, pre-existing
    in already-closed PR22 code) instead of a diagnostic.
    """
    analyzer = QualityAnalyzer("test")
    points = np.array([[0.0, 0.0], [float("inf"), 5.0], [3.0, 3.0]])

    section = analyzer.analyze_density(points, resolution=1.0)

    assert section is None
    assert analyzer._diagnostics[-1].code == "DENSITY_NON_FINITE_POINTS"
    assert analyzer._diagnostics[-1].severity == DiagnosticSeverity.ERROR


def test_density_with_nan_returns_none_not_a_crash() -> None:
    analyzer = QualityAnalyzer("test")
    points = np.array([[0.0, 0.0], [float("nan"), 5.0]])

    section = analyzer.analyze_density(points, resolution=1.0)

    assert section is None
    assert analyzer._diagnostics[-1].code == "DENSITY_NON_FINITE_POINTS"


def test_density_with_valid_points_computes_normally() -> None:
    analyzer = QualityAnalyzer("test")
    rng = np.random.default_rng(1)
    points = rng.uniform(0, 20, (50, 2))

    section = analyzer.analyze_density(points, resolution=2.0)

    assert section is not None
    assert section.mean > 0.0


def test_density_with_geographic_crs_warns_but_still_computes() -> None:
    analyzer = QualityAnalyzer("test")
    rng = np.random.default_rng(1)
    points = rng.uniform(0, 20, (50, 2))

    section = analyzer.analyze_density(points, resolution=2.0, crs="EPSG:4326")

    assert section is not None  # not blocked
    assert any(d.code == "DENSITY_GEOGRAPHIC_CRS" for d in analyzer._diagnostics)


def test_density_with_unparseable_crs_does_not_crash() -> None:
    analyzer = QualityAnalyzer("test")
    rng = np.random.default_rng(1)
    points = rng.uniform(0, 20, (50, 2))

    section = analyzer.analyze_density(points, resolution=2.0, crs="not a real crs")

    assert section is not None


# ---------------------------------------------------------------------
# QualityAnalyzer.analyze_coverage
# ---------------------------------------------------------------------


def test_coverage_with_infinite_value_returns_none_with_diagnostic() -> None:
    analyzer = QualityAnalyzer("test")
    points = np.array([[0.0, 0.0], [float("inf"), 5.0]])

    section = analyzer.analyze_coverage(points, bounds=(0.0, 0.0, 10.0, 10.0), cell_size=1.0)

    assert section is None
    assert analyzer._diagnostics[-1].code == "COVERAGE_NON_FINITE_POINTS"


def test_coverage_with_valid_points_computes_percentage() -> None:
    analyzer = QualityAnalyzer("test")
    rng = np.random.default_rng(1)
    points = rng.uniform(0, 20, (50, 2))

    section = analyzer.analyze_coverage(points, bounds=(0.0, 0.0, 20.0, 20.0), cell_size=5.0)

    assert section is not None
    assert 0.0 <= section.coverage_percentage <= 100.0


def test_coverage_with_nonpositive_cell_size_returns_none() -> None:
    analyzer = QualityAnalyzer("test")
    points = np.array([[1.0, 1.0]])

    section = analyzer.analyze_coverage(points, bounds=(0.0, 0.0, 10.0, 10.0), cell_size=0.0)

    assert section is None
    assert analyzer._diagnostics[-1].code == "COVERAGE_INVALID_CELL_SIZE"


# ---------------------------------------------------------------------
# QualityAnalyzer.analyze_classification
# ---------------------------------------------------------------------


def test_classification_without_code_reports_na_not_zero() -> None:
    analyzer = QualityAnalyzer("test")
    codes = np.array([1, 2, 2, 1, 2])

    section = analyzer.analyze_classification(codes)

    assert section is not None
    assert section.code_provided is False
    assert section.ground_percentage is None  # NOT 0.0 -- N/A is not the same as zero


def test_classification_with_code_computes_percentage() -> None:
    analyzer = QualityAnalyzer("test")
    codes = np.array([1, 2, 2, 1, 2])

    section = analyzer.analyze_classification(codes, matched_code=2)

    assert section is not None
    assert section.code_provided is True
    assert section.ground_percentage == pytest.approx(60.0)


def test_classification_with_code_not_present_warns_but_reports_zero() -> None:
    analyzer = QualityAnalyzer("test")
    codes = np.array([1, 1, 2, 2, 2])

    section = analyzer.analyze_classification(codes, matched_code=99)

    assert section is not None
    assert section.matched_points == 0
    assert any(d.code == "CLASSIFICATION_CODE_NOT_FOUND" for d in analyzer._diagnostics)


def test_classification_with_no_data_returns_none() -> None:
    analyzer = QualityAnalyzer("test")

    section = analyzer.analyze_classification(np.array([], dtype=np.int64))

    assert section is None
    assert analyzer._diagnostics[-1].code == "CLASSIFICATION_NO_DATA"


# ---------------------------------------------------------------------
# QualityAnalyzer.build -- CRS and partial results
# ---------------------------------------------------------------------


def test_build_without_crs_warns() -> None:
    analyzer = QualityAnalyzer("test")

    result = analyzer.build()

    assert result.spatial_reference is not None
    assert result.spatial_reference.crs is None
    assert any(d.code == "SPATIAL_REFERENCE_MISSING" for d in analyzer._diagnostics)


def test_build_with_all_sections_none_is_still_a_valid_result() -> None:
    """A report where nothing could be computed is still valid, not an error."""
    analyzer = QualityAnalyzer("empty project")

    result = analyzer.build()

    assert result.accuracy is None
    assert result.density is None
    assert result.coverage is None
    assert result.classification is None
    assert result.project_name == "empty project"
    # confirmed real behavior: does not raise, produces valid JSON
    assert "empty project" in result.to_json()


def test_build_combines_independently_computed_sections() -> None:
    analyzer = QualityAnalyzer("mixed project")
    accuracy = analyzer.analyze_accuracy(
        np.array([[0.0, 0.0, 10.0], [10.0, 10.0, 12.0]]),
        np.array([[0.0, 0.0, 10.1], [10.0, 10.0, 12.0]]),
    )
    density = analyzer.analyze_density(np.random.default_rng(1).uniform(0, 20, (50, 2)), resolution=2.0)

    result = analyzer.build(accuracy=accuracy, density=density, crs="EPSG:32618")

    assert result.accuracy is not None
    assert result.density is not None
    assert result.coverage is None  # never computed -- stays None, not a fabricated value
    assert result.classification is None


# ---------------------------------------------------------------------
# QualityResult.to_json -- independent of reportlab
# ---------------------------------------------------------------------


def test_to_json_works_without_any_pdf_dependency() -> None:
    analyzer = QualityAnalyzer("json only")
    result = analyzer.build(crs="EPSG:32618")

    json_text = result.to_json()

    assert "json only" in json_text
    assert "EPSG:32618" in json_text


# ---------------------------------------------------------------------
# pdf_renderer -- the real field-name-drift regression
# ---------------------------------------------------------------------


def test_render_pdf_with_full_result_produces_a_real_file(tmp_path) -> None:
    """
    Regression: pdf_renderer.py used to reference field names
    (source_crs/target_crs/transformation) that no longer exist on
    SpatialReference after the Point 2 contract change, raising
    AttributeError on every real call. This exercises every section
    at once so a future field rename is caught immediately.
    """
    analyzer = QualityAnalyzer("full report")
    accuracy = analyzer.analyze_accuracy(
        np.array([[0.0, 0.0, 10.0], [10.0, 10.0, 12.0]]),
        np.array([[0.0, 0.0, 10.1], [10.0, 10.0, 12.0]]),
    )
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 20, (50, 2))
    density = analyzer.analyze_density(xy, resolution=2.0)
    coverage = analyzer.analyze_coverage(xy, bounds=(0.0, 0.0, 20.0, 20.0), cell_size=5.0)
    classification = analyzer.analyze_classification(np.array([1, 2, 2]), matched_code=2)
    result = analyzer.build(
        accuracy=accuracy,
        density=density,
        coverage=coverage,
        classification=classification,
        crs="EPSG:32618",
    )

    output_path = tmp_path / "full_report.pdf"
    returned_path = render_pdf(result, output_path)

    assert returned_path == output_path
    assert output_path.stat().st_size > 0


def test_render_pdf_with_partial_result_still_produces_a_valid_pdf(tmp_path) -> None:
    """
    A QualityResult where every section is None (nothing could be
    computed) must still render a valid PDF, not raise -- confirmed
    real behavior, not assumed.
    """
    analyzer = QualityAnalyzer("partial report")
    result = analyzer.build()  # every section None, plus the SPATIAL_REFERENCE_MISSING diagnostic

    output_path = tmp_path / "partial_report.pdf"
    render_pdf(result, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_pdf_to_an_unwritable_path_raises_specific_error(tmp_path) -> None:
    result = QualityResult(project_name="bad path test")

    with pytest.raises(PDFRenderError):
        render_pdf(result, "/this/path/cannot/possibly/exist/report.pdf")
