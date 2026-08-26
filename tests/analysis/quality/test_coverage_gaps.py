"""
Targeted coverage suite for topocore.analysis.quality.completeness,
.chamfer, .hausdorff, .gps_control, .correctness, .rmse, and
.registration -- PR20 coverage phase.

Formalizes real-domain-behavior verification already performed
manually during this session's audit (confirmed no bugs in any of
these 7 files) into permanent pytest regressions, closing the gaps
left by the existing PR19 regression files (which only exercised
basic happy-path cases for most of these modules):

- CompletenessAnalysis.compute(): a hand-constructed grid where
  exactly half the cells are covered (coverage_ratio == 0.5 exactly),
  points outside the reference bbox correctly excluded via masking,
  and every validation path. compute_from_grid()'s bitwise-AND
  coverage logic verified against a hand-computed 5-cell reference /
  3-cell-covered case.
- ChamferDistance: identical point sets (chamfer == 0), and --
  critically -- confirmed the scipy cKDTree path and the pure-NumPy
  fallback (_HAS_SCIPY=False) give IDENTICAL results on the same
  data, not just that both run without error.
- HausdorffDistance: a constructed outlier case where the classical
  (percentile=100) distance is dominated by a single far point
  (confirmed == that point's exact distance), while the modified
  (percentile=50) distance is robust to it -- plus the same
  scipy/fallback agreement check as Chamfer.
- GPSControlAnalysis: a hand-constructed confusion case (2 of 3
  points passing all three per-axis tolerances, 1 failing on Z).
- CorrectnessAnalysis: a hand-built confusion matrix (TP/commission-
  FP/disagreement-FP/FN in one 1x4 grid) with exactly known
  precision/recall, plus the "both grids entirely invalid" guard and
  its "model entirely invalid but reference has data" non-error edge
  case (all false negatives, not an exception).
- RMSEAnalysis: the 3-4-5 (n,2) and 3-4-12-13 (n,3) Pythagorean
  cases for exact horizontal/vertical/total verification, plus
  compute_3d and componentwise agreeing with compute() on the same
  data.
- RegistrationQuality: fitness computed against a known pass/fail
  split, and RMSE/max/mean verified against a constructed
  3-4-12-13-plus-zero residual set.

No bugs found -- only test coverage was added.
"""

from __future__ import annotations

import numpy as np
import pytest

import topocore.analysis.quality.chamfer as chamfer_module
import topocore.analysis.quality.hausdorff as hausdorff_module
from topocore.analysis.exceptions import QualityError
from topocore.analysis.quality.chamfer import ChamferDistance
from topocore.analysis.quality.completeness import CompletenessAnalysis
from topocore.analysis.quality.correctness import CorrectnessAnalysis
from topocore.analysis.quality.gps_control import GPSControlAnalysis
from topocore.analysis.quality.hausdorff import HausdorffDistance
from topocore.analysis.quality.registration import RegistrationQuality
from topocore.analysis.quality.rmse import RMSEAnalysis

# ----------------------------------------------------------------------
# CompletenessAnalysis
# ----------------------------------------------------------------------


def test_completeness_compute_known_half_coverage() -> None:
    analysis = CompletenessAnalysis(resolution=1.0)
    xs, ys = np.meshgrid(np.arange(0.5, 5, 1.0), np.arange(0.5, 10, 1.0))
    points = np.column_stack([xs.ravel(), ys.ravel()])

    result = analysis.compute(points, (0.0, 0.0, 10.0, 10.0))

    assert result.coverage_ratio == pytest.approx(0.5)
    assert result.total_area == pytest.approx(100.0)
    assert result.uncovered_area == pytest.approx(50.0)


def test_completeness_points_outside_bbox_are_masked_out() -> None:
    analysis = CompletenessAnalysis(resolution=1.0)
    points = np.array([[100.0, 100.0], [0.5, 0.5]])

    result = analysis.compute(points, (0.0, 0.0, 10.0, 10.0))

    assert result.coverage_ratio == pytest.approx(0.01)


def test_completeness_compute_from_grid_bitwise_and() -> None:
    analysis = CompletenessAnalysis(resolution=1.0)
    data_grid = np.array([[True, True, False], [False, True, True]])
    reference_grid = np.array([[True, True, True], [True, True, False]])

    result = analysis.compute_from_grid(data_grid, reference_grid)

    assert result.coverage_ratio == pytest.approx(0.6)


def test_completeness_call_matches_compute() -> None:
    analysis = CompletenessAnalysis(resolution=1.0)
    points = np.array([[1.0, 1.0]])
    assert (
        analysis(points, (0.0, 0.0, 10.0, 10.0)).coverage_ratio
        == analysis.compute(points, (0.0, 0.0, 10.0, 10.0)).coverage_ratio
    )


def test_completeness_resolution_property() -> None:
    assert CompletenessAnalysis(resolution=2.5).resolution == pytest.approx(2.5)


def test_completeness_rejects_nonpositive_resolution() -> None:
    with pytest.raises(QualityError, match="positive"):
        CompletenessAnalysis(resolution=0.0)


def test_completeness_rejects_empty_points() -> None:
    with pytest.raises(QualityError, match="not be empty"):
        CompletenessAnalysis().compute(np.empty((0, 2)), (0.0, 0.0, 10.0, 10.0))


def test_completeness_rejects_invalid_bbox_extent() -> None:
    with pytest.raises(QualityError, match="invalid extent"):
        CompletenessAnalysis().compute(np.array([[1.0, 1.0]]), (10.0, 0.0, 0.0, 10.0))


def test_completeness_compute_from_grid_rejects_empty_reference() -> None:
    with pytest.raises(QualityError, match="no valid cells"):
        CompletenessAnalysis().compute_from_grid(np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool))


def test_completeness_compute_from_grid_rejects_shape_mismatch() -> None:
    with pytest.raises(QualityError, match="shapes must match"):
        CompletenessAnalysis().compute_from_grid(np.zeros((2, 2), dtype=bool), np.zeros((3, 3), dtype=bool))


# ----------------------------------------------------------------------
# ChamferDistance
# ----------------------------------------------------------------------


def test_chamfer_identical_sets_is_zero() -> None:
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    result = ChamferDistance().compute(points, points)
    assert result.chamfer == pytest.approx(0.0)


def test_chamfer_scipy_and_fallback_agree() -> None:
    set_a = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    set_b = set_a + np.array([3.0, 4.0])
    metric = ChamferDistance()

    with_scipy = metric.compute(set_a, set_b)

    original = chamfer_module._HAS_SCIPY
    chamfer_module._HAS_SCIPY = False
    try:
        without_scipy = metric.compute(set_a, set_b)
    finally:
        chamfer_module._HAS_SCIPY = original

    assert with_scipy.chamfer == pytest.approx(without_scipy.chamfer)


def test_chamfer_call_matches_compute() -> None:
    points = np.array([[0.0, 0.0]])
    metric = ChamferDistance()
    assert metric(points, points).chamfer == metric.compute(points, points).chamfer


def test_chamfer_rejects_mismatched_dimensionality() -> None:
    with pytest.raises(QualityError, match="same dimensionality"):
        ChamferDistance().compute(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0, 0.0]]))


def test_chamfer_rejects_nan() -> None:
    with pytest.raises(QualityError, match="NaN or infinite"):
        ChamferDistance().compute(np.array([[0.0, np.nan]]), np.array([[1.0, 1.0]]))


def test_chamfer_rejects_empty_set() -> None:
    with pytest.raises(QualityError, match="not be empty"):
        ChamferDistance().compute(np.empty((0, 2)), np.array([[1.0, 1.0]]))


# ----------------------------------------------------------------------
# HausdorffDistance
# ----------------------------------------------------------------------


def test_hausdorff_classic_dominated_by_outlier() -> None:
    set_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    set_b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    result = HausdorffDistance(percentile=100.0).compute(set_a, set_b)

    assert result.directed_forward == pytest.approx(99.0)
    assert result.directed_backward == pytest.approx(0.0)
    assert result.hausdorff == pytest.approx(99.0)


def test_hausdorff_modified_is_robust_to_outlier() -> None:
    set_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    set_b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    result = HausdorffDistance(percentile=50.0).compute(set_a, set_b)

    assert result.directed_forward < 10.0  # nowhere near the 99.0 classical value


def test_hausdorff_scipy_and_fallback_agree() -> None:
    set_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    set_b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    metric = HausdorffDistance()

    with_scipy = metric.compute(set_a, set_b)

    original = hausdorff_module._HAS_SCIPY
    hausdorff_module._HAS_SCIPY = False
    try:
        without_scipy = metric.compute(set_a, set_b)
    finally:
        hausdorff_module._HAS_SCIPY = original

    assert with_scipy.hausdorff == pytest.approx(without_scipy.hausdorff)


def test_hausdorff_percentile_property() -> None:
    assert HausdorffDistance(percentile=75.0).percentile == pytest.approx(75.0)


def test_hausdorff_call_matches_compute() -> None:
    set_a = np.array([[0.0, 0.0, 0.0]])
    set_b = np.array([[1.0, 1.0, 1.0]])
    metric = HausdorffDistance()
    assert metric(set_a, set_b).hausdorff == metric.compute(set_a, set_b).hausdorff


@pytest.mark.parametrize("percentile", [0.0, 101.0, -5.0])
def test_hausdorff_rejects_invalid_percentile(percentile: float) -> None:
    with pytest.raises(QualityError, match="Percentile"):
        HausdorffDistance(percentile=percentile)


def test_hausdorff_rejects_wrong_shape() -> None:
    with pytest.raises(QualityError, match="shape \\(n, 3\\)"):
        HausdorffDistance().compute(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0, 0.0]]))


def test_hausdorff_rejects_empty_set() -> None:
    with pytest.raises(QualityError, match="not be empty"):
        HausdorffDistance().compute(np.empty((0, 3)), np.array([[0.0, 0.0, 0.0]]))


# ----------------------------------------------------------------------
# GPSControlAnalysis
# ----------------------------------------------------------------------


def test_gps_control_known_pass_fail_split() -> None:
    analysis = GPSControlAnalysis(tolerance_x=0.05, tolerance_y=0.05, tolerance_z=0.10)
    reference = np.array([[100.0, 200.0, 50.0], [110.0, 210.0, 55.0], [120.0, 220.0, 60.0]])
    observed = np.array([[100.02, 200.01, 50.05], [110.0, 210.0, 55.0], [120.0, 220.0, 60.5]])

    result = analysis.compute(reference, observed)

    assert result.points_passing == 2
    assert result.points_failing == 1
    assert result.max_deviation == pytest.approx(0.5)
    assert result.tolerance == pytest.approx(0.10)


def test_gps_control_tolerance_properties() -> None:
    analysis = GPSControlAnalysis(tolerance_x=0.05, tolerance_y=0.06, tolerance_z=0.07)
    assert analysis.tolerance_x == pytest.approx(0.05)
    assert analysis.tolerance_y == pytest.approx(0.06)
    assert analysis.tolerance_z == pytest.approx(0.07)


def test_gps_control_call_matches_compute() -> None:
    analysis = GPSControlAnalysis()
    reference = np.array([[0.0, 0.0, 0.0]])
    observed = np.array([[0.0, 0.0, 0.0]])
    assert analysis(reference, observed).points_passing == analysis.compute(reference, observed).points_passing


def test_gps_control_rejects_nonpositive_tolerance() -> None:
    with pytest.raises(QualityError, match="positive"):
        GPSControlAnalysis(tolerance_x=-1.0)


def test_gps_control_rejects_shape_mismatch() -> None:
    with pytest.raises(QualityError, match="Shape mismatch"):
        GPSControlAnalysis().compute(np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))


def test_gps_control_rejects_empty_arrays() -> None:
    with pytest.raises(QualityError, match="not be empty"):
        GPSControlAnalysis().compute(np.empty((0, 3)), np.empty((0, 3)))


def test_gps_control_rejects_nan() -> None:
    with pytest.raises(QualityError, match="NaN or infinite"):
        GPSControlAnalysis().compute(np.array([[1.0, np.nan, 1.0]]), np.array([[1.0, 1.0, 1.0]]))


# ----------------------------------------------------------------------
# CorrectnessAnalysis
# ----------------------------------------------------------------------


def test_correctness_known_confusion_matrix() -> None:
    analysis = CorrectnessAnalysis(tolerance=0.1)
    # cell0: TP (agree); cell1: FP-disagreement; cell2: FP-commission (model has data, ref NaN); cell3: FN
    model = np.array([[1.0, 5.0, 3.0, np.nan]])
    reference = np.array([[1.05, 5.5, np.nan, 4.0]])

    result = analysis.compute(model, reference)

    assert result.true_positives == 1
    assert result.false_positives == 2
    assert result.false_negatives == 1
    assert result.precision == pytest.approx(1 / 3)
    assert result.recall == pytest.approx(0.5)


def test_correctness_model_entirely_invalid_but_reference_has_data_is_not_an_error() -> None:
    analysis = CorrectnessAnalysis()
    result = analysis.compute(np.full((1, 2), np.nan), np.array([[1.0, 2.0]]))
    assert result.false_negatives == 2


def test_correctness_tolerance_property() -> None:
    assert CorrectnessAnalysis(tolerance=0.2).tolerance == pytest.approx(0.2)


def test_correctness_call_matches_compute() -> None:
    analysis = CorrectnessAnalysis()
    model = np.array([[1.0]])
    reference = np.array([[1.0]])
    assert analysis(model, reference).true_positives == analysis.compute(model, reference).true_positives


def test_correctness_rejects_nonpositive_tolerance() -> None:
    with pytest.raises(QualityError, match="positive"):
        CorrectnessAnalysis(tolerance=0.0)


def test_correctness_rejects_shape_mismatch() -> None:
    with pytest.raises(QualityError, match="shapes must match"):
        CorrectnessAnalysis().compute(np.array([[1.0, 2.0]]), np.array([[1.0]]))


def test_correctness_rejects_both_grids_entirely_invalid() -> None:
    with pytest.raises(QualityError, match="no valid data"):
        CorrectnessAnalysis().compute(np.full((2, 2), np.nan), np.full((2, 2), np.nan))


# ----------------------------------------------------------------------
# RMSEAnalysis
# ----------------------------------------------------------------------


def test_rmse_2d_known_3_4_5() -> None:
    reference = np.array([[0.0, 0.0], [0.0, 0.0]])
    observed = np.array([[3.0, 4.0], [3.0, 4.0]])
    result = RMSEAnalysis.compute(reference, observed)
    assert result.horizontal == pytest.approx(5.0)


def test_rmse_3d_known_3_4_12_13() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    observed = np.array([[3.0, 4.0, 12.0], [3.0, 4.0, 12.0]])
    result = RMSEAnalysis.compute(reference, observed)
    assert result.horizontal == pytest.approx(5.0)
    assert result.vertical == pytest.approx(12.0)
    assert result.total == pytest.approx(13.0)


def test_rmse_compute_3d_matches_compute() -> None:
    reference = np.array([[0.0, 0.0, 0.0]])
    observed = np.array([[3.0, 4.0, 12.0]])
    assert RMSEAnalysis.compute_3d(reference, observed).total == RMSEAnalysis.compute(reference, observed).total


def test_rmse_componentwise_matches_compute() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    observed = np.array([[3.0, 4.0, 12.0], [3.0, 4.0, 12.0]])
    componentwise_result = RMSEAnalysis.componentwise(
        reference[:, 0],
        reference[:, 1],
        reference[:, 2],
        observed[:, 0],
        observed[:, 1],
        observed[:, 2],
    )
    assert componentwise_result.total == RMSEAnalysis.compute(reference, observed).total


def test_rmse_call_matches_compute() -> None:
    rmse = RMSEAnalysis()
    reference = np.array([0.0])
    observed = np.array([3.0])
    assert rmse(reference, observed).vertical == RMSEAnalysis.compute(reference, observed).vertical


def test_rmse_rejects_unsupported_shape() -> None:
    with pytest.raises(QualityError, match="Unsupported shape"):
        RMSEAnalysis.compute(np.zeros((2, 4)), np.zeros((2, 4)))


def test_rmse_rejects_empty() -> None:
    with pytest.raises(QualityError, match="not be empty"):
        RMSEAnalysis.compute(np.empty(0), np.empty(0))


def test_rmse_componentwise_rejects_shape_mismatch() -> None:
    with pytest.raises(QualityError, match="same shape"):
        RMSEAnalysis.componentwise(
            np.array([1.0]),
            np.array([1.0, 2.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
        )


# ----------------------------------------------------------------------
# RegistrationQuality
# ----------------------------------------------------------------------


def test_registration_fitness_known_pass_fail_split() -> None:
    quality = RegistrationQuality(tolerance=0.1)
    residuals = np.array([0.05, -0.08, 0.15, 0.2, 0.02])
    result = quality.compute(residuals)
    assert result.fitness == pytest.approx(0.6)


def test_registration_3d_rmse_known_values() -> None:
    quality = RegistrationQuality(tolerance=0.1)
    residuals = np.array([[3.0, 4.0, 12.0], [0.0, 0.0, 0.0]])
    result = quality.compute(residuals)
    assert result.max_distance == pytest.approx(13.0)
    assert result.mean_distance == pytest.approx(6.5)
    assert result.rmse == pytest.approx(np.sqrt(169 / 2))


def test_registration_tolerance_property() -> None:
    assert RegistrationQuality(tolerance=0.2).tolerance == pytest.approx(0.2)


def test_registration_call_matches_compute() -> None:
    quality = RegistrationQuality()
    residuals = np.array([0.01])
    assert quality(residuals).fitness == quality.compute(residuals).fitness


def test_registration_rejects_nonpositive_tolerance() -> None:
    with pytest.raises(QualityError, match="positive"):
        RegistrationQuality(tolerance=0.0)


def test_registration_rejects_unsupported_shape() -> None:
    with pytest.raises(QualityError, match="Unsupported residual shape"):
        RegistrationQuality().compute(np.zeros((3, 2)))


def test_registration_rejects_nan_does_not_silently_exclude() -> None:
    """Confirms the PR19 fix: NaN residuals are rejected outright, never silently excluded."""
    with pytest.raises(QualityError, match="NaN or infinite"):
        RegistrationQuality().compute(np.array([1.0, np.nan]))


def test_registration_rejects_empty() -> None:
    with pytest.raises(QualityError, match="not be empty"):
        RegistrationQuality().compute(np.empty(0))
