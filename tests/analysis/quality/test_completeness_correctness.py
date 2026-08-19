"""
Regression suite for topocore.analysis.quality.completeness and
.correctness -- PR19.

CorrectnessAnalysis specifically verified with a hand-built
TP/FP/FN case that exercises the invalid-vs-low-quality distinction
via NaN: NaN in either grid means "no data at this cell" (a
coverage/agreement concept, not corrupt-data rejection) -- correctly
contributes to commission/omission counts rather than aborting the
whole computation. No bugs found in either module.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis.exceptions import QualityError
from topocore.analysis.quality.completeness import CompletenessAnalysis
from topocore.analysis.quality.correctness import CorrectnessAnalysis

# ----------------------------------------------------------------------
# CompletenessAnalysis
# ----------------------------------------------------------------------


def test_half_coverage() -> None:
    xs, ys = [], []
    for row in range(5):
        for col in range(10):
            xs.append(col + 0.5)
            ys.append(row + 0.5)
    points = np.column_stack([xs, ys])

    result = CompletenessAnalysis(resolution=1.0).compute(points, (0.0, 0.0, 10.0, 10.0))
    assert result.coverage_ratio == pytest.approx(0.5)


def test_full_coverage() -> None:
    xs, ys = [], []
    for row in range(10):
        for col in range(10):
            xs.append(col + 0.5)
            ys.append(row + 0.5)
    points = np.column_stack([xs, ys])

    result = CompletenessAnalysis(resolution=1.0).compute(points, (0.0, 0.0, 10.0, 10.0))
    assert result.coverage_ratio == pytest.approx(1.0)


def test_points_outside_bbox_do_not_count() -> None:
    points = np.array([[100.0, 100.0]])
    result = CompletenessAnalysis(resolution=1.0).compute(points, (0.0, 0.0, 10.0, 10.0))
    assert result.coverage_ratio == pytest.approx(0.0)


def test_rejects_invalid_bbox() -> None:
    with pytest.raises(QualityError):
        CompletenessAnalysis().compute(np.array([[1.0, 1.0]]), (10.0, 10.0, 0.0, 0.0))


# ----------------------------------------------------------------------
# CorrectnessAnalysis -- hand-built TP/FP/FN case.
# ----------------------------------------------------------------------


def test_hand_built_confusion_matrix() -> None:
    """
    (0,0): both valid, agree -> TP
    (0,1): both valid, disagree -> FP (disagreement)
    (1,0): model has data, reference NaN -> FP (commission)
    (1,1): model NaN, reference has data -> FN
    """
    model = np.array([[10.0, 20.0], [30.0, np.nan]])
    reference = np.array([[10.01, 21.0], [np.nan, 40.0]])

    result = CorrectnessAnalysis(tolerance=0.05).compute(model, reference)

    assert result.true_positives == 1
    assert result.false_positives == 2
    assert result.false_negatives == 1

    expected_precision = 1 / 3
    expected_recall = 1 / 2
    expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)

    assert result.precision == pytest.approx(expected_precision)
    assert result.recall == pytest.approx(expected_recall)
    assert result.f1_score == pytest.approx(expected_f1)


def test_perfect_agreement_gives_f1_of_one() -> None:
    grid = np.array([[10.0, 20.0], [30.0, 40.0]])
    result = CorrectnessAnalysis(tolerance=0.05).compute(grid, grid.copy())
    assert result.f1_score == pytest.approx(1.0)


def test_rejects_both_grids_entirely_invalid() -> None:
    with pytest.raises(QualityError):
        CorrectnessAnalysis().compute(np.full((2, 2), np.nan), np.full((2, 2), np.nan))


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(QualityError):
        CorrectnessAnalysis().compute(np.zeros((2, 2)), np.zeros((3, 3)))
