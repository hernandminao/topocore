"""
Regression suite for topocore.analysis.quality.hausdorff.
HausdorffDistance and .chamfer.ChamferDistance -- PR19. Both
verified against hand-computed known geometric cases. No bugs found.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from topocore.analysis.exceptions import QualityError
from topocore.analysis.quality.chamfer import ChamferDistance
from topocore.analysis.quality.hausdorff import HausdorffDistance


def test_hausdorff_known_case() -> None:
    set_a = np.array([[0.0, 0.0, 0.0]])
    set_b = np.array([[3.0, 4.0, 0.0], [10.0, 10.0, 10.0]])

    result = HausdorffDistance(percentile=100.0).compute(set_a, set_b)

    assert result.directed_forward == pytest.approx(5.0)
    assert result.directed_backward == pytest.approx(math.sqrt(300))
    assert result.hausdorff == pytest.approx(math.sqrt(300))


def test_hausdorff_identical_sets_is_zero() -> None:
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = HausdorffDistance().compute(points, points)
    assert result.hausdorff == pytest.approx(0.0)


def test_hausdorff_rejects_empty_set() -> None:
    with pytest.raises(QualityError):
        HausdorffDistance().compute(np.empty((0, 3)), np.array([[1.0, 1.0, 1.0]]))


def test_hausdorff_rejects_invalid_percentile() -> None:
    with pytest.raises(QualityError):
        HausdorffDistance(percentile=0.0)
    with pytest.raises(QualityError):
        HausdorffDistance(percentile=101.0)


def test_chamfer_known_case() -> None:
    set_a = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    set_b = np.array([[0.0, 3.0, 0.0], [10.0, 4.0, 0.0]])

    result = ChamferDistance().compute(set_a, set_b)

    assert result.directed_forward == pytest.approx(3.5)
    assert result.directed_backward == pytest.approx(3.5)
    assert result.chamfer == pytest.approx(3.5)


def test_chamfer_identical_sets_is_zero() -> None:
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = ChamferDistance().compute(points, points)
    assert result.chamfer == pytest.approx(0.0)


def test_chamfer_supports_2d() -> None:
    result = ChamferDistance().compute(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]]))
    assert result.chamfer == pytest.approx(5.0)


def test_chamfer_rejects_empty_set() -> None:
    with pytest.raises(QualityError):
        ChamferDistance().compute(np.empty((0, 3)), np.array([[1.0, 1.0, 1.0]]))
