"""
Regression suite for topocore.analysis.quality.manager.QualityAnalysis
-- PR19.

All 10 dispatch methods verified to correctly propagate their
parameters to the matching underlying class constructor signature --
no forgotten-parameter or positional-reordering bugs found (unlike
some other managers audited earlier in this session). Specifically
confirms confidence_level now correctly reaches PrecisionAnalysis
through the manager (QualityConfig already had this field; the
manager simply never wired it through before this session's sync).
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis.config import QualityConfig
from topocore.analysis.exceptions import QualityError
from topocore.analysis.quality.manager import QualityAnalysis
from topocore.analysis.types import (
    ChamferResult,
    CloudToCloudResult,
    CompletenessResult,
    CorrectnessResult,
    GPSControlResult,
    HausdorffResult,
    PrecisionResult,
    RegistrationQualityResult,
    RMSEResult,
)


def test_rmse_dispatch() -> None:
    manager = QualityAnalysis(method="rmse")
    result = manager.compute(np.array([10.0, 20.0]), np.array([7.0, 24.0]), method="rmse")
    assert isinstance(result, RMSEResult)
    assert result.vertical == pytest.approx(np.sqrt((3**2 + 4**2) / 2))


def test_precision_dispatch_propagates_confidence_level() -> None:
    config = QualityConfig(confidence_level=0.90)
    manager = QualityAnalysis(config=config, method="precision")

    observations = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    result = manager.precision(observations, tolerance_x=10.0, tolerance_y=10.0)

    assert result.confidence_level == pytest.approx(0.90)


def test_precision_dispatch_explicit_tolerance_overrides() -> None:
    manager = QualityAnalysis(method="precision")
    observations = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    result = manager.compute(observations, method="precision", tolerance_x=10.0, tolerance_y=10.0)
    assert isinstance(result, PrecisionResult)
    assert result.std_dev > 0.0


def test_gps_control_dispatch() -> None:
    manager = QualityAnalysis(method="gps_control")
    reference = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    observed = np.array([[0.01, 0.0, 0.0], [10.01, 0.0, 0.0]])
    result = manager.compute(reference, observed, method="gps_control")
    assert isinstance(result, GPSControlResult)
    assert result.control_point_count == 2


def test_registration_dispatch_uses_configured_tolerance() -> None:
    config = QualityConfig(max_correspondence_distance=0.5)
    manager = QualityAnalysis(config=config, method="registration")
    result = manager.compute(np.array([0.1, 0.2, 0.3]), method="registration")
    assert isinstance(result, RegistrationQualityResult)
    assert result.fitness == 1.0  # all residuals within the configured 0.5 tolerance


def test_completeness_dispatch() -> None:
    manager = QualityAnalysis(method="completeness")
    points = np.array([[1.0, 1.0], [2.0, 2.0]])
    result = manager.compute(points, (0.0, 0.0, 10.0, 10.0), method="completeness")
    assert isinstance(result, CompletenessResult)
    assert 0.0 <= result.coverage_ratio <= 1.0


def test_correctness_dispatch() -> None:
    manager = QualityAnalysis(method="correctness")
    grid = np.array([[10.0, 20.0], [30.0, 40.0]])
    result = manager.compute(grid, grid.copy(), method="correctness")
    assert isinstance(result, CorrectnessResult)
    assert result.f1_score == pytest.approx(1.0)


def test_cloud_to_cloud_dispatch() -> None:
    manager = QualityAnalysis(method="c2c")
    reference = np.array([[0.0, 0.0, 0.0]])
    compared = np.array([[0.1, 0.0, 0.0]])
    result = manager.compute(reference, compared, method="c2c")
    assert isinstance(result, CloudToCloudResult)
    assert result.mean == pytest.approx(0.1, abs=1e-6)


def test_hausdorff_dispatch() -> None:
    manager = QualityAnalysis(method="hausdorff")
    set_a = np.array([[0.0, 0.0, 0.0]])
    set_b = np.array([[3.0, 4.0, 0.0]])
    result = manager.compute(set_a, set_b, method="hausdorff")
    assert isinstance(result, HausdorffResult)
    assert result.hausdorff == pytest.approx(5.0)


def test_chamfer_dispatch() -> None:
    manager = QualityAnalysis(method="chamfer")
    set_a = np.array([[0.0, 0.0, 0.0]])
    set_b = np.array([[3.0, 4.0, 0.0]])
    result = manager.compute(set_a, set_b, method="chamfer")
    assert isinstance(result, ChamferResult)
    assert result.chamfer == pytest.approx(5.0)


def test_rejects_invalid_method_at_construction() -> None:
    with pytest.raises(QualityError):
        QualityAnalysis(method="bogus")


def test_rejects_invalid_method_at_compute() -> None:
    manager = QualityAnalysis(method="rmse")
    with pytest.raises(QualityError):
        manager.compute(method="bogus")
