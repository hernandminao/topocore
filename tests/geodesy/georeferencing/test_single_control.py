"""
Tests for topocore.geodesy.georeferencing.single_control.fit_translation_only().
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.geodesy.georeferencing import ControlPoint
from topocore.geodesy.georeferencing.exceptions import InsufficientControlPointsError
from topocore.geodesy.georeferencing.single_control import fit_translation_only


def test_single_control_point_gives_exact_translation() -> None:
    control = ControlPoint(id="1", source=(0.0, 0.0, 0.0), target=(5.0, 3.0, 2.0))
    result = fit_translation_only([control])

    assert result.parameters.tx == pytest.approx(5.0)
    assert result.parameters.ty == pytest.approx(3.0)
    assert result.parameters.tz == pytest.approx(2.0)
    assert result.parameters.rx == 0.0
    assert result.parameters.ry == 0.0
    assert result.parameters.rz == 0.0
    assert result.parameters.scale == 0.0
    assert result.rms == pytest.approx(0.0, abs=1e-12)
    assert len(result.residuals) == 1
    assert result.residuals[0].magnitude == pytest.approx(0.0, abs=1e-12)


def test_multiple_consistent_controls_give_the_same_translation() -> None:
    controls = [
        ControlPoint(id="1", source=(0.0, 0.0, 0.0), target=(10.0, 20.0, 30.0)),
        ControlPoint(id="2", source=(5.0, 5.0, 5.0), target=(15.0, 25.0, 35.0)),
        ControlPoint(id="3", source=(-3.0, 8.0, 1.0), target=(7.0, 28.0, 31.0)),
    ]
    result = fit_translation_only(controls)

    assert result.parameters.tx == pytest.approx(10.0)
    assert result.parameters.ty == pytest.approx(20.0)
    assert result.parameters.tz == pytest.approx(30.0)
    assert result.rms == pytest.approx(0.0, abs=1e-9)


def test_inconsistent_controls_average_with_documented_nonzero_rms() -> None:
    """
    Documented behavior, not a rejection: with genuine inconsistency,
    tx/ty/tz are the mean of target-source across controls, and RMS
    is genuinely nonzero -- confirmed against a hand-computed value.
    """
    controls = [
        ControlPoint(id="1", source=(0.0, 0.0, 0.0), target=(5.0, 3.0, 2.0)),
        ControlPoint(id="2", source=(10.0, 0.0, 0.0), target=(16.0, 3.0, 2.0)),
        ControlPoint(id="3", source=(0.0, 10.0, 0.0), target=(5.0, 14.0, 2.0)),
    ]
    result = fit_translation_only(controls)

    assert result.parameters.tx == pytest.approx(16.0 / 3.0)
    assert result.parameters.ty == pytest.approx(10.0 / 3.0)
    assert result.parameters.tz == pytest.approx(2.0)
    assert result.rms == pytest.approx(0.6666666666666669)


def test_zero_controls_raises() -> None:
    with pytest.raises(InsufficientControlPointsError, match="at least 1"):
        fit_translation_only([])


def test_control_point_rejects_non_finite_source() -> None:
    from topocore.geodesy.exceptions import ValidationError

    with pytest.raises(ValidationError, match="NaN or infinite"):
        ControlPoint(id="1", source=(float("nan"), 0.0, 0.0), target=(1.0, 1.0, 1.0))


def test_control_point_rejects_non_finite_target() -> None:
    from topocore.geodesy.exceptions import ValidationError

    with pytest.raises(ValidationError, match="NaN or infinite"):
        ControlPoint(id="1", source=(0.0, 0.0, 0.0), target=(float("inf"), 1.0, 1.0))


def test_helmert_fit_result_rms_is_a_real_float_never_none() -> None:
    """HelmertFitResult.rms is always a genuine number -- confirmed by type, not just by this one value."""
    control = ControlPoint(id="1", source=(0.0, 0.0, 0.0), target=(5.0, 3.0, 2.0))
    result = fit_translation_only([control])

    assert isinstance(result.rms, float)
    assert not np.isnan(result.rms)
