"""
Regression suite for topocore.analysis.quality.precision,
.gps_control, and .registration -- PR19.

This module was synced mid-session against Hernán's own corrected
real repository, which fixed several issues my own earlier audit
either missed or got backward:

1. precision.py: the confidence-interval margin now correctly uses
   the STANDARD ERROR OF THE MEAN (t * s / sqrt(n)), not the raw
   sample standard deviation (t * s) -- a real statistics bug my
   earlier audit missed (I only verified the t-value against scipy,
   never the full margin formula). Confirmed directly: for n=3,
   s=sqrt(2), the old (wrong) formula gave margin=6.08, the correct
   formula gives margin=3.51.

2. precision.py also gained a configurable `confidence_level`
   parameter, correctly used in the margin computation -- but a
   SEPARATE bug was found and fixed in the SAME session-sync pass:
   PrecisionResult.confidence_level was hardcoded to 0.95 in the
   return statement, ignoring self._confidence_level entirely.
   Confirmed directly: constructing with confidence_level=0.90 or
   0.99 still reported 0.95 in the result. Fixed to use
   self._confidence_level.

3. registration.py: NaN/Inf residuals are now REJECTED outright,
   not silently excluded. My own earlier audit had concluded
   exclusion was a legitimate, domain-appropriate choice (ICP
   correspondence residuals commonly have "no match" entries) --
   this was WRONG. For a QUALITY metric specifically, "no match
   found" is itself a quality signal (poor correspondence) that
   must be visible in the fitness score, not hidden by only
   averaging over the well-matched subset -- exactly the
   invalid-vs-low-quality distinction this session was asked to
   watch for, applied one layer deeper than my own audit went.

4. gps_control.py: unaffected by the sync (already correctly
   rejected NaN/Inf outright).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t

from topocore.analysis.exceptions import QualityError
from topocore.analysis.quality.gps_control import GPSControlAnalysis
from topocore.analysis.quality.precision import PrecisionAnalysis
from topocore.analysis.quality.registration import RegistrationQuality

# ----------------------------------------------------------------------
# PrecisionAnalysis -- the confidence-margin formula fix.
# ----------------------------------------------------------------------


def test_confidence_margin_uses_standard_error_of_the_mean() -> None:
    """
    The exact regression: before the fix, margin = t * s (no
    division by sqrt(n)), giving 6.08 instead of the correct 3.51
    for this n=3 case.
    """
    observations = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    result = PrecisionAnalysis(tolerance_x=10, tolerance_y=10).compute(observations)

    n = 3
    s = np.sqrt(2)
    t_value = t.ppf(0.975, n - 1)
    expected_margin = t_value * s / np.sqrt(n)

    assert result.confidence_interval[1] == pytest.approx(expected_margin)
    assert result.confidence_interval[1] != pytest.approx(t_value * s)  # NOT the old, wrong formula


def test_confidence_level_is_reported_correctly_not_hardcoded() -> None:
    """
    The second regression found during this same sync: the RESULT's
    own confidence_level field was hardcoded to 0.95 regardless of
    what was actually configured and used for the margin.
    """
    observations = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    result_90 = PrecisionAnalysis(tolerance_x=10, tolerance_y=10, confidence_level=0.90).compute(observations)
    result_99 = PrecisionAnalysis(tolerance_x=10, tolerance_y=10, confidence_level=0.99).compute(observations)

    assert result_90.confidence_level == pytest.approx(0.90)
    assert result_99.confidence_level == pytest.approx(0.99)
    assert result_90.confidence_level != result_99.confidence_level


def test_confidence_level_affects_the_actual_margin() -> None:
    observations = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    result_90 = PrecisionAnalysis(tolerance_x=10, tolerance_y=10, confidence_level=0.90).compute(observations)
    result_99 = PrecisionAnalysis(tolerance_x=10, tolerance_y=10, confidence_level=0.99).compute(observations)

    assert result_99.confidence_interval[1] > result_90.confidence_interval[1]  # wider interval for higher confidence


def test_rejects_invalid_confidence_level() -> None:
    with pytest.raises(QualityError):
        PrecisionAnalysis(confidence_level=1.5)
    with pytest.raises(QualityError):
        PrecisionAnalysis(confidence_level=0.0)


def test_rejects_fewer_than_three_observations() -> None:
    with pytest.raises(QualityError):
        PrecisionAnalysis().compute(np.array([[1.0, 1.0], [2.0, 2.0]]))


def test_rejects_nan_as_invalid_not_low_precision() -> None:
    with pytest.raises(QualityError):
        PrecisionAnalysis().compute(np.array([[1.0, 1.0], [np.nan, 2.0], [3.0, 3.0]]))


# ----------------------------------------------------------------------
# GPSControlAnalysis -- unaffected by the sync.
# ----------------------------------------------------------------------


def test_hand_built_pass_fail_case() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    observed = np.array([[0.01, 0.01, 0.01], [10.10, 0.0, 0.0], [0.0, 0.0, 10.5]])

    result = GPSControlAnalysis(tolerance_x=0.05, tolerance_y=0.05, tolerance_z=0.10).compute(reference, observed)

    assert result.points_passing == 1
    assert result.points_failing == 2
    assert result.max_deviation == pytest.approx(0.5)


def test_gps_control_rejects_nan_coordinates() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    observed = np.array([[0.0, 0.0, np.nan], [1.0, 1.0, 1.0]])
    with pytest.raises(QualityError):
        GPSControlAnalysis().compute(reference, observed)


# ----------------------------------------------------------------------
# RegistrationQuality -- the NaN-rejection fix.
# ----------------------------------------------------------------------


def test_known_fitness_and_rmse() -> None:
    residuals = np.array([0.01, 0.02, 0.03, 0.10])
    result = RegistrationQuality(tolerance=0.05).compute(residuals)

    assert result.fitness == pytest.approx(0.75)
    assert result.rmse == pytest.approx(np.sqrt(np.mean(residuals**2)))


def test_nan_residuals_now_rejected_not_silently_excluded() -> None:
    """
    The corrected regression: an earlier version of this codebase
    (and this session's own, mistaken, prior audit) treated NaN
    residuals as "no correspondence found" and excluded them from
    the fitness computation. This was wrong for a QUALITY metric:
    silently excluding unmatched points would artificially inflate
    the reported fitness, hiding a real registration problem.
    """
    residuals = np.array([0.01, 0.02, np.nan, 0.10])
    with pytest.raises(QualityError):
        RegistrationQuality(tolerance=0.05).compute(residuals)


def test_3d_residuals_use_euclidean_norm() -> None:
    residuals = np.array([[3.0, 4.0, 0.0]])
    result = RegistrationQuality(tolerance=10.0).compute(residuals)
    assert result.rmse == pytest.approx(5.0)


def test_rejects_empty_residuals() -> None:
    with pytest.raises(QualityError):
        RegistrationQuality().compute(np.array([]))


def test_rejects_nonfinite_tolerance() -> None:
    with pytest.raises(QualityError):
        RegistrationQuality(tolerance=float("inf"))
