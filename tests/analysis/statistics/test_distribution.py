"""
Regression suite for topocore.analysis.statistics.distribution.
DistributionStatistics -- PR19.

Includes a real, severe bug found and fixed earlier in this session:
skewness and kurtosis mixed SAMPLE standard deviation (ddof=1,
Bessel's correction) in the denominator with POPULATION-style
moments (dividing by N, not N-1) in the numerator -- a non-standard
hybrid matching neither scipy.stats.skew/kurtosis(bias=True) nor
bias=False. For large samples the discrepancy was small (~0.3% at
n=1000), but for small samples -- common in terrain statistics --
it became severe: confirmed with n=6, kurtosis differed by 1676% and
FLIPPED SIGN (-0.88 vs +0.056), which would have reported a
qualitatively wrong distribution shape. No other code in this
repository read DistributionStats.skewness/.kurtosis, so there was
no existing convention to preserve; fixed to use population
statistics consistently throughout (ddof=0), matching
scipy.stats.skew/kurtosis(bias=True) exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from topocore.analysis.exceptions import StatisticsError
from topocore.analysis.statistics.distribution import DistributionStatistics


def test_skewness_matches_scipy_population_convention_small_sample() -> None:
    """
    The exact regression: before the fix, n=6 gave a kurtosis of the
    WRONG SIGN relative to the population-consistent scipy result.
    """
    rng = np.random.default_rng(1)
    values = rng.exponential(scale=2.0, size=6)

    result = DistributionStatistics(num_bins=3).compute(values)

    expected_skew = stats.skew(values, bias=True)
    expected_kurt = stats.kurtosis(values, bias=True)

    assert result.skewness == pytest.approx(expected_skew, abs=1e-9)
    assert result.kurtosis == pytest.approx(expected_kurt, abs=1e-9)


def test_skewness_matches_scipy_population_convention_large_sample() -> None:
    rng = np.random.default_rng(0)
    values = rng.exponential(scale=2.0, size=1000)

    result = DistributionStatistics(num_bins=10).compute(values)

    expected_skew = stats.skew(values, bias=True)
    expected_kurt = stats.kurtosis(values, bias=True)

    assert result.skewness == pytest.approx(expected_skew, abs=1e-9)
    assert result.kurtosis == pytest.approx(expected_kurt, abs=1e-9)


def test_symmetric_distribution_has_near_zero_skewness() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # perfectly symmetric
    result = DistributionStatistics(num_bins=3).compute(values)
    assert result.skewness == pytest.approx(0.0, abs=1e-9)


def test_constant_values_give_zero_skewness_and_kurtosis() -> None:
    values = np.array([5.0, 5.0, 5.0, 5.0])
    result = DistributionStatistics(num_bins=2).compute(values)
    assert result.skewness == 0.0
    assert result.kurtosis == 0.0


def test_nan_and_inf_excluded_from_computation() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.inf, -np.inf])
    result = DistributionStatistics(num_bins=2).compute(values)
    assert sum(result.counts) == 4


def test_rejects_fewer_than_four_finite_values() -> None:
    with pytest.raises(StatisticsError):
        DistributionStatistics().compute(np.array([1.0, 2.0, 3.0]))


def test_percentiles_known_values() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = DistributionStatistics().percentiles(values, [0.0, 50.0, 100.0])
    assert result[0.0] == pytest.approx(1.0)
    assert result[50.0] == pytest.approx(5.5)
    assert result[100.0] == pytest.approx(10.0)


def test_percentiles_rejects_out_of_range() -> None:
    with pytest.raises(StatisticsError):
        DistributionStatistics().percentiles(np.array([1.0, 2.0]), [150.0])


def test_quantile_ranges_known_iqr() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = DistributionStatistics().quantile_ranges(values)
    assert result["iqr"] == pytest.approx(result["q3"] - result["q1"])
    assert result["lower_whisker"] >= float(np.min(values))
    assert result["upper_whisker"] <= float(np.max(values))
