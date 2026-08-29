"""
Coverage audit tests for topocore.processing.filters.statistical.StatisticalOutlierFilter.

Only the constructor's `k < 1` and `min_points < 0` validations were
confirmed missing from the original coverage report -- std_ratio<=0
and the strict/non-strict min_points fallback branches (mirroring
the same pattern already found in RadiusOutlierFilter during the
classification/rules.py audit) were already covered by the existing
test suite.

name() is documented as orphaned -- zero external callers confirmed
via grep.
"""

from __future__ import annotations

import pytest
from topocore.processing.exceptions import FilterError
from topocore.processing.filters.statistical import StatisticalOutlierFilter


def test_k_less_than_one_rejected() -> None:
    with pytest.raises(FilterError, match="k must be at least 1"):
        StatisticalOutlierFilter(k=0)


def test_min_points_negative_rejected() -> None:
    with pytest.raises(FilterError, match="min_points cannot be negative"):
        StatisticalOutlierFilter(min_points=-1)
