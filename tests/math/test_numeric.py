"""
Coverage audit tests for topocore.math.numeric.

Phase C audit finding: the entire module is confirmed unadopted --
zero real callers anywhere in the codebase for any of its 7
functions (confirmed via exhaustive grep; the many `mean(...)` /
`square(...)` hits found are all `np.mean()`/`.mean()` NumPy calls or
entirely unrelated methods on other classes -- e.g.
processing/backend/base.py's own abstract `mean()`/`square()`
methods, part of a SEPARATE, ALSO fully unadopted NumPy/CuPy/Torch/
JAX backend-abstraction layer; neither module superseded the other,
the actual processing code simply calls raw numpy operations
directly instead of going through either abstraction).

Not treated as dead code to remove: these are simple, correct,
directly-testable, general-purpose utility functions with no
architectural blocker preventing real use, matching this audit's
established policy for legitimate-but-currently-unused public API
(as with SegmentationManager, FilterManager, and others audited
earlier in this session).

Minor inconsistency noted, not changed here: `__all__` lists only
clamp, lerp, mean, and safe_divide -- cube, square, and sign are
defined and exported as module attributes (still importable
directly) but omitted from `__all__`, with no comment explaining the
distinction. Left as-is pending an explicit decision, consistent
with this audit's discipline of not fixing things "because they look
improvable" without a demonstrated defect.

safe_divide()'s zero-check delegates to math.tolerance.is_zero()
(confirmed elsewhere in this session to be actively used,
well-tested infrastructure) -- confirmed directly this correctly
rejects a near-zero (not exactly zero) denominator via that
function's own tolerance, not a naive `== 0` check.
"""

from __future__ import annotations

import pytest
from topocore.core.exceptions import MathError
from topocore.math.numeric import clamp, cube, lerp, mean, safe_divide, sign, square

# ----------------------------------------------------------------------
# clamp()
# ----------------------------------------------------------------------


def test_clamp_within_range_returns_value_unchanged() -> None:
    assert clamp(5, 0, 10) == 5


def test_clamp_below_minimum_returns_minimum() -> None:
    assert clamp(-5, 0, 10) == 0


def test_clamp_above_maximum_returns_maximum() -> None:
    assert clamp(15, 0, 10) == 10


def test_clamp_with_equal_min_and_max_returns_that_value() -> None:
    assert clamp(5, 3, 3) == 3


def test_clamp_rejects_minimum_greater_than_maximum() -> None:
    with pytest.raises(MathError, match="minimum cannot be greater than maximum"):
        clamp(5, 10, 0)


# ----------------------------------------------------------------------
# lerp()
# ----------------------------------------------------------------------


def test_lerp_midpoint() -> None:
    assert lerp(0, 10, 0.5) == 5.0


def test_lerp_at_t_zero_returns_start() -> None:
    assert lerp(0, 10, 0.0) == 0.0


def test_lerp_at_t_one_returns_end() -> None:
    assert lerp(0, 10, 1.0) == 10.0


def test_lerp_supports_extrapolation_beyond_zero_one() -> None:
    assert lerp(0, 10, 1.5) == 15.0
    assert lerp(0, 10, -0.5) == -5.0


# ----------------------------------------------------------------------
# safe_divide()
# ----------------------------------------------------------------------


def test_safe_divide_normal_case() -> None:
    assert safe_divide(10, 2) == 5.0


def test_safe_divide_rejects_exact_zero_denominator() -> None:
    with pytest.raises(MathError, match="Division by zero"):
        safe_divide(10, 0)


def test_safe_divide_rejects_near_zero_denominator_via_is_zero_tolerance() -> None:
    with pytest.raises(MathError, match="Division by zero"):
        safe_divide(1.0, 1e-15)


# ----------------------------------------------------------------------
# mean()
# ----------------------------------------------------------------------


def test_mean_of_several_values() -> None:
    assert mean([1, 2, 3, 4]) == 2.5


def test_mean_of_single_value() -> None:
    assert mean([7]) == 7.0


def test_mean_rejects_empty_sequence() -> None:
    with pytest.raises(MathError, match="empty sequence"):
        mean([])


# ----------------------------------------------------------------------
# cube(), square(), sign() -- defined but omitted from __all__.
# ----------------------------------------------------------------------


def test_cube() -> None:
    assert cube(3) == 27


def test_square() -> None:
    assert square(4) == 16


def test_sign_positive_negative_and_zero() -> None:
    assert sign(5) == 1.0
    assert sign(-5) == -1.0
    assert sign(0) == 0.0
