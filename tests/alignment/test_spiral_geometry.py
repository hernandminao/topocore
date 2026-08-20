"""
Geometry tests for SpiralElement, per the Entrega 2 checklist:

    l=0 matches start; l=L matches end; A**2 = R*L; curvature varies
    correctly along the spiral; radius_start/radius_end are
    respected; both clockwise=True and clockwise=False; both
    directions; numerical Fresnel verification.

Reference values below (u/v at l=0,10,25,40,50 for R=100, L=50) were
computed independently via scipy.special.fresnel directly in this
file's companion analysis, NOT by calling
topocore.alignment.algorithms.spiral -- they are hardcoded literals,
not re-derived from the code under test.
"""

from __future__ import annotations

import math

import pytest

from topocore.alignment.algorithms.spiral import curvature_at, evaluate_spiral
from topocore.alignment.elements import SpiralElement
from topocore.geometry.point2d import Point2D

R = 100.0
L = 50.0

# Independently computed via scipy.special.fresnel (see session notes).
_REFERENCE_UV: dict[float, tuple[float, float]] = {
    0.0: (0.0, 0.0),
    10.0: (9.999900000462965, 0.033333095238852826),
    25.0: (24.990236140904354, 0.5206880295935199),
    40.0: (39.89772129129052, 2.1294355570355172),
    50.0: (49.68840292147947, 4.148102426854749),
}


def _entry_ccw() -> SpiralElement:
    """
    Axis-aligned (rotation=0) entry spiral: zero-curvature point at
    the global origin, tangent along +x, counterclockwise-bending.
    Because rotation=0 and translation=0, global coordinates equal
    the canonical local (u, v) coordinates directly -- the simplest
    possible case to check against the independent reference values.
    """
    end_u, end_v = _REFERENCE_UV[L]
    return SpiralElement(
        start=Point2D(0.0, 0.0),
        end=Point2D(end_u, end_v),
        pi=Point2D(33.44311735685825, 0.0),
        radius_start=math.inf,
        radius_end=R,
        length=L,
        clockwise=False,
    )


def _entry_cw() -> SpiralElement:
    end_u, end_v = _REFERENCE_UV[L]
    return SpiralElement(
        start=Point2D(0.0, 0.0),
        end=Point2D(end_u, -end_v),  # mirrored v for clockwise
        pi=Point2D(33.44311735685825, 0.0),
        radius_start=math.inf,
        radius_end=R,
        length=L,
        clockwise=True,
    )


def _exit_ccw() -> SpiralElement:
    """
    Mirror of _entry_ccw(): zero-curvature point at the global
    origin is now `end`, not `start`.

    PI is NOT simply (33.44311735685825, 0.0) -- the same value used
    for the entry-spiral fixtures -- despite this being "the same
    physical curve, reversed". Confirmed (PR18C session, against a
    real PLATEIA export) that PI computation is genuinely different
    between entry and exit spirals: l' increases toward `start` for
    an exit spiral, i.e. OPPOSITE to physical travel, so the local
    frame's mirroring (and therefore the derived rotation and PI)
    differs from the entry case. This value was computed
    independently via the corrected formula, not copied from the
    entry fixture.
    """
    finite_u, finite_v = _REFERENCE_UV[L]
    return SpiralElement(
        start=Point2D(finite_u, finite_v),
        end=Point2D(0.0, 0.0),
        pi=Point2D(32.98019368655387, 5.545171137303717),
        radius_start=R,
        radius_end=math.inf,
        length=L,
        clockwise=False,
    )


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------


def test_entry_spiral_l0_matches_start() -> None:
    spiral = _entry_ccw()
    point = evaluate_spiral(spiral, 0.0)

    assert point.x == pytest.approx(spiral.start.x, abs=1e-9)
    assert point.y == pytest.approx(spiral.start.y, abs=1e-9)


def test_entry_spiral_lL_matches_end() -> None:
    spiral = _entry_ccw()
    point = evaluate_spiral(spiral, spiral.length)

    assert point.x == pytest.approx(spiral.end.x, abs=1e-6)
    assert point.y == pytest.approx(spiral.end.y, abs=1e-6)


def test_exit_spiral_l0_matches_start() -> None:
    spiral = _exit_ccw()
    point = evaluate_spiral(spiral, 0.0)

    assert point.x == pytest.approx(spiral.start.x, abs=1e-6)
    assert point.y == pytest.approx(spiral.start.y, abs=1e-6)


def test_exit_spiral_lL_matches_end() -> None:
    spiral = _exit_ccw()
    point = evaluate_spiral(spiral, spiral.length)

    assert point.x == pytest.approx(spiral.end.x, abs=1e-9)
    assert point.y == pytest.approx(spiral.end.y, abs=1e-9)


# ----------------------------------------------------------------------
# Independent Fresnel reference values at interior points
# ----------------------------------------------------------------------


@pytest.mark.parametrize("length", [0.0, 10.0, 25.0, 40.0, 50.0])
def test_entry_spiral_matches_independent_fresnel_reference(length: float) -> None:
    spiral = _entry_ccw()
    point = evaluate_spiral(spiral, length)

    expected_u, expected_v = _REFERENCE_UV[length]

    # Axis-aligned construction: global (x, y) == local (u, v).
    assert point.x == pytest.approx(expected_u, abs=1e-6)
    assert point.y == pytest.approx(expected_v, abs=1e-6)


@pytest.mark.parametrize("length", [0.0, 10.0, 25.0, 40.0, 50.0])
def test_clockwise_spiral_mirrors_v_against_reference(length: float) -> None:
    spiral = _entry_cw()
    point = evaluate_spiral(spiral, length)

    expected_u, expected_v = _REFERENCE_UV[length]

    assert point.x == pytest.approx(expected_u, abs=1e-6)
    assert point.y == pytest.approx(-expected_v, abs=1e-6)


# ----------------------------------------------------------------------
# A**2 = R * L
# ----------------------------------------------------------------------


def test_clothoid_constant_squared_equals_r_times_l() -> None:
    spiral = _entry_ccw()

    # At l=L, curvature magnitude must be exactly 1/R -- which only
    # holds if A**2 = R*L was used consistently throughout.
    curvature_at_end = curvature_at(spiral, spiral.length)

    assert abs(curvature_at_end) == pytest.approx(1.0 / R)


# ----------------------------------------------------------------------
# Curvature behavior
# ----------------------------------------------------------------------


def test_entry_spiral_curvature_zero_at_start() -> None:
    spiral = _entry_ccw()
    assert curvature_at(spiral, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_entry_spiral_curvature_matches_1_over_r_at_end() -> None:
    spiral = _entry_ccw()
    assert curvature_at(spiral, spiral.length) == pytest.approx(1.0 / R)  # CCW -> positive sign


def test_exit_spiral_curvature_matches_1_over_r_at_start() -> None:
    spiral = _exit_ccw()
    assert curvature_at(spiral, 0.0) == pytest.approx(1.0 / R)


def test_exit_spiral_curvature_zero_at_end() -> None:
    spiral = _exit_ccw()
    assert curvature_at(spiral, spiral.length) == pytest.approx(0.0, abs=1e-12)


def test_curvature_varies_linearly_along_entry_spiral() -> None:
    spiral = _entry_ccw()

    samples = [0.0, 12.5, 25.0, 37.5, 50.0]
    curvatures = [abs(curvature_at(spiral, length)) for length in samples]

    for length, k in zip(samples, curvatures, strict=True):
        assert k == pytest.approx(length / (R * L), abs=1e-9)


def test_curvature_matches_finite_difference_estimate() -> None:
    """
    Independent numerical cross-check: estimate curvature from three
    closely-spaced points on the actual evaluated curve (discrete
    curvature via the turning-angle-over-arc-length approximation),
    and compare against the analytic curvature_at() value at the
    midpoint. This exercises the full local->global transform, not
    just the analytic formula in isolation.
    """
    spiral = _entry_ccw()

    l_mid = 25.0
    h = 0.05

    p0 = evaluate_spiral(spiral, l_mid - h)
    p1 = evaluate_spiral(spiral, l_mid)
    p2 = evaluate_spiral(spiral, l_mid + h)

    v1 = (p1.x - p0.x, p1.y - p0.y)
    v2 = (p2.x - p1.x, p2.y - p1.y)

    heading1 = math.atan2(v1[1], v1[0])
    heading2 = math.atan2(v2[1], v2[0])

    # Turning angle over arc length traveled ~ curvature.
    delta_heading = heading2 - heading1
    numeric_curvature = delta_heading / h

    analytic_curvature = curvature_at(spiral, l_mid)

    assert numeric_curvature == pytest.approx(analytic_curvature, rel=1e-2)


# ----------------------------------------------------------------------
# Rigid-transform equivariance: rotating/translating the whole
# spiral must rotate/translate every evaluated point identically.
# ----------------------------------------------------------------------


def test_rotated_translated_spiral_is_consistent_with_axis_aligned_reference() -> None:
    reference = _entry_ccw()

    angle = math.radians(37.0)  # arbitrary, non-axis-aligned
    offset = Point2D(500.0, -200.0)

    def rotate_translate(point: Point2D) -> Point2D:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return Point2D(
            offset.x + point.x * cos_a - point.y * sin_a,
            offset.y + point.x * sin_a + point.y * cos_a,
        )

    rotated = SpiralElement(
        start=rotate_translate(reference.start),
        end=rotate_translate(reference.end),
        pi=rotate_translate(reference.pi),
        radius_start=reference.radius_start,
        radius_end=reference.radius_end,
        length=reference.length,
        clockwise=reference.clockwise,
    )

    for length in [0.0, 10.0, 25.0, 40.0, 50.0]:
        reference_point = evaluate_spiral(reference, length)
        rotated_point = evaluate_spiral(rotated, length)
        expected_point = rotate_translate(reference_point)

        assert rotated_point.x == pytest.approx(expected_point.x, abs=1e-6)
        assert rotated_point.y == pytest.approx(expected_point.y, abs=1e-6)
