"""
Tests for VerticalCurve, implementing the Hickerson (1964) method.

Reference values are taken from two independently published worked
examples (PDHonline Course L121, "Vertical Curves", Jan Van Sickle),
NOT derived from this module's own formulas -- true independent
verification, not a self-consistency check.

    Symmetric:  PVC (station 1000, elev 100.00 ft), PVI (station
                1300, elev 106.00 ft), g1=+2%, g2=-3%, L1=L2=300 ft
                -> elevation at station 1250 = 102.40 ft

    Asymmetric: PVI (station 4831, elev 724.01 ft), g1=-4%, g2=+3%,
                length_in=431.00 ft, length_out=441.43 ft
                -> PVC elev 741.25 ft, PVT elev 737.25 ft (approx,
                   source rounds length_out),
                   CVC (curve elevation at pvi_station) = 731.64 ft
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from topocore.alignment.exceptions import AlignmentGeometryError
from topocore.alignment.vertical_elements import VerticalCurve


def _symmetric() -> VerticalCurve:
    return VerticalCurve(
        pvi_station=1300.0,
        pvi_elevation=106.0,
        incoming_grade=0.02,
        outgoing_grade=-0.03,
        length_in=300.0,
        length_out=300.0,
    )


def _asymmetric() -> VerticalCurve:
    return VerticalCurve(
        pvi_station=4831.0,
        pvi_elevation=724.01,
        incoming_grade=-0.04,
        outgoing_grade=0.03,
        length_in=431.0,
        length_out=441.43,
    )


# ----------------------------------------------------------------------
# Construction validation
# ----------------------------------------------------------------------


def test_rejects_non_positive_length_in() -> None:
    with pytest.raises(AlignmentGeometryError):
        VerticalCurve(
            pvi_station=0.0,
            pvi_elevation=0.0,
            incoming_grade=0.01,
            outgoing_grade=-0.01,
            length_in=0.0,
            length_out=100.0,
        )


def test_rejects_non_positive_length_out() -> None:
    with pytest.raises(AlignmentGeometryError):
        VerticalCurve(
            pvi_station=0.0,
            pvi_elevation=0.0,
            incoming_grade=0.01,
            outgoing_grade=-0.01,
            length_in=100.0,
            length_out=-5.0,
        )


# ----------------------------------------------------------------------
# Symmetric vs asymmetric classification
# ----------------------------------------------------------------------


def test_symmetric_curve_is_symmetric() -> None:
    assert _symmetric().is_symmetric is True


def test_asymmetric_curve_is_not_symmetric() -> None:
    assert _asymmetric().is_symmetric is False


# ----------------------------------------------------------------------
# PVC/PVT stations and elevations (straight tangent lines, exact in
# both models -- unaffected by the Hickerson correction)
# ----------------------------------------------------------------------


def test_symmetric_pvc_pvt_stations() -> None:
    curve = _symmetric()
    assert curve.pvc_station == pytest.approx(1000.0)
    assert curve.pvt_station == pytest.approx(1600.0)


def test_symmetric_pvc_pvt_elevations_match_worked_example() -> None:
    curve = _symmetric()
    assert curve.pvc_elevation == pytest.approx(100.0)
    assert curve.pvt_elevation == pytest.approx(97.0)


def test_asymmetric_pvc_pvt_elevations_match_worked_example() -> None:
    curve = _asymmetric()
    assert curve.pvc_elevation == pytest.approx(741.25)
    assert curve.pvt_elevation == pytest.approx(737.2529, abs=1e-3)  # source rounds L2


# ----------------------------------------------------------------------
# PVI is generally OFF the curve -- the core correction. CVC (curve
# elevation at pvi_station) must NOT equal pvi_elevation in general.
# ----------------------------------------------------------------------


def test_symmetric_cvc_elevation_matches_worked_example() -> None:
    """
    Independently published reference: elevation at the midpoint of
    the curve = 102.25 ft, NOT the PVI's own 106.00 ft.
    """
    curve = _symmetric()
    assert curve.cvc_elevation == pytest.approx(102.25)
    assert curve.cvc_elevation != pytest.approx(curve.pvi_elevation)


def test_asymmetric_cvc_elevation_matches_worked_example() -> None:
    """
    Independently published reference: CVC elevation = 731.64 ft,
    NOT the PVI's own 724.01 ft.
    """
    curve = _asymmetric()
    assert curve.cvc_elevation == pytest.approx(731.64, abs=1e-2)
    assert curve.cvc_elevation != pytest.approx(curve.pvi_elevation)


def test_symmetric_elevation_at_pvi_station_matches_cvc_not_pvi() -> None:
    curve = _symmetric()
    assert curve.elevation_at(curve.pvi_station) == pytest.approx(curve.cvc_elevation)
    assert curve.elevation_at(curve.pvi_station) != pytest.approx(curve.pvi_elevation)


# ----------------------------------------------------------------------
# elevation_at() at an interior station (Method B / Method A worked
# example): station 1250, expected 102.40 ft.
# ----------------------------------------------------------------------


def test_symmetric_elevation_at_interior_station_matches_worked_example() -> None:
    curve = _symmetric()
    assert curve.elevation_at(1250.0) == pytest.approx(102.4, abs=1e-2)


# ----------------------------------------------------------------------
# mid_grade: direct-weighted average (Hickerson), reduces to
# (g1+g2)/2 for the symmetric case.
# ----------------------------------------------------------------------


def test_symmetric_mid_grade_is_average_of_endpoint_grades() -> None:
    curve = _symmetric()
    assert curve.mid_grade == pytest.approx((0.02 + -0.03) / 2.0)


def test_asymmetric_mid_grade_matches_direct_weighted_formula() -> None:
    curve = _asymmetric()

    # mid_grade = (g1*L1 + g2*L2) / L -- computed independently of
    # curve.mid_grade's own implementation, direct (not inverse)
    # weighting per Hickerson.
    expected = (curve.incoming_grade * curve.length_in + curve.outgoing_grade * curve.length_out) / curve.length

    assert curve.mid_grade == pytest.approx(expected)
    assert curve.mid_grade == pytest.approx(-0.0045815710142934115, abs=1e-9)


# ----------------------------------------------------------------------
# Grade continuity at PVC/PVT/CVC (slope IS continuous throughout,
# even though curvature/rate is not)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("build", [_symmetric, _asymmetric])
def test_grade_at_pvc_matches_incoming_grade(
    build: Callable[[], VerticalCurve],
) -> None:
    curve = build()
    assert curve.grade_at(curve.pvc_station) == pytest.approx(curve.incoming_grade)


@pytest.mark.parametrize("build", [_symmetric, _asymmetric])
def test_grade_at_pvt_matches_outgoing_grade(
    build: Callable[[], VerticalCurve],
) -> None:
    curve = build()
    assert curve.grade_at(curve.pvt_station) == pytest.approx(curve.outgoing_grade)


@pytest.mark.parametrize("build", [_symmetric, _asymmetric])
def test_grade_at_pvi_station_matches_mid_grade_from_both_sides(
    build: Callable[[], VerticalCurve],
) -> None:
    """
    Slope is continuous at CVC -- approaching from either the first
    or second component parabola gives the same mid_grade.
    """
    curve = build()
    epsilon = 1e-6

    from_first_half = curve.grade_at(curve.pvi_station - epsilon)
    from_second_half = curve.grade_at(curve.pvi_station + epsilon)

    assert from_first_half == pytest.approx(curve.mid_grade, abs=1e-4)
    assert from_second_half == pytest.approx(curve.mid_grade, abs=1e-4)


# ----------------------------------------------------------------------
# Curvature (rate) IS discontinuous at CVC for the asymmetric case --
# a documented, accepted property of the Hickerson method, not a bug.
# Symmetric case: rate_in == rate_out (continuous).
# ----------------------------------------------------------------------


def test_symmetric_rate_in_equals_rate_out() -> None:
    curve = _symmetric()
    assert curve.rate_in == pytest.approx(curve.rate_out)


def test_asymmetric_rate_in_differs_from_rate_out() -> None:
    curve = _asymmetric()
    assert curve.rate_in != pytest.approx(curve.rate_out)


def test_asymmetric_rate_ratio_matches_squared_length_ratio() -> None:
    """
    Classic Hickerson property: rate_in / rate_out = (length_out / length_in) ** 2.
    """
    curve = _asymmetric()

    ratio = curve.rate_in / curve.rate_out
    expected_ratio = (curve.length_out / curve.length_in) ** 2

    assert ratio == pytest.approx(expected_ratio)


# ----------------------------------------------------------------------
# Elevation continuity at PVT: the second component parabola,
# evaluated at its own far end, must reproduce pvt_elevation exactly.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("build", [_symmetric, _asymmetric])
def test_elevation_at_pvt_matches_pvt_elevation(
    build: Callable[[], VerticalCurve],
) -> None:
    curve = build()
    assert curve.elevation_at(curve.pvt_station) == pytest.approx(curve.pvt_elevation)


@pytest.mark.parametrize("build", [_symmetric, _asymmetric])
def test_elevation_at_pvc_matches_pvc_elevation(
    build: Callable[[], VerticalCurve],
) -> None:
    curve = build()
    assert curve.elevation_at(curve.pvc_station) == pytest.approx(curve.pvc_elevation)
