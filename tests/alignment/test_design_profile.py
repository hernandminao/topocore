"""
Tests for GradeSegment and DesignProfile, including station,
elevation, AND grade (slope) continuity between consecutive
elements -- per Hernán's explicit requirement.
"""

from __future__ import annotations

import pytest

from topocore.alignment.exceptions import (
    AlignmentError,
    AlignmentGeometryError,
    AlignmentStationError,
)
from topocore.alignment.models import DesignProfile
from topocore.alignment.vertical_elements import GradeSegment, VerticalCurve

# ----------------------------------------------------------------------
# GradeSegment
# ----------------------------------------------------------------------


def test_grade_segment_rejects_non_positive_length() -> None:
    with pytest.raises(AlignmentGeometryError):
        GradeSegment(
            start_station=100.0,
            end_station=100.0,
            start_elevation=0.0,
            end_elevation=0.0,
        )


def test_grade_segment_computes_grade_percent() -> None:
    segment = GradeSegment(start_station=0.0, end_station=100.0, start_elevation=0.0, end_elevation=5.0)

    assert segment.grade == pytest.approx(0.05)
    assert segment.grade_percent == pytest.approx(5.0)


def test_grade_segment_elevation_at_mid() -> None:
    segment = GradeSegment(start_station=0.0, end_station=100.0, start_elevation=10.0, end_elevation=20.0)

    assert segment.elevation_at(50.0) == pytest.approx(15.0)


def test_grade_segment_grade_at_is_constant() -> None:
    segment = GradeSegment(start_station=0.0, end_station=100.0, start_elevation=10.0, end_elevation=20.0)

    assert segment.grade_at(0.0) == pytest.approx(segment.grade)
    assert segment.grade_at(100.0) == pytest.approx(segment.grade)
    assert segment.grade_at(37.0) == pytest.approx(segment.grade)


# ----------------------------------------------------------------------
# DesignProfile construction
# ----------------------------------------------------------------------


def test_design_profile_rejects_blank_name() -> None:
    segment = GradeSegment(start_station=0.0, end_station=100.0, start_elevation=0.0, end_elevation=1.0)

    with pytest.raises(AlignmentError):
        DesignProfile(alignment_name="  ", elements=(segment,))


def test_design_profile_rejects_empty_elements() -> None:
    with pytest.raises(AlignmentError):
        DesignProfile(alignment_name="A", elements=())


def test_design_profile_rejects_station_gap() -> None:
    seg1 = GradeSegment(start_station=0.0, end_station=100.0, start_elevation=0.0, end_elevation=3.0)
    seg2 = GradeSegment(start_station=105.0, end_station=200.0, start_elevation=3.0, end_elevation=1.0)

    with pytest.raises(AlignmentGeometryError, match="station"):
        DesignProfile(alignment_name="A", elements=(seg1, seg2))


def test_design_profile_rejects_elevation_discontinuity() -> None:
    seg1 = GradeSegment(start_station=0.0, end_station=100.0, start_elevation=0.0, end_elevation=3.0)
    seg2 = GradeSegment(start_station=100.0, end_station=200.0, start_elevation=99.0, end_elevation=1.0)

    with pytest.raises(AlignmentGeometryError, match="elevation"):
        DesignProfile(alignment_name="A", elements=(seg1, seg2))


def test_design_profile_rejects_grade_discontinuity_between_grade_segments() -> None:
    """
    Two grade segments that meet at the same station and elevation
    (C0-continuous) but with different slopes (C1-discontinuous) --
    a real kink, must be rejected.
    """
    seg1 = GradeSegment(start_station=0.0, end_station=100.0, start_elevation=0.0, end_elevation=3.0)  # 3% grade
    seg2 = GradeSegment(start_station=100.0, end_station=200.0, start_elevation=3.0, end_elevation=13.0)  # 10% grade

    with pytest.raises(AlignmentGeometryError, match="slope"):
        DesignProfile(alignment_name="A", elements=(seg1, seg2))


def test_design_profile_accepts_continuous_grade_segments() -> None:
    seg1 = GradeSegment(start_station=0.0, end_station=100.0, start_elevation=0.0, end_elevation=3.0)
    seg2 = GradeSegment(start_station=100.0, end_station=200.0, start_elevation=3.0, end_elevation=6.0)  # same 3%

    DesignProfile(alignment_name="A", elements=(seg1, seg2))  # must not raise


def test_design_profile_rejects_grade_discontinuity_at_curve_boundary() -> None:
    """
    A grade segment that does NOT match a following VerticalCurve's
    incoming_grade -- a slope kink right at the PVC -- must be
    rejected, exactly as it would be between two GradeSegments.
    """
    curve = VerticalCurve(
        pvi_station=1000.0,
        pvi_elevation=50.0,
        incoming_grade=0.03,
        outgoing_grade=-0.02,
        length_in=100.0,
        length_out=100.0,
    )
    # Leading grade segment ends at the curve's PVC station/elevation,
    # but with the WRONG slope (5%, not the curve's incoming_grade 3%).
    wrong_slope_segment = GradeSegment(
        start_station=800.0,
        end_station=curve.pvc_station,
        start_elevation=curve.pvc_elevation - 0.05 * (curve.pvc_station - 800.0),
        end_elevation=curve.pvc_elevation,
    )

    with pytest.raises(AlignmentGeometryError, match="slope"):
        DesignProfile(alignment_name="A", elements=(wrong_slope_segment, curve))


def test_design_profile_accepts_correctly_matched_grade_then_curve() -> None:
    curve = VerticalCurve(
        pvi_station=1000.0,
        pvi_elevation=50.0,
        incoming_grade=0.03,
        outgoing_grade=-0.02,
        length_in=100.0,
        length_out=100.0,
    )
    leading_segment = GradeSegment(
        start_station=800.0,
        end_station=curve.pvc_station,
        start_elevation=curve.pvc_elevation - curve.incoming_grade * (curve.pvc_station - 800.0),
        end_elevation=curve.pvc_elevation,
    )
    trailing_segment = GradeSegment(
        start_station=curve.pvt_station,
        end_station=curve.pvt_station + 150.0,
        start_elevation=curve.pvt_elevation,
        end_elevation=curve.pvt_elevation + curve.outgoing_grade * 150.0,
    )

    DesignProfile(alignment_name="A", elements=(leading_segment, curve, trailing_segment))  # must not raise


# ----------------------------------------------------------------------
# DesignProfile.elevation_at()
# ----------------------------------------------------------------------


def test_design_profile_elevation_at_dispatches_to_grade_segment() -> None:
    seg1 = GradeSegment(start_station=0.0, end_station=100.0, start_elevation=0.0, end_elevation=10.0)
    profile = DesignProfile(alignment_name="A", elements=(seg1,))

    assert profile.elevation_at(50.0) == pytest.approx(5.0)


def test_design_profile_elevation_at_dispatches_to_vertical_curve() -> None:
    curve = VerticalCurve(
        pvi_station=1000.0,
        pvi_elevation=50.0,
        incoming_grade=0.03,
        outgoing_grade=-0.02,
        length_in=100.0,
        length_out=100.0,
    )
    profile = DesignProfile(alignment_name="A", elements=(curve,))

    assert profile.elevation_at(1000.0) == pytest.approx(curve.cvc_elevation)
    assert profile.elevation_at(curve.pvc_station) == pytest.approx(curve.pvc_elevation)
    assert profile.elevation_at(curve.pvt_station) == pytest.approx(curve.pvt_elevation)


def test_design_profile_elevation_at_out_of_range_raises() -> None:
    seg1 = GradeSegment(start_station=0.0, end_station=100.0, start_elevation=0.0, end_elevation=10.0)
    profile = DesignProfile(alignment_name="A", elements=(seg1,))

    with pytest.raises(AlignmentStationError):
        profile.elevation_at(150.0)


def test_design_profile_elevation_at_multi_element_chain() -> None:
    curve = VerticalCurve(
        pvi_station=1000.0,
        pvi_elevation=50.0,
        incoming_grade=0.03,
        outgoing_grade=-0.02,
        length_in=100.0,
        length_out=100.0,
    )
    leading_segment = GradeSegment(
        start_station=800.0,
        end_station=curve.pvc_station,
        start_elevation=curve.pvc_elevation - curve.incoming_grade * (curve.pvc_station - 800.0),
        end_elevation=curve.pvc_elevation,
    )
    trailing_segment = GradeSegment(
        start_station=curve.pvt_station,
        end_station=curve.pvt_station + 150.0,
        start_elevation=curve.pvt_elevation,
        end_elevation=curve.pvt_elevation + curve.outgoing_grade * 150.0,
    )
    profile = DesignProfile(alignment_name="A", elements=(leading_segment, curve, trailing_segment))

    # Query a point in each of the three elements.
    assert profile.elevation_at(850.0) == pytest.approx(leading_segment.elevation_at(850.0))
    assert profile.elevation_at(1000.0) == pytest.approx(curve.cvc_elevation)
    assert profile.elevation_at(curve.pvt_station + 50.0) == pytest.approx(
        trailing_segment.elevation_at(curve.pvt_station + 50.0)
    )
