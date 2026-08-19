"""
Round-trip tests for LandXML <Alignments> (<CoordGeom> horizontal
geometry + <Profile>/<ProfAlign> vertical geometry), integrating the
already-closed topocore.alignment domain.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from topocore.alignment.elements import ArcElement, LineElement, SpiralElement
from topocore.alignment.models import Alignment, DesignProfile
from topocore.alignment.vertical_elements import GradeSegment, VerticalCurve
from topocore.geometry.point2d import Point2D
from topocore.io.landxml.exceptions import LandXMLValidationError
from topocore.io.landxml.models import LandXMLDocument, NamedAlignment
from topocore.io.landxml.reader import LandXMLReader
from topocore.io.landxml.writer import LandXMLWriter


def _sample_alignment() -> Alignment:
    line = LineElement(Point2D(0.0, 0.0), Point2D(100.0, 0.0))

    arc = ArcElement(
        start=Point2D(100.0, 0.0),
        end=Point2D(150.0, 50.0),
        center=Point2D(100.0, 50.0),
        radius=50.0,
        clockwise=False,
    )

    # Same R=100, L=50 entry spiral verified in the alignment Entrega 2
    # test suite, anchored at the arc's end point, axis-aligned.
    spiral = SpiralElement(
        start=Point2D(150.0, 50.0),
        end=Point2D(199.68840292147948, 54.14810242685475),
        pi=Point2D(183.44311735685824, 50.0),
        radius_start=math.inf,
        radius_end=100.0,
        length=50.0,
        clockwise=False,
    )

    return Alignment(name="Centerline", elements=(line, arc, spiral), start_station=0.0)


def _sample_profile(alignment: Alignment) -> DesignProfile:
    curve1 = VerticalCurve(
        pvi_station=100.0,
        pvi_elevation=50.0,
        incoming_grade=0.03,
        outgoing_grade=-0.02,
        length_in=40.0,
        length_out=40.0,
    )
    leading = GradeSegment(
        start_station=0.0,
        end_station=curve1.pvc_station,
        start_elevation=curve1.pvc_elevation - 0.03 * (curve1.pvc_station - 0.0),
        end_elevation=curve1.pvc_elevation,
    )
    middle = GradeSegment(
        start_station=curve1.pvt_station,
        end_station=curve1.pvt_station + 100.0,
        start_elevation=curve1.pvt_elevation,
        end_elevation=curve1.pvt_elevation + curve1.outgoing_grade * 100.0,
    )
    curve2_pvi_station = middle.end_station + 60.0
    curve2 = VerticalCurve(
        pvi_station=curve2_pvi_station,
        pvi_elevation=middle.end_elevation + curve1.outgoing_grade * 60.0,
        incoming_grade=curve1.outgoing_grade,
        outgoing_grade=0.025,
        length_in=50.0,
        length_out=30.0,
    )
    bridge = GradeSegment(
        start_station=middle.end_station,
        end_station=curve2.pvc_station,
        start_elevation=middle.end_elevation,
        end_elevation=curve2.pvc_elevation,
    )
    trailing = GradeSegment(
        start_station=curve2.pvt_station,
        end_station=curve2.pvt_station + 80.0,
        start_elevation=curve2.pvt_elevation,
        end_elevation=curve2.pvt_elevation + 0.025 * 80.0,
    )

    return DesignProfile(
        alignment_name=alignment.name,
        elements=(leading, curve1, middle, bridge, curve2, trailing),
    )


@pytest.fixture
def sample_document() -> LandXMLDocument:
    alignment = _sample_alignment()
    profile = _sample_profile(alignment)
    named = NamedAlignment(name="Centerline", alignment=alignment, profile=profile, desc="Eje principal")
    return LandXMLDocument(alignments=(named,))


def test_roundtrip_alignment_name_and_desc(sample_document: LandXMLDocument, tmp_path: Path) -> None:
    path = tmp_path / "alignment.xml"
    LandXMLWriter(path).write(sample_document)
    result = LandXMLReader(path).read()

    assert len(result.alignments) == 1
    assert result.alignments[0].name == "Centerline"
    assert result.alignments[0].desc == "Eje principal"


def test_roundtrip_preserves_horizontal_element_count_and_type(
    sample_document: LandXMLDocument, tmp_path: Path
) -> None:
    path = tmp_path / "alignment.xml"
    LandXMLWriter(path).write(sample_document)
    result = LandXMLReader(path).read()

    elements = result.alignments[0].alignment.elements
    assert len(elements) == 3
    assert isinstance(elements[0], LineElement)
    assert isinstance(elements[1], ArcElement)
    assert isinstance(elements[2], SpiralElement)


def test_roundtrip_preserves_horizontal_geometry_exactly(sample_document: LandXMLDocument, tmp_path: Path) -> None:
    path = tmp_path / "alignment.xml"
    LandXMLWriter(path).write(sample_document)
    result = LandXMLReader(path).read()

    original = sample_document.alignments[0].alignment
    round_tripped = result.alignments[0].alignment

    assert round_tripped.length == pytest.approx(original.length)

    for station in [0.0, 50.0, 100.0, original.length]:
        original_point = original.station_to_point(station)
        round_tripped_point = round_tripped.station_to_point(station)

        assert round_tripped_point.x == pytest.approx(original_point.x, abs=1e-6)
        assert round_tripped_point.y == pytest.approx(original_point.y, abs=1e-6)


def test_roundtrip_preserves_profile_element_count_and_type(sample_document: LandXMLDocument, tmp_path: Path) -> None:
    path = tmp_path / "alignment.xml"
    LandXMLWriter(path).write(sample_document)
    result = LandXMLReader(path).read()

    profile = result.alignments[0].profile
    assert profile is not None
    assert len(profile.elements) == 6

    types = [type(element).__name__ for element in profile.elements]
    assert types == [
        "GradeSegment",
        "VerticalCurve",
        "GradeSegment",
        "GradeSegment",
        "VerticalCurve",
        "GradeSegment",
    ]


def test_roundtrip_preserves_symmetric_and_asymmetric_curves(sample_document: LandXMLDocument, tmp_path: Path) -> None:
    path = tmp_path / "alignment.xml"
    LandXMLWriter(path).write(sample_document)
    result = LandXMLReader(path).read()

    profile = result.alignments[0].profile
    assert profile is not None

    curve1 = profile.elements[1]
    curve2 = profile.elements[4]
    assert isinstance(curve1, VerticalCurve)
    assert isinstance(curve2, VerticalCurve)

    assert curve1.is_symmetric is True
    assert curve2.is_symmetric is False

    original_profile = sample_document.alignments[0].profile
    assert original_profile is not None
    original_curve1 = original_profile.elements[1]
    original_curve2 = original_profile.elements[4]
    assert isinstance(original_curve1, VerticalCurve)
    assert isinstance(original_curve2, VerticalCurve)

    assert curve1.pvi_station == pytest.approx(original_curve1.pvi_station)
    assert curve1.pvi_elevation == pytest.approx(original_curve1.pvi_elevation)
    assert curve1.length_in == pytest.approx(original_curve1.length_in)
    assert curve1.length_out == pytest.approx(original_curve1.length_in)

    assert curve2.length_in == pytest.approx(original_curve2.length_in)
    assert curve2.length_out == pytest.approx(original_curve2.length_out)
    assert curve2.length_in != pytest.approx(curve2.length_out)


def test_roundtrip_preserves_design_elevation_at_sample_stations(
    sample_document: LandXMLDocument, tmp_path: Path
) -> None:
    path = tmp_path / "alignment.xml"
    LandXMLWriter(path).write(sample_document)
    result = LandXMLReader(path).read()

    original_profile = sample_document.alignments[0].profile
    round_tripped_profile = result.alignments[0].profile
    assert original_profile is not None
    assert round_tripped_profile is not None

    for station in [
        10.0,
        100.0,
        200.0,
        300.0,
        400.0,
        round_tripped_profile.end_station,
    ]:
        original_elev = original_profile.elevation_at(station)
        round_tripped_elev = round_tripped_profile.elevation_at(station)
        assert round_tripped_elev == pytest.approx(original_elev, abs=1e-6)


def test_write_returns_alignment_count(sample_document: LandXMLDocument, tmp_path: Path) -> None:
    report = LandXMLWriter(tmp_path / "alignment.xml").write(sample_document)
    assert report.alignment_count == 1


def test_read_with_report_returns_alignment_count(sample_document: LandXMLDocument, tmp_path: Path) -> None:
    path = tmp_path / "alignment.xml"
    LandXMLWriter(path).write(sample_document)
    _, report = LandXMLReader(path).read_with_report()
    assert report.alignment_count == 1


def test_write_rejects_duplicated_alignment_names(tmp_path: Path) -> None:
    alignment = _sample_alignment()
    doc = LandXMLDocument(
        alignments=(
            NamedAlignment(name="Dup", alignment=alignment),
            NamedAlignment(name="Dup", alignment=alignment),
        )
    )

    with pytest.raises(LandXMLValidationError):
        LandXMLWriter(tmp_path / "dup.xml").write(doc)
