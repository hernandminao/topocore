"""
End-to-end round-trip regression suite for
topocore.io.landxml.writer.LandXMLWriter and .reader.LandXMLReader
-- PR19.

Verifies write -> read round trips for the three document content
types LandXMLDocument supports: surfaces (TIN), point groups
(CgPoints, with id/code/order preserved), and alignments (horizontal
Line/Arc geometry, with radius and turn direction preserved). Also
verifies duplicate-name validation. No bugs found in this pass (this
module was already extensively audited in PR18B/PR18C; this PR19
pass re-confirms correctness with fresh, independent round-trip
checks rather than assuming prior work covers everything).
"""

from __future__ import annotations

import math

import pytest

from topocore.alignment.elements import ArcElement, LineElement
from topocore.alignment.models import Alignment
from topocore.geometry.point2d import Point2D
from topocore.geometry.point3d import Point3D
from topocore.io.landxml.exceptions import LandXMLValidationError
from topocore.io.landxml.models import (
    LandXMLDocument,
    NamedAlignment,
    NamedPointGroup,
    NamedSurface,
)
from topocore.io.landxml.reader import LandXMLReader
from topocore.io.landxml.writer import LandXMLWriter
from topocore.survey.models import SurveyPoint, SurveyPointSet
from topocore.terrain.tin import TIN


def test_surface_round_trip_preserves_vertices(tmp_path) -> None:  # type: ignore[no-untyped-def]
    points = (
        Point3D(775431.17, 2603215.79, 62.9),
        Point3D(775441.17, 2603215.79, 65.2),
        Point3D(775431.17, 2603225.79, 60.1),
    )
    tin = TIN.from_points(points)
    doc = LandXMLDocument(surfaces=(NamedSurface(name="TestSurface", tin=tin),))

    path = str(tmp_path / "surface.xml")
    LandXMLWriter(path).write(doc)
    result = LandXMLReader(path).read()

    surface = result.surfaces[0]
    assert surface.name == "TestSurface"
    assert surface.tin.triangle_count == 1

    recovered = {tuple(round(c, 2) for c in (v.x, v.y, v.z)) for v in surface.tin.triangle_vertices(0)}
    original = {(p.x, p.y, p.z) for p in points}
    assert recovered == original


def test_point_group_round_trip_preserves_order_id_and_code(tmp_path) -> None:  # type: ignore[no-untyped-def]
    points = SurveyPointSet(
        points=(
            SurveyPoint(id="P1", x=100.0, y=200.0, z=10.5, code="CERCA"),
            SurveyPoint(id="P2", x=101.0, y=201.0, z=11.5, code="ARBOL"),
            SurveyPoint(id="P3", x=102.0, y=202.0, z=12.5, code=None),
        )
    )
    doc = LandXMLDocument(point_groups=(NamedPointGroup(name="TestPoints", points=points),))

    path = str(tmp_path / "points.xml")
    LandXMLWriter(path).write(doc)
    result = LandXMLReader(path).read()

    read_points = result.point_groups[0].points
    assert len(read_points) == 3
    for original, read in zip(points, read_points, strict=True):
        assert read.id == original.id
        assert read.x == pytest.approx(original.x)
        assert read.y == pytest.approx(original.y)
        assert read.z == pytest.approx(original.z)
        assert read.code == original.code


def test_alignment_round_trip_preserves_line_and_arc_geometry(tmp_path) -> None:  # type: ignore[no-untyped-def]
    line = LineElement(start=Point2D(0.0, 0.0), end=Point2D(100.0, 0.0))

    center = Point2D(100.0, 200.0)
    end = Point2D(center.x + 200.0 * math.cos(0), center.y + 200.0 * math.sin(0))
    arc = ArcElement(start=Point2D(100.0, 0.0), end=end, center=center, radius=200.0, clockwise=False)

    alignment = Alignment(name="TestAlign", elements=(line, arc))
    doc = LandXMLDocument(alignments=(NamedAlignment(name="TestAlign", alignment=alignment),))

    path = str(tmp_path / "alignment.xml")
    LandXMLWriter(path).write(doc)
    result = LandXMLReader(path).read()

    read_elements = result.alignments[0].alignment.elements
    assert len(read_elements) == 2
    assert isinstance(read_elements[0], LineElement)
    assert isinstance(read_elements[1], ArcElement)

    read_arc = read_elements[1]
    assert read_arc.radius == pytest.approx(200.0)
    assert read_arc.clockwise is False
    assert read_arc.start.x == pytest.approx(arc.start.x)
    assert read_arc.end.x == pytest.approx(arc.end.x)


def test_rejects_duplicate_surface_names(tmp_path) -> None:  # type: ignore[no-untyped-def]
    points = (Point3D(0, 0, 0), Point3D(1, 0, 1), Point3D(0, 1, 2))
    tin = TIN.from_points(points)
    doc = LandXMLDocument(surfaces=(NamedSurface(name="Dup", tin=tin), NamedSurface(name="Dup", tin=tin)))

    with pytest.raises(LandXMLValidationError):
        LandXMLWriter(str(tmp_path / "dup.xml")).write(doc)
