"""
Tests closing gaps identified in the post-implementation audit
(multiple surfaces/groups, linear unit / CRS round-trip, malformed
Surface without Definition, empty Pnts, non-sequential alphanumeric
point ids, unsupported pntRef).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from topocore.geometry.point3d import Point3D
from topocore.io.landxml.coordinates import parse_point_text
from topocore.io.landxml.exceptions import LandXMLParseError
from topocore.io.landxml.models import (
    LandXMLDocument,
    LinearUnit,
    NamedPointGroup,
    NamedSurface,
)
from topocore.io.landxml.reader import LandXMLReader
from topocore.io.landxml.writer import LandXMLWriter
from topocore.survey.models import SurveyPoint, SurveyPointSet
from topocore.terrain.tin import TIN


def _flat_triangle_tin(z: float) -> TIN:
    vertices = (
        Point3D(0.0, 0.0, z),
        Point3D(1.0, 0.0, z),
        Point3D(0.0, 1.0, z),
    )
    simplices = np.array([[0, 1, 2]], dtype=np.int32)
    return TIN.from_mesh(vertices, simplices)


def test_roundtrip_multiple_surfaces_and_point_groups(tmp_path: Path) -> None:
    document = LandXMLDocument(
        surfaces=(
            NamedSurface(name="Existing", tin=_flat_triangle_tin(10.0)),
            NamedSurface(name="Proposed", tin=_flat_triangle_tin(12.0)),
        ),
        point_groups=(
            NamedPointGroup(
                name="Control",
                points=SurveyPointSet(points=(SurveyPoint(id="1", x=0.0, y=0.0, z=0.0),)),
            ),
            NamedPointGroup(
                name="Boundary",
                points=SurveyPointSet(points=(SurveyPoint(id="1", x=5.0, y=5.0, z=0.0),)),
            ),
        ),
    )

    path = tmp_path / "multi.xml"
    report = LandXMLWriter(path).write(document)
    result = LandXMLReader(path).read()

    assert report.surface_count == 2
    assert report.point_group_count == 2
    assert {s.name for s in result.surfaces} == {"Existing", "Proposed"}
    assert {g.name for g in result.point_groups} == {"Control", "Boundary"}

    # Same point id ("1") reused across two different groups must
    # not collide -- uniqueness is scoped per-collection, per the
    # LandXML-1.2 rule audited earlier.
    control = next(g for g in result.point_groups if g.name == "Control")
    boundary = next(g for g in result.point_groups if g.name == "Boundary")
    assert control.points[0].x == pytest.approx(0.0)
    assert boundary.points[0].x == pytest.approx(5.0)


@pytest.mark.parametrize("unit", [LinearUnit.METER, LinearUnit.FOOT])
def test_linear_unit_roundtrips(unit: LinearUnit, tmp_path: Path) -> None:
    document = LandXMLDocument(linear_unit=unit)
    path = tmp_path / "units.xml"

    LandXMLWriter(path).write(document)
    result = LandXMLReader(path).read()

    assert result.linear_unit == unit


def test_crs_roundtrips(tmp_path: Path) -> None:
    document = LandXMLDocument(crs="MAGNA-SIRGAS / Colombia Bogota zone")
    path = tmp_path / "crs.xml"

    LandXMLWriter(path).write(document)
    result = LandXMLReader(path).read()

    assert result.crs == "MAGNA-SIRGAS / Colombia Bogota zone"


def test_surface_without_definition_is_skipped_with_warning(tmp_path: Path) -> None:
    text = """<?xml version="1.0" ?>
    <LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
      <Surfaces>
        <Surface name="Broken"></Surface>
      </Surfaces>
    </LandXML>
    """
    path = tmp_path / "no_definition.xml"
    path.write_text(text, encoding="utf-8")

    document, report = LandXMLReader(path).read_with_report()

    assert len(document.surfaces) == 0
    assert report.warning_count == 1
    assert "Broken" in report.warnings[0]
    assert "no <Definition>" in report.warnings[0]


def test_surface_with_empty_pnts_raises_clear_parse_error(tmp_path: Path) -> None:
    text = """<?xml version="1.0" ?>
    <LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
      <Surfaces>
        <Surface name="Empty">
          <Definition surfType="TIN">
            <Pnts></Pnts>
          </Definition>
        </Surface>
      </Surfaces>
    </LandXML>
    """
    path = tmp_path / "empty_pnts.xml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(LandXMLParseError):
        LandXMLReader(path).read()


def test_non_sequential_alphanumeric_point_ids(tmp_path: Path) -> None:
    text = """<?xml version="1.0" ?>
    <LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
      <Surfaces>
        <Surface name="S1">
          <Definition surfType="TIN">
            <Pnts>
              <P id="A7">0.0 0.0 10.0</P>
              <P id="B3">0.0 10.0 11.0</P>
              <P id="Z99">10.0 0.0 12.0</P>
            </Pnts>
            <Faces><F>A7 B3 Z99</F></Faces>
          </Definition>
        </Surface>
      </Surfaces>
    </LandXML>
    """
    path = tmp_path / "alnum_ids.xml"
    path.write_text(text, encoding="utf-8")

    document = LandXMLReader(path).read()

    assert document.surfaces[0].tin.triangle_count == 1
    assert document.surfaces[0].tin.vertex_count == 3


def test_pntref_gives_explicit_unsupported_error() -> None:
    with pytest.raises(LandXMLParseError, match="pntRef"):
        parse_point_text("")  # text content is empty when pntRef is used instead


def test_read_handles_namespace_with_explicit_prefix(tmp_path: Path) -> None:
    """
    ElementTree normalizes any XML namespace to Clark notation
    ({uri}tag) regardless of whether the source file used a default
    namespace or an explicit prefix (e.g. <landxml:LandXML
    xmlns:landxml="...">) -- local_tag() strips it the same way
    either way. This confirms that empirically rather than assuming
    it from ElementTree's documented behavior.
    """
    text = """<?xml version="1.0" ?>
    <landxml:LandXML xmlns:landxml="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
      <landxml:Surfaces>
        <landxml:Surface name="Prefixed">
          <landxml:Definition surfType="TIN">
            <landxml:Pnts>
              <landxml:P id="1">0.0 0.0 0.0</landxml:P>
              <landxml:P id="2">0.0 1.0 0.0</landxml:P>
              <landxml:P id="3">1.0 0.0 0.0</landxml:P>
            </landxml:Pnts>
            <landxml:Faces><landxml:F>1 2 3</landxml:F></landxml:Faces>
          </landxml:Definition>
        </landxml:Surface>
      </landxml:Surfaces>
    </landxml:LandXML>
    """
    path = tmp_path / "prefixed_namespace.xml"
    path.write_text(text, encoding="utf-8")

    document = LandXMLReader(path).read()

    assert len(document.surfaces) == 1
    assert document.surfaces[0].name == "Prefixed"
    assert document.surfaces[0].tin.triangle_count == 1
