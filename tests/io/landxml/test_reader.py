"""
Tests for LandXMLReader against hand-written LandXML text, including
real-world-shaped examples audited from Civil3D/MicroSurvey/Bentley
documentation during the PR18B design phase.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from topocore.io.landxml.exceptions import LandXMLParseError, LandXMLValidationError
from topocore.io.landxml.reader import LandXMLReader

_MICROSURVEY_STYLE = """<?xml version="1.0" ?>
<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
  <Units>
    <Metric linearUnit="meter" />
  </Units>
  <Surfaces>
    <Surface name="Surface 1">
      <Definition surfType="TIN">
        <Pnts>
          <P id="1">0.0 0.0 10.0</P>
          <P id="2">0.0 10.0 11.0</P>
          <P id="3">10.0 0.0 12.0</P>
        </Pnts>
        <Faces>
          <F>1 2 3</F>
        </Faces>
      </Definition>
    </Surface>
  </Surfaces>
</LandXML>
"""

_CGPOINTS_STYLE = """<?xml version="1.0" ?>
<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
  <CgPoints name="Set Points">
    <CgPoint name="1" code="PCOR">1145.99755444 -626.27773323 0.000000</CgPoint>
    <CgPoint name="2" code="PCOR">1759.50228244 -493.99972666 0.000000</CgPoint>
  </CgPoints>
</LandXML>
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_read_surface_from_microsurvey_style_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "surface.xml", _MICROSURVEY_STYLE)

    document = LandXMLReader(path).read()

    assert len(document.surfaces) == 1
    surface = document.surfaces[0]
    assert surface.name == "Surface 1"
    assert surface.tin.vertex_count == 3
    assert surface.tin.triangle_count == 1


def test_read_surface_applies_north_east_elev_convention(tmp_path: Path) -> None:
    path = _write(tmp_path / "surface.xml", _MICROSURVEY_STYLE)

    document = LandXMLReader(path).read()
    first_vertex = document.surfaces[0].tin.vertices[0]

    # <P id="1">0.0 0.0 10.0</P> -> north=0, east=0, elev=10
    assert first_vertex.x == pytest.approx(0.0)
    assert first_vertex.y == pytest.approx(0.0)
    assert first_vertex.z == pytest.approx(10.0)

    second_vertex = document.surfaces[0].tin.vertices[1]
    # <P id="2">0.0 10.0 11.0</P> -> north=0, east=10 -> x=10, y=0
    assert second_vertex.x == pytest.approx(10.0)
    assert second_vertex.y == pytest.approx(0.0)


def test_read_cgpoints_from_real_world_style_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "points.xml", _CGPOINTS_STYLE)

    document = LandXMLReader(path).read()

    assert len(document.point_groups) == 1
    group = document.point_groups[0]
    assert group.name == "Set Points"
    assert len(group.points) == 2
    assert group.points[0].id == "1"
    assert group.points[0].code == "PCOR"


def test_read_rejects_malformed_xml(tmp_path: Path) -> None:
    path = _write(tmp_path / "broken.xml", "<LandXML><Surfaces>")

    with pytest.raises(LandXMLParseError):
        LandXMLReader(path).read()


def test_read_rejects_face_referencing_unknown_point_id(tmp_path: Path) -> None:
    text = _MICROSURVEY_STYLE.replace("<F>1 2 3</F>", "<F>1 2 99</F>")
    path = _write(tmp_path / "bad_face.xml", text)

    with pytest.raises(LandXMLValidationError):
        LandXMLReader(path).read()


def test_read_rejects_duplicated_point_id(tmp_path: Path) -> None:
    text = _MICROSURVEY_STYLE.replace('<P id="2">', '<P id="1">')
    path = _write(tmp_path / "dup_id.xml", text)

    with pytest.raises(LandXMLValidationError):
        LandXMLReader(path).read()


def test_read_rejects_wrong_root_element(tmp_path: Path) -> None:
    path = _write(tmp_path / "wrong_root.xml", "<NotLandXML></NotLandXML>")

    with pytest.raises(LandXMLValidationError):
        LandXMLReader(path).read()


def test_read_skips_non_tin_surface_with_warning(tmp_path: Path) -> None:
    text = _MICROSURVEY_STYLE.replace('surfType="TIN"', 'surfType="GRID"')
    path = _write(tmp_path / "grid_surface.xml", text)

    document, report = LandXMLReader(path).read_with_report()

    assert len(document.surfaces) == 0
    assert report.warning_count == 1
    assert "GRID" in report.warnings[0]


def test_read_skips_alignment_without_coordgeom_with_warning(tmp_path: Path) -> None:
    """
    An <Alignment> with no <CoordGeom> at all has no geometry to
    build -- skipped with a warning (same pattern as a <Surface>
    without <Definition>), not silently dropped and not a hard
    error.
    """
    text = _MICROSURVEY_STYLE.replace(
        "</Surfaces>",
        '</Surfaces><Alignments><Alignment name="Road 1"></Alignment></Alignments>',
    )
    path = _write(tmp_path / "with_alignment.xml", text)

    document, report = LandXMLReader(path).read_with_report()

    assert len(document.surfaces) == 1
    assert len(document.alignments) == 0
    assert any("Road 1" in warning for warning in report.warnings)


def test_read_missing_file_raises_parse_error(tmp_path: Path) -> None:
    with pytest.raises(LandXMLParseError):
        LandXMLReader(tmp_path / "does_not_exist.xml").read()
