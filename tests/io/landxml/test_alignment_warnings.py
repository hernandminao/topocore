"""
Tests for the "skip with explicit warning" paths for unsupported
<Alignment> content -- crvType != arc, spiType != clothoid, and
curve-to-curve compound spirals. Per Hernán's requirement, the
warning must identify exactly which Alignment/element and, for the
compound-spiral case, the radiusStart/radiusEnd values.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from topocore.io.landxml.exceptions import LandXMLValidationError
from topocore.io.landxml.reader import LandXMLReader

_BASE = """<?xml version="1.0" ?>
<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
  <Alignments>
    <Alignment name="Road 1" length="100.0" staStart="0.0">
      <CoordGeom>
        {geometry}
      </CoordGeom>
    </Alignment>
  </Alignments>
</LandXML>
"""


def _write(path: Path, geometry: str) -> Path:
    path.write_text(_BASE.format(geometry=geometry), encoding="utf-8")
    return path


def test_chord_curve_crvtype_is_skipped_with_specific_warning(tmp_path: Path) -> None:
    geometry = """
    <Curve rot="cw" crvType="chord" length="50.0" radius="100.0">
      <Start>0.0 0.0</Start>
      <End>0.0 50.0</End>
    </Curve>
    """
    path = _write(tmp_path / "chord.xml", geometry)

    document, report = LandXMLReader(path).read_with_report()

    assert len(document.alignments) == 0  # no supported elements -> whole alignment skipped
    assert any("Road 1" in w and "chord" in w for w in report.warnings)


def test_non_clothoid_spiral_is_skipped_with_specific_warning(tmp_path: Path) -> None:
    geometry = """
    <Line length="50.0">
      <Start>0.0 0.0</Start>
      <End>0.0 50.0</End>
    </Line>
    <Spiral spiType="cubic" rot="cw" length="30.0" radiusStart="INF" radiusEnd="100.0">
      <Start>0.0 50.0</Start>
      <PI>0.0 65.0</PI>
      <End>0.0 80.0</End>
    </Spiral>
    """
    path = _write(tmp_path / "cubic_spiral.xml", geometry)

    document, report = LandXMLReader(path).read_with_report()

    # The Line is still read even though the Spiral is skipped.
    assert len(document.alignments) == 1
    assert len(document.alignments[0].alignment.elements) == 1
    assert any("Road 1" in w and "cubic" in w for w in report.warnings)


def test_compound_curve_to_curve_spiral_is_skipped_with_radius_identification(
    tmp_path: Path,
) -> None:
    geometry = """
    <Line length="50.0">
      <Start>0.0 0.0</Start>
      <End>0.0 50.0</End>
    </Line>
    <Spiral spiType="clothoid" rot="cw" length="30.0" radiusStart="200.0" radiusEnd="100.0">
      <Start>0.0 50.0</Start>
      <PI>0.0 65.0</PI>
      <End>0.0 80.0</End>
    </Spiral>
    """
    path = _write(tmp_path / "compound_spiral.xml", geometry)

    document, report = LandXMLReader(path).read_with_report()

    assert len(document.alignments) == 1
    assert len(document.alignments[0].alignment.elements) == 1  # only the Line

    warning = next(w for w in report.warnings if "Road 1" in w)
    assert "200.0" in warning
    assert "100.0" in warning


def test_curve_without_center_is_skipped_with_warning(tmp_path: Path) -> None:
    """
    A <Curve> without <Center> (chord/3-point curve variant some
    producers emit) is not a hard failure -- skipped with a warning,
    same pattern as unsupported crvType.
    """
    geometry = """
    <Curve rot="cw" crvType="arc" radius="100.0">
      <Start>0.0 0.0</Start>
      <End>100.0 100.0</End>
    </Curve>
    """
    path = _write(tmp_path / "no_center.xml", geometry)

    document, report = LandXMLReader(path).read_with_report()

    assert len(document.alignments) == 0
    assert any("Road 1" in w and "Center" in w for w in report.warnings)


def test_alignment_missing_name_raises_validation_error(tmp_path: Path) -> None:
    text = """<?xml version="1.0" ?>
    <LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
      <Alignments><Alignment length="10.0"><CoordGeom>
        <Line length="10.0"><Start>0.0 0.0</Start><End>10.0 0.0</End></Line>
      </CoordGeom></Alignment></Alignments>
    </LandXML>
    """
    path = tmp_path / "no_name.xml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(LandXMLValidationError):
        LandXMLReader(path).read()


def test_line_missing_start_raises_clear_validation_error(tmp_path: Path) -> None:
    text = """<?xml version="1.0" ?>
    <LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
      <Alignments><Alignment name="A"><CoordGeom>
        <Line length="10.0"><End>10.0 0.0</End></Line>
      </CoordGeom></Alignment></Alignments>
    </LandXML>
    """
    path = tmp_path / "no_start.xml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(LandXMLValidationError, match="Start"):
        LandXMLReader(path).read()
