"""
Tests confirming AlignmentGeometryError (a domain exception) never
escapes LandXMLReader as a raw traceback -- it must always be
wrapped as LandXMLParseError, with the original exception chained
via `from exc` for diagnosis. Independent of the tolerance question:
this must hold even when geometry is genuinely, un-reconcilably
invalid (exceeds even the widened LandXML import tolerance).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from topocore.io.landxml.exceptions import LandXMLParseError
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


def test_grossly_inconsistent_curve_raises_clean_parse_error_not_traceback(
    tmp_path: Path,
) -> None:
    geometry = """
    <Curve rot="cw" crvType="arc" radius="1000.0">
      <Start>0.0 50.0</Start>
      <Center>0.0 0.0</Center>
      <End>50.0 0.0</End>
    </Curve>
    """
    path = _write(tmp_path / "bad_curve.xml", geometry)

    with pytest.raises(LandXMLParseError) as excinfo:
        LandXMLReader(path).read()

    assert "Road 1" in str(excinfo.value)
    assert "<Curve>" in str(excinfo.value)
    # Original domain exception is chained for diagnosis.
    assert excinfo.value.__cause__ is not None


def test_coincident_line_endpoints_raise_clean_parse_error(tmp_path: Path) -> None:
    geometry = """
    <Line length="0.0">
      <Start>10.0 10.0</Start>
      <End>10.0 10.0</End>
    </Line>
    """
    path = _write(tmp_path / "zero_length_line.xml", geometry)

    with pytest.raises(LandXMLParseError) as excinfo:
        LandXMLReader(path).read()

    assert excinfo.value.__cause__ is not None


def test_slightly_off_curve_accepted_via_import_tolerance_with_warning(
    tmp_path: Path,
) -> None:
    """
    A discrepancy just above the domain tolerance but within the
    LandXML import tolerance is accepted, with an explicit warning
    -- confirms the reconciliation path itself (not just the real
    file) via a minimal, hand-built case.
    """
    geometry = """
    <Curve rot="ccw" crvType="arc" radius="100.000000005">
      <Start>0.0 100.0</Start>
      <Center>0.0 0.0</Center>
      <End>100.0 0.0</End>
    </Curve>
    """
    path = _write(tmp_path / "slightly_off.xml", geometry)

    document, report = LandXMLReader(path).read_with_report()

    assert len(document.alignments) == 1
    assert any("import tolerance" in w for w in report.warnings)
