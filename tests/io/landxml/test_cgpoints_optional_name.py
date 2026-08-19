"""
Tests for the PR18B correction: <CgPoints name="..."> is optional
per the LandXML schema -- confirmed against a genuine Autodesk
Civil 3D 2007 export (GSG_features_alignments.xml) whose first
<CgPoints> (survey ground points, ~2400 points) has no 'name'
attribute at all, while four other <CgPoints> in the same file do
(named "Landscape trees", "Pond", "Road centerline", "Storm
manholes"). <CgPoint name="..."> (the individual point) remains
required and validated -- this is a different element with a
different rule.

TopoCore's own NamedPointGroup.name still needs a non-blank string
(it's how TopoCore identifies the group) -- LandXMLReader generates
a deterministic name for an unnamed group rather than rejecting
valid LandXML or silently dropping the group. This is a TopoCore/IO
adaptation, not a tightening of the LandXML standard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from topocore.io.landxml.exceptions import LandXMLValidationError
from topocore.io.landxml.reader import LandXMLReader

_UNNAMED_AND_NAMED = """<?xml version="1.0" ?>
<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
  <CgPoints>
    <CgPoint name="1" code="GRND">100.0 200.0 10.0</CgPoint>
    <CgPoint name="2" code="GRND">101.0 201.0 11.0</CgPoint>
  </CgPoints>
  <CgPoints name="Road centerline">
    <CgPoint name="10" code="CL">500.0 600.0 20.0</CgPoint>
  </CgPoints>
  <CgPoints>
    <CgPoint name="20" code="MISC">700.0 800.0 30.0</CgPoint>
  </CgPoints>
</LandXML>
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_unnamed_cgpoints_reads_without_raising(tmp_path: Path) -> None:
    path = _write(tmp_path / "unnamed.xml", _UNNAMED_AND_NAMED)

    document = LandXMLReader(path).read()  # must not raise

    assert len(document.point_groups) == 3


def test_unnamed_cgpoints_gets_deterministic_generated_name(tmp_path: Path) -> None:
    path = _write(tmp_path / "unnamed.xml", _UNNAMED_AND_NAMED)

    document = LandXMLReader(path).read()

    names = [group.name for group in document.point_groups]
    assert names == ["Unnamed_CgPoints_1", "Road centerline", "Unnamed_CgPoints_2"]


def test_named_cgpoints_keeps_its_declared_name(tmp_path: Path) -> None:
    path = _write(tmp_path / "unnamed.xml", _UNNAMED_AND_NAMED)

    document = LandXMLReader(path).read()

    named_group = next(g for g in document.point_groups if g.name == "Road centerline")
    assert named_group.points[0].id == "10"


def test_unnamed_cgpoints_points_are_preserved(tmp_path: Path) -> None:
    path = _write(tmp_path / "unnamed.xml", _UNNAMED_AND_NAMED)

    document = LandXMLReader(path).read()

    unnamed_group = next(g for g in document.point_groups if g.name == "Unnamed_CgPoints_1")
    assert len(unnamed_group.points) == 2
    assert unnamed_group.points[0].id == "1"
    assert unnamed_group.points[1].id == "2"


def test_cgpoint_individual_name_is_still_required(tmp_path: Path) -> None:
    """
    The correction is scoped to <CgPoints> (the group); <CgPoint>
    (the individual point) is a different element and its 'name'
    remains required and validated.
    """
    text = """<?xml version="1.0" ?>
    <LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
      <CgPoints>
        <CgPoint code="GRND">100.0 200.0 10.0</CgPoint>
      </CgPoints>
    </LandXML>
    """
    path = _write(tmp_path / "no_point_name.xml", text)

    with pytest.raises(LandXMLValidationError):
        LandXMLReader(path).read()


# ----------------------------------------------------------------------
# Real Civil 3D file: the exact case that drove this correction.
# ----------------------------------------------------------------------

_REAL_FIXTURE = Path(__file__).parent / "fixtures" / "GSG_features_alignments.xml"


@pytest.mark.skipif(not _REAL_FIXTURE.exists(), reason="Real Civil 3D fixture not present")
def test_real_civil3d_file_pntref_points_skipped_with_specific_warning() -> None:
    """
    'Landscape trees' in the real file uses <CgPoint pntRef="..."/>
    exclusively (no literal coordinates) -- confirmed by inspection.
    The group itself must still be read (name preserved), with each
    pntRef point skipped and reported individually, not a group-wide
    rejection over a technically-absent 'name'.
    """
    document, report = LandXMLReader(_REAL_FIXTURE).read_with_report()

    trees = next(g for g in document.point_groups if g.name == "Landscape trees")
    assert len(trees.points) == 0  # every point in this group uses pntRef

    pntref_warnings = [w for w in report.warnings if "pntRef" in w and "Landscape trees" in w]
    assert len(pntref_warnings) > 0


@pytest.mark.skipif(not _REAL_FIXTURE.exists(), reason="Real Civil 3D fixture not present")
def test_real_civil3d_file_unnamed_cgpoints_reads_successfully() -> None:
    document = LandXMLReader(_REAL_FIXTURE).read()

    assert len(document.point_groups) == 5  # 1 unnamed + 4 named, per the real file

    names = [group.name for group in document.point_groups]
    assert "Unnamed_CgPoints_1" in names
    assert "Landscape trees" in names
    assert "Pond" in names
    assert "Road centerline" in names
    assert "Storm manholes" in names


@pytest.mark.skipif(not _REAL_FIXTURE.exists(), reason="Real Civil 3D fixture not present")
def test_real_civil3d_file_unnamed_group_has_survey_points() -> None:
    document = LandXMLReader(_REAL_FIXTURE).read()

    unnamed = next(g for g in document.point_groups if g.name == "Unnamed_CgPoints_1")
    assert len(unnamed.points) > 0
    # The real group mixes multiple feature codes (GRND, LSCP, PIPE,
    # POND, ...) -- confirmed by inspection, not assumed to be
    # exclusively "GRND".
    codes = {point.code for point in unnamed.points}
    assert "GRND" in codes
    assert len(codes) > 1
