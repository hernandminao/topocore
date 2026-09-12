"""
Tests for SurveyPointSet.crs and its 3 real consumers:
SurveyTXTReader.read() (.prj detection), transform_survey()
(horizontal -- updates crs to target), transform_survey_vertical()
(vertical -- preserves crs unchanged).

Confirmed during this capability's own audit: SurveyPointSet is
constructed at exactly 5 call sites across 3 modules from 3 different
project phases (survey/reader.py x2, geodesy/transform.py,
geodesy/vertical/transform.py, io/landxml/reader.py). Adding `crs`
with a default of None keeps all 5 working unchanged; this suite
verifies the 3 sites that needed real behavior changes, plus confirms
LandXML's own construction site is genuinely unaffected.

A real circular import was found and fixed while adding this field:
topocore.survey.models importing topocore.geodesy.crs.CRS directly
(even the specific submodule, not just the package) created a cycle,
since topocore.geodesy's own __init__.py eagerly imports
geodesy/transform.py, which itself imports topocore.survey.models.
Fixed via `if TYPE_CHECKING` (survey/models.py already has
`from __future__ import annotations`, so the type annotation itself
never needs a real runtime import).
"""

from __future__ import annotations

from pathlib import Path

import pyproj
import pytest
from topocore.geodesy import CRS, CoordinateTransformer
from topocore.geodesy.transform import transform_survey
from topocore.geodesy.vertical import GeoidGrid, VerticalTransformer
from topocore.geodesy.vertical_datum import VerticalDatum
from topocore.survey.formats import SurveyFormat
from topocore.survey.models import SurveyPoint, SurveyPointSet
from topocore.survey.reader import SurveyTXTReader

# ----------------------------------------------------------------------
# 1. SurveyPointSet.crs itself -- additive, all 5 existing construction
# sites keep working with the default.
# ----------------------------------------------------------------------


def test_crs_defaults_to_none() -> None:
    assert SurveyPointSet(points=()).crs is None
    assert SurveyPointSet(points=(SurveyPoint(id="1", x=0, y=0, z=0),)).crs is None


def test_crs_can_be_set_at_construction() -> None:
    crs = CRS.from_epsg(32618)
    sps = SurveyPointSet(points=(), crs=crs)
    assert sps.crs is crs


def test_survey_point_set_is_still_frozen() -> None:
    sps = SurveyPointSet(points=())
    with pytest.raises(AttributeError):
        sps.crs = CRS.from_epsg(4326)  # type: ignore[misc]


# ----------------------------------------------------------------------
# 2. SurveyTXTReader.read() -- .prj detection.
# ----------------------------------------------------------------------


def test_survey_txt_reader_picks_up_prj_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "survey.txt"
    path.write_text("1,0.0,0.0,100.0,PT\n", encoding="utf-8")
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(3116).to_wkt(), encoding="utf-8")

    result = SurveyTXTReader(path, format=SurveyFormat.ID_XYZ_CODE).read()
    assert result.crs is not None
    assert result.crs.epsg == 3116


def test_survey_txt_reader_without_prj_leaves_crs_none(tmp_path: Path) -> None:
    path = tmp_path / "survey.txt"
    path.write_text("1,0.0,0.0,100.0,PT\n", encoding="utf-8")

    result = SurveyTXTReader(path, format=SurveyFormat.ID_XYZ_CODE).read()
    assert result.crs is None


def test_survey_txt_reader_prj_never_alters_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "survey.txt"
    path.write_text("1,1000.0,1000.0,100.0,PT\n", encoding="utf-8")
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(3116).to_wkt(), encoding="utf-8")

    result = SurveyTXTReader(path, format=SurveyFormat.ID_XYZ_CODE).read()
    assert result.points[0].x == 1000.0
    assert result.points[0].y == 1000.0


def test_empty_survey_file_still_checks_for_prj(tmp_path: Path) -> None:
    """The empty-file early-return branch was updated too -- confirmed explicitly, not assumed."""
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    (tmp_path / "empty.prj").write_text(pyproj.CRS.from_epsg(4326).to_wkt(), encoding="utf-8")

    result = SurveyTXTReader(path).read()
    assert len(result.points) == 0
    assert result.crs is not None
    assert result.crs.epsg == 4326


# ----------------------------------------------------------------------
# 3. transform_survey() -- horizontal, updates crs to target.
# ----------------------------------------------------------------------


def test_transform_survey_sets_crs_to_target() -> None:
    survey = SurveyPointSet(points=(SurveyPoint(id="1", x=-74.0, y=4.7, z=2600.0),))
    transformer = CoordinateTransformer(CRS.from_epsg(4326), CRS.from_epsg(32618))

    result = transform_survey(survey, transformer)

    assert result.crs is not None
    assert result.crs.epsg == 32618


def test_transform_survey_does_not_mutate_input_crs() -> None:
    survey = SurveyPointSet(points=(SurveyPoint(id="1", x=-74.0, y=4.7, z=2600.0),), crs=CRS.from_epsg(4326))
    transformer = CoordinateTransformer(CRS.from_epsg(4326), CRS.from_epsg(32618))

    transform_survey(survey, transformer)

    assert survey.crs is not None
    assert survey.crs.epsg == 4326


# ----------------------------------------------------------------------
# 4. transform_survey_vertical() -- preserves crs unchanged.
# ----------------------------------------------------------------------


def test_transform_survey_vertical_preserves_crs(synthetic_geoid_path: Path) -> None:
    """The core fix: without this, the default SurveyPointSet(points=...) constructor would silently reset crs to None."""
    utm18 = CRS.from_epsg(32618)
    survey = SurveyPointSet(points=(SurveyPoint(id="1", x=1.0, y=1.0, z=100.0),), crs=utm18)

    vertical_transformer = VerticalTransformer(
        source_datum=VerticalDatum(name="ellipsoidal"),
        target_datum=VerticalDatum(name="orthometric", geoid_model="SYNTHETIC"),
        geoid=GeoidGrid.from_geotiff(synthetic_geoid_path),
    )

    from topocore.geodesy.vertical.transform import transform_survey_vertical

    result = transform_survey_vertical(survey, vertical_transformer)

    assert result.crs is utm18  # same object, not just an equal one -- genuinely preserved, not reconstructed
    assert result.points[0].z == 50.0  # confirms Z was genuinely shifted, not just crs preserved on a no-op


def test_transform_survey_vertical_preserves_none_crs(
    synthetic_geoid_path: Path,
) -> None:
    survey = SurveyPointSet(points=(SurveyPoint(id="1", x=1.0, y=1.0, z=100.0),))  # crs=None
    vertical_transformer = VerticalTransformer(
        source_datum=VerticalDatum(name="ellipsoidal"),
        target_datum=VerticalDatum(name="orthometric", geoid_model="SYNTHETIC"),
        geoid=GeoidGrid.from_geotiff(synthetic_geoid_path),
    )

    from topocore.geodesy.vertical.transform import transform_survey_vertical

    result = transform_survey_vertical(survey, vertical_transformer)
    assert result.crs is None


# ----------------------------------------------------------------------
# 5. LandXMLReader -- unaffected, confirmed with a real file.
# ----------------------------------------------------------------------


def test_landxml_point_groups_are_unaffected(tmp_path: Path) -> None:
    from topocore.io.landxml.reader import LandXMLReader

    content = (
        '<?xml version="1.0"?>\n'
        '<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">\n'
        '  <Units><Metric linearUnit="meter"/></Units>\n'
        '  <CgPoints name="Points" desc="Test">\n'
        '    <CgPoint name="P1">0.0 0.0 10.0</CgPoint>\n'
        "  </CgPoints>\n"
        "</LandXML>\n"
    )
    path = tmp_path / "test.xml"
    path.write_text(content, encoding="utf-8")

    document = LandXMLReader(path).read()
    point_group = document.point_groups[0]

    assert point_group.points.crs is None
    assert len(point_group.points.points) == 1
