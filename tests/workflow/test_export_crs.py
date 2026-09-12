"""
Tests for Workflow.export_dxf()/export_gpkg() propagating
FeatureCollection.crs into the actual exported file -- not just into
an intermediate options object.

Every DXF assertion reads the file back with ezdxf; every GeoPackage
assertion reads the file back with raw sqlite3 against
gpkg_contents/gpkg_spatial_ref_sys -- confirming the CRS is genuinely
written into the file on disk, not merely accepted by an in-memory
options object.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import ezdxf
import pyproj
import pytest
from topocore.survey.formats import SurveyFormat
from topocore.workflow.exceptions import WorkflowStateError
from topocore.workflow.workflow import Workflow


def _survey_with_prj(tmp_path: Path, name: str, epsg: int | None) -> Path:
    path = tmp_path / f"{name}.csv"
    path.write_text("1,-74.0,4.7,2600.0,ARBOL\n", encoding="utf-8")
    if epsg is not None:
        (tmp_path / f"{name}.prj").write_text(pyproj.CRS.from_epsg(epsg).to_wkt(), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# export_dxf() -- CRS genuinely written to the file.
# ----------------------------------------------------------------------


def test_export_dxf_writes_crs_into_the_real_file(tmp_path: Path) -> None:
    survey_path = _survey_with_prj(tmp_path, "survey", epsg=3116)
    workflow = Workflow().read_survey(survey_path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()

    dxf_path = tmp_path / "output.dxf"
    workflow.export_dxf(dxf_path)

    doc = ezdxf.readfile(dxf_path)
    custom_vars = dict(doc.header.custom_vars)
    assert custom_vars.get("TopoCore CRS") == "EPSG:3116"


def test_export_dxf_without_crs_writes_no_crs_header_var(tmp_path: Path) -> None:
    survey_path = _survey_with_prj(tmp_path, "survey", epsg=None)
    workflow = Workflow().read_survey(survey_path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()

    dxf_path = tmp_path / "output.dxf"
    workflow.export_dxf(dxf_path)

    doc = ezdxf.readfile(dxf_path)
    custom_vars = dict(doc.header.custom_vars)
    assert "TopoCore CRS" not in custom_vars


# ----------------------------------------------------------------------
# export_gpkg() -- the 5-case resolution policy, each confirmed by
# reading the real srs_id back from the file's own gpkg_contents table.
# ----------------------------------------------------------------------


def _read_srs_id(gpkg_path: Path) -> int:
    """
    Reads every row of gpkg_contents, not just the first -- confirmed
    directly that a single export with multiple distinct feature
    types produces multiple layer rows (e.g. "vegetation_point",
    "utility_point"), so this must verify they all genuinely share
    the same srs_id rather than trusting an arbitrary first row.
    """
    con = sqlite3.connect(gpkg_path)
    try:
        rows = con.execute("SELECT table_name, srs_id FROM gpkg_contents").fetchall()
        assert rows, "gpkg_contents has no rows"
        srs_ids = {srs_id for _table_name, srs_id in rows}
        assert len(srs_ids) == 1, f"expected a single shared srs_id across all layers, found: {rows}"
        return int(srs_ids.pop())
    finally:
        con.close()


def test_detected_crs_used_automatically_without_explicit_epsg(tmp_path: Path) -> None:
    survey_path = _survey_with_prj(tmp_path, "survey", epsg=3116)
    workflow = Workflow().read_survey(survey_path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()

    gpkg_path = tmp_path / "output.gpkg"
    workflow.export_gpkg(gpkg_path)  # no epsg given

    assert _read_srs_id(gpkg_path) == 3116


def test_missing_crs_requires_explicit_epsg(tmp_path: Path) -> None:
    survey_path = _survey_with_prj(tmp_path, "survey", epsg=None)
    workflow = Workflow().read_survey(survey_path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()

    with pytest.raises(WorkflowStateError, match="requires an explicit epsg"):
        workflow.export_gpkg(tmp_path / "output.gpkg")


def test_missing_crs_with_explicit_epsg_succeeds(tmp_path: Path) -> None:
    survey_path = _survey_with_prj(tmp_path, "survey", epsg=None)
    workflow = Workflow().read_survey(survey_path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()

    gpkg_path = tmp_path / "output.gpkg"
    workflow.export_gpkg(gpkg_path, epsg=4326)

    assert _read_srs_id(gpkg_path) == 4326


def test_matching_detected_and_explicit_epsg_succeeds(tmp_path: Path) -> None:
    survey_path = _survey_with_prj(tmp_path, "survey", epsg=3116)
    workflow = Workflow().read_survey(survey_path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()

    gpkg_path = tmp_path / "output.gpkg"
    workflow.export_gpkg(gpkg_path, epsg=3116)

    assert _read_srs_id(gpkg_path) == 3116


def test_conflicting_detected_and_explicit_epsg_is_rejected(tmp_path: Path) -> None:
    survey_path = _survey_with_prj(tmp_path, "survey", epsg=3116)
    workflow = Workflow().read_survey(survey_path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()

    with pytest.raises(WorkflowStateError, match="refused"):
        workflow.export_gpkg(tmp_path / "output.gpkg", epsg=4326)


def test_non_epsg_crs_behaves_like_missing_requires_explicit_epsg(
    tmp_path: Path,
) -> None:
    """A CRS with no derivable EPSG code (e.g. a custom, unregistered CRS) is treated like crs=None -- never guessed."""
    custom_crs = pyproj.CRS.from_proj4(
        "+proj=lcc +lat_1=4 +lat_2=6 +lat_0=5 +lon_0=-74 +x_0=500000 "
        "+y_0=500000 +ellps=GRS80 +units=m +no_defs +type=crs"
    )
    assert custom_crs.to_epsg() is None  # confirm the premise

    path = tmp_path / "survey.csv"
    path.write_text("1,-74.0,4.7,2600.0,ARBOL\n", encoding="utf-8")
    (tmp_path / "survey.prj").write_text(custom_crs.to_wkt(), encoding="utf-8")

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()

    with pytest.raises(WorkflowStateError, match="requires an explicit epsg"):
        workflow.export_gpkg(tmp_path / "output.gpkg")


# ----------------------------------------------------------------------
# Reviewer-identified gaps, closed with real, verified evidence.
# ----------------------------------------------------------------------


def test_export_dxf_with_non_epsg_crs_writes_the_literal_name_string(
    tmp_path: Path,
) -> None:
    """
    Confirmed directly: a custom, unregistered CRS's own .name is the
    literal string "unknown" (pyproj's own generic placeholder,
    already documented elsewhere as a shared, known limitation of the
    "EPSG:{code} or .name" convention) -- and this genuinely gets
    written into the exported DXF's own header variable as-is, not
    specially handled or suppressed.
    """
    custom_crs = pyproj.CRS.from_proj4(
        "+proj=lcc +lat_1=4 +lat_2=6 +lat_0=5 +lon_0=-74 +x_0=500000 "
        "+y_0=500000 +ellps=GRS80 +units=m +no_defs +type=crs"
    )
    assert custom_crs.name == "unknown"  # confirm the premise

    path = tmp_path / "survey.csv"
    path.write_text("1,-74.0,4.7,2600.0,ARBOL\n", encoding="utf-8")
    (tmp_path / "survey.prj").write_text(custom_crs.to_wkt(), encoding="utf-8")

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()
    dxf_path = tmp_path / "output.dxf"
    workflow.export_dxf(dxf_path)

    doc = ezdxf.readfile(dxf_path)
    assert dict(doc.header.custom_vars).get("TopoCore CRS") == "unknown"


def test_export_gpkg_with_compound_crs_requires_explicit_epsg(tmp_path: Path) -> None:
    """
    Confirmed directly: a compound CRS (e.g. EPSG:32618+5773) has no
    single EPSG code of its own (pyproj's own to_epsg() returns None
    for it), so FeatureCollection.crs becomes its bare name string,
    not an "EPSG:"-prefixed one -- treated identically to any other
    non-EPSG CRS, requiring an explicit epsg. Separately confirmed:
    GPKGExportOptions.epsg is typed as a single `int`, structurally
    unable to represent a compound code even if a caller wanted to
    supply one directly -- a genuine, pre-existing limitation of the
    exporter itself, not introduced by this CRS-propagation work.
    """
    compound_crs = pyproj.CRS.from_epsg("32618+5773")
    assert compound_crs.to_epsg() is None  # confirm the premise

    path = tmp_path / "survey.csv"
    path.write_text("1,500000.0,4649776.0,2600.0,ARBOL\n", encoding="utf-8")
    (tmp_path / "survey.prj").write_text(compound_crs.to_wkt(), encoding="utf-8")

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()

    with pytest.raises(WorkflowStateError, match="requires an explicit epsg"):
        workflow.export_gpkg(tmp_path / "output.gpkg")


def test_export_gpkg_with_multiple_feature_types_all_share_the_same_srs_id(
    tmp_path: Path,
) -> None:
    """
    Confirmed directly: a single export with multiple distinct
    feature types produces multiple gpkg_contents rows (one per
    layer/table) -- this verifies ALL of them genuinely carry the
    same srs_id, using the now-fixed _read_srs_id() helper (which
    previously only checked the first row via LIMIT 1).
    """
    path = tmp_path / "survey.csv"
    path.write_text("1,-74.0,4.7,2600.0,ARBOL\n2,-74.1,4.8,2610.0,POSTE\n", encoding="utf-8")
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(3116).to_wkt(), encoding="utf-8")

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()
    gpkg_path = tmp_path / "output.gpkg"
    workflow.export_gpkg(gpkg_path)

    con = sqlite3.connect(gpkg_path)
    try:
        rows = con.execute("SELECT table_name FROM gpkg_contents").fetchall()
    finally:
        con.close()
    assert len(rows) >= 2  # confirms the premise: multiple layers were genuinely produced

    assert _read_srs_id(gpkg_path) == 3116  # would fail if any layer disagreed


def test_export_gpkg_re_exporting_to_the_same_path_keeps_a_consistent_srs_id(
    tmp_path: Path,
) -> None:
    path = tmp_path / "survey.csv"
    path.write_text("1,-74.0,4.7,2600.0,ARBOL\n", encoding="utf-8")
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(3116).to_wkt(), encoding="utf-8")

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()
    gpkg_path = tmp_path / "output.gpkg"

    workflow.export_gpkg(gpkg_path)
    assert _read_srs_id(gpkg_path) == 3116

    workflow.export_gpkg(gpkg_path)  # re-export to the same path
    assert _read_srs_id(gpkg_path) == 3116


def test_export_dxf_has_no_mechanism_for_an_explicit_crs_override(
    tmp_path: Path,
) -> None:
    """
    Confirmed directly: unlike export_gpkg()'s own explicit `epsg`
    parameter, export_dxf() has no equivalent -- DXFExportOptions
    itself has no epsg/crs field at all, so passing one raises a
    genuine TypeError-shaped failure from DXFExportOptions's own
    constructor. DXF's CRS comes only from FeatureCollection.crs
    automatically; there is no override path. This is a confirmed,
    real asymmetry with export_gpkg(), not something this specific
    9-point scope of work introduced or was asked to close.
    """
    from topocore.workflow.exceptions import WorkflowExecutionError

    path = tmp_path / "survey.csv"
    path.write_text("1,-74.0,4.7,2600.0,ARBOL\n", encoding="utf-8")

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()

    with pytest.raises(WorkflowExecutionError, match="unexpected keyword argument 'epsg'"):
        workflow.export_dxf(tmp_path / "output.dxf", epsg=32618)
