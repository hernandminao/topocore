"""
Tests for the LAS/LAZ/E57 native-CRS-with-.prj-fallback policy:

    native CRS present  -> always wins, .prj never even consulted
    native CRS absent   -> fall back to .prj (ExternalCRSDetector)
    neither present     -> None

This directly matches this capability's own audit findings: today,
LAS/LAZ/E57 never consulted `.prj` at all (confirmed by direct
execution before this integration existed) -- there was genuinely no
conflict scenario possible in the code. This test suite locks in the
policy now that it exists.

LandXML is explicitly out of scope here -- its own reader returns
LandXMLDocument, not PointCloud, so apply_crs_with_native_priority()
(which requires a PointCloud with a .crs setter) does not apply to
it without a separate design, deferred to a later phase.
"""

from __future__ import annotations

from pathlib import Path

import laspy  # type: ignore[import-untyped]
import numpy as np
import pye57  # type: ignore[import-untyped]
import pyproj
import pytest
from pyproj.enums import WktVersion
from topocore.io.e57.reader import E57Reader
from topocore.io.exceptions import PointCloudIOError
from topocore.io.las.reader import LASReader
from topocore.io.laz.reader import LAZReader
from topocore.pointcloud.attributes import PointAttribute


def _write_las(path: Path, *, crs: pyproj.CRS | None, laz: bool = False) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    if crs is not None:
        header.add_crs(crs)
    data = laspy.LasData(header)
    n = 5
    rng = np.random.default_rng(0)
    data.x = rng.uniform(500000, 500100, n)
    data.y = rng.uniform(4000000, 4000100, n)
    data.z = rng.uniform(100, 200, n)
    if laz:
        data.write(str(path), laz_backend=laspy.LazBackend.Lazrs)
    else:
        data.write(str(path))


def _write_e57(path: Path) -> None:
    """A default E57 write -- coordinateMetadata is always empty via this API, confirmed during this project's own audit."""
    e57_file = pye57.E57(str(path), mode="w")
    n = 5
    rng = np.random.default_rng(0)
    e57_file.write_scan_raw(
        {
            "cartesianX": rng.uniform(0, 10, n),
            "cartesianY": rng.uniform(0, 10, n),
            "cartesianZ": rng.uniform(0, 5, n),
        }
    )
    e57_file.close()


# ----------------------------------------------------------------------
# The 9-case matrix, for LAS.
# ----------------------------------------------------------------------


def test_native_present_no_prj(tmp_path: Path) -> None:
    path = tmp_path / "case.las"
    _write_las(path, crs=pyproj.CRS.from_epsg(32618))
    assert LASReader(path).read().crs == "EPSG:32618"


def test_native_present_prj_matches(tmp_path: Path) -> None:
    path = tmp_path / "case.las"
    _write_las(path, crs=pyproj.CRS.from_epsg(32618))
    (tmp_path / "case.prj").write_text(pyproj.CRS.from_epsg(32618).to_wkt(), encoding="utf-8")
    assert LASReader(path).read().crs == "EPSG:32618"


def test_native_present_prj_disagrees_native_wins(tmp_path: Path) -> None:
    """The core safety guarantee: a valid native CRS is never overwritten, even by a genuinely different .prj."""
    path = tmp_path / "case.las"
    _write_las(path, crs=pyproj.CRS.from_epsg(32618))
    (tmp_path / "case.prj").write_text(pyproj.CRS.from_epsg(4326).to_wkt(), encoding="utf-8")
    assert LASReader(path).read().crs == "EPSG:32618"


def test_native_absent_prj_valid(tmp_path: Path) -> None:
    path = tmp_path / "case.las"
    _write_las(path, crs=None)
    (tmp_path / "case.prj").write_text(pyproj.CRS.from_epsg(9377).to_wkt(), encoding="utf-8")
    assert LASReader(path).read().crs == "EPSG:9377"


def test_native_absent_prj_absent(tmp_path: Path) -> None:
    path = tmp_path / "case.las"
    _write_las(path, crs=None)
    assert LASReader(path).read().crs is None


def test_native_absent_prj_invalid(tmp_path: Path) -> None:
    path = tmp_path / "case.las"
    _write_las(path, crs=None)
    (tmp_path / "case.prj").write_text("garbage text, not WKT", encoding="utf-8")
    assert LASReader(path).read().crs is None


def test_native_absent_prj_esri_style(tmp_path: Path) -> None:
    path = tmp_path / "case.las"
    _write_las(path, crs=None)
    (tmp_path / "case.prj").write_text(pyproj.CRS.from_epsg(26910).to_wkt(WktVersion.WKT1_ESRI), encoding="utf-8")
    assert LASReader(path).read().crs == "EPSG:26910"


def test_missing_file_behavior_preserved(tmp_path: Path) -> None:
    with pytest.raises(PointCloudIOError):
        LASReader(tmp_path / "does_not_exist.las").read()


def test_coordinates_are_never_altered_by_crs_detection(tmp_path: Path) -> None:
    """The CRS integration must never touch X/Y/Z -- verified bit-for-bit identical between __iter__() and read()."""
    path = tmp_path / "case.las"
    _write_las(path, crs=pyproj.CRS.from_epsg(32618))
    (tmp_path / "case.prj").write_text(pyproj.CRS.from_epsg(4326).to_wkt(), encoding="utf-8")

    x_via_iter = next(iter(LASReader(path)))[PointAttribute.X].copy()
    x_via_read = next(iter(LASReader(path).read()))[PointAttribute.X]

    assert np.array_equal(x_via_iter, x_via_read)


# ----------------------------------------------------------------------
# LAZ -- same policy, confirmed independently (not just "inherits from LAS").
# ----------------------------------------------------------------------


def test_laz_native_present_prj_disagrees_native_wins(tmp_path: Path) -> None:
    path = tmp_path / "case.laz"
    _write_las(path, crs=pyproj.CRS.from_epsg(32618), laz=True)
    (tmp_path / "case.prj").write_text(pyproj.CRS.from_epsg(4326).to_wkt(), encoding="utf-8")
    assert LAZReader(path).read().crs == "EPSG:32618"


def test_laz_native_absent_prj_fallback(tmp_path: Path) -> None:
    path = tmp_path / "case.laz"
    _write_las(path, crs=None, laz=True)
    (tmp_path / "case.prj").write_text(pyproj.CRS.from_epsg(9377).to_wkt(), encoding="utf-8")
    assert LAZReader(path).read().crs == "EPSG:9377"


# ----------------------------------------------------------------------
# E57 -- same policy; native detection's own known 4-way ambiguity
# (empty/absent/corrupt/invalid all collapse to None) is confirmed to
# be treated uniformly as "try .prj".
# ----------------------------------------------------------------------


def test_e57_no_coordinate_metadata_falls_back_to_prj(tmp_path: Path) -> None:
    path = tmp_path / "case.e57"
    _write_e57(path)
    (tmp_path / "case.prj").write_text(pyproj.CRS.from_epsg(9377).to_wkt(), encoding="utf-8")

    assert E57Reader(path, chunk_size=1000).read().crs == "EPSG:9377"


def test_e57_no_coordinate_metadata_no_prj_is_none(tmp_path: Path) -> None:
    path = tmp_path / "case.e57"
    _write_e57(path)

    assert E57Reader(path, chunk_size=1000).read().crs is None
