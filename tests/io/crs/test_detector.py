"""
Tests for topocore.io.crs.detector.detect_crs -- the single,
format-agnostic facade dispatching to whichever per-format strategy
applies. This module does not re-verify each format's own detection
logic in depth (already covered by test_geoid_grid.py-style suites
per format) -- it verifies the DISPATCH itself: the right strategy is
chosen for each extension, and the facade's own explicit contract
(missing file raises, unrecognized extension returns None, native
priority over .prj fallback is preserved) holds end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import laspy  # type: ignore[import-untyped]
import numpy as np
import pye57  # type: ignore[import-untyped]
import pyproj
import pytest
from topocore.io.crs import detect_crs

_PLY_CONTENT = """ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
end_header
1.0 2.0 3.0
"""

_LANDXML_TEMPLATE = """<?xml version="1.0"?>
<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
  <Units><Metric linearUnit="meter"/></Units>
{crs_element}
</LandXML>
"""


def _write_las(path: Path, *, crs: pyproj.CRS | None) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    if crs is not None:
        header.add_crs(crs)
    data = laspy.LasData(header)
    n = 5
    rng = np.random.default_rng(0)
    data.x = rng.uniform(500000, 500100, n)
    data.y = rng.uniform(4000000, 4000100, n)
    data.z = rng.uniform(100, 200, n)
    data.write(str(path))


def _write_e57_empty_metadata(path: Path) -> None:
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
# LAS/LAZ -- native priority, .prj fallback.
# ----------------------------------------------------------------------


def test_las_native_crs_used_directly(tmp_path: Path) -> None:
    path = tmp_path / "survey.las"
    _write_las(path, crs=pyproj.CRS.from_epsg(32618))

    detected = detect_crs(path)
    assert detected is not None
    assert detected.epsg == 32618


def test_las_no_native_falls_back_to_prj(tmp_path: Path) -> None:
    path = tmp_path / "survey.las"
    _write_las(path, crs=None)
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(9377).to_wkt(), encoding="utf-8")

    detected = detect_crs(path)
    assert detected is not None
    assert detected.epsg == 9377


def test_las_native_never_overridden_by_disagreeing_prj(tmp_path: Path) -> None:
    path = tmp_path / "survey.las"
    _write_las(path, crs=pyproj.CRS.from_epsg(32618))
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(4326).to_wkt(), encoding="utf-8")

    detected = detect_crs(path)
    assert detected is not None
    assert detected.epsg == 32618


def test_laz_extension_uses_the_same_las_strategy(tmp_path: Path) -> None:
    path = tmp_path / "survey.laz"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_crs(pyproj.CRS.from_epsg(32618))
    data = laspy.LasData(header)
    n = 5
    rng = np.random.default_rng(0)
    data.x = rng.uniform(500000, 500100, n)
    data.y = rng.uniform(4000000, 4000100, n)
    data.z = rng.uniform(100, 200, n)
    data.write(str(path), laz_backend=laspy.LazBackend.Lazrs)

    detected = detect_crs(path)
    assert detected is not None
    assert detected.epsg == 32618


# ----------------------------------------------------------------------
# E57 -- native priority, .prj fallback.
# ----------------------------------------------------------------------


def test_e57_no_coordinate_metadata_falls_back_to_prj(tmp_path: Path) -> None:
    path = tmp_path / "scan.e57"
    _write_e57_empty_metadata(path)
    (tmp_path / "scan.prj").write_text(pyproj.CRS.from_epsg(4326).to_wkt(), encoding="utf-8")

    detected = detect_crs(path)
    assert detected is not None
    assert detected.epsg == 4326


def test_e57_no_metadata_no_prj_is_none(tmp_path: Path) -> None:
    path = tmp_path / "scan.e57"
    _write_e57_empty_metadata(path)

    assert detect_crs(path) is None


# ----------------------------------------------------------------------
# LandXML -- native only, no .prj fallback.
# ----------------------------------------------------------------------


def test_landxml_native_used(tmp_path: Path) -> None:
    path = tmp_path / "surface.xml"
    path.write_text(
        _LANDXML_TEMPLATE.format(crs_element='  <CoordinateSystem name="EPSG:32617"/>'),
        encoding="utf-8",
    )

    detected = detect_crs(path)
    assert detected is not None
    assert detected.epsg == 32617


def test_landxml_no_native_no_prj_fallback(tmp_path: Path) -> None:
    """Confirms the explicit design decision: .xml gets no .prj fallback, unlike LAS/LAZ/E57."""
    path = tmp_path / "surface.xml"
    path.write_text(_LANDXML_TEMPLATE.format(crs_element=""), encoding="utf-8")
    (tmp_path / "surface.prj").write_text(pyproj.CRS.from_epsg(32617).to_wkt(), encoding="utf-8")

    assert detect_crs(path) is None


# ----------------------------------------------------------------------
# PLY/XYZ/CSV/PTS -- .prj only.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extension", "content"),
    [
        ("ply", _PLY_CONTENT),
        ("xyz", "1.0 2.0 3.0\n"),
        ("csv", "x,y,z\n1.0,2.0,3.0\n"),
        ("pts", "1\n1.0 2.0 3.0\n"),
    ],
)
def test_prj_only_formats_use_the_shared_external_detector(tmp_path: Path, extension: str, content: str) -> None:
    path = tmp_path / f"data.{extension}"
    path.write_text(content, encoding="utf-8")
    (tmp_path / "data.prj").write_text(pyproj.CRS.from_epsg(32618).to_wkt(), encoding="utf-8")

    detected = detect_crs(path)
    assert detected is not None
    assert detected.epsg == 32618


@pytest.mark.parametrize("extension", ["ply", "xyz", "csv", "pts"])
def test_prj_only_formats_without_prj_return_none(tmp_path: Path, extension: str) -> None:
    path = tmp_path / f"data.{extension}"
    path.write_text("some content", encoding="utf-8")

    assert detect_crs(path) is None


# ----------------------------------------------------------------------
# Unrecognized extension and missing file.
# ----------------------------------------------------------------------


def test_unrecognized_extension_returns_none_not_an_error(tmp_path: Path) -> None:
    path = tmp_path / "data.unknownformat"
    path.write_text("some content", encoding="utf-8")

    assert detect_crs(path) is None


def test_missing_file_raises_regardless_of_extension(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        detect_crs(tmp_path / "does_not_exist.las")


def test_missing_file_with_unrecognized_extension_still_raises(tmp_path: Path) -> None:
    """Confirms missing-file is checked before extension dispatch -- never confused with 'unrecognized format'."""
    with pytest.raises(FileNotFoundError, match="not found"):
        detect_crs(tmp_path / "does_not_exist.unknownformat")


def test_accepts_str_path_as_well_as_path_object(tmp_path: Path) -> None:
    path = tmp_path / "survey.las"
    _write_las(path, crs=pyproj.CRS.from_epsg(32618))

    detected = detect_crs(str(path))
    assert detected is not None
    assert detected.epsg == 32618


# ----------------------------------------------------------------------
# Reviewer-identified gaps, closed with real, verified evidence.
# ----------------------------------------------------------------------


def test_laz_no_native_falls_back_to_prj_through_the_facade(tmp_path: Path) -> None:
    path = tmp_path / "survey.laz"
    header = laspy.LasHeader(point_format=3, version="1.2")
    data = laspy.LasData(header)
    n = 5
    rng = np.random.default_rng(0)
    data.x = rng.uniform(500000, 500100, n)
    data.y = rng.uniform(4000000, 4000100, n)
    data.z = rng.uniform(100, 200, n)
    data.write(str(path), laz_backend=laspy.LazBackend.Lazrs)
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(9377).to_wkt(), encoding="utf-8")

    detected = detect_crs(path)
    assert detected is not None
    assert detected.epsg == 9377


def test_e57_native_never_overridden_by_disagreeing_prj(tmp_path: Path) -> None:
    path = tmp_path / "scan.e57"
    wkt_native = pyproj.CRS.from_epsg(32618).to_wkt()
    image_file = pye57.libe57.ImageFile(str(path), "w")
    root = image_file.root()
    root.set(
        "formatName",
        pye57.libe57.StringNode(image_file, "ASTM E57 3D Imaging Data File"),
    )
    root.set(
        "guid",
        pye57.libe57.StringNode(image_file, "{00000000-0000-0000-0000-000000000000}"),
    )
    root.set("coordinateMetadata", pye57.libe57.StringNode(image_file, wkt_native))
    image_file.close()
    (tmp_path / "scan.prj").write_text(pyproj.CRS.from_epsg(4326).to_wkt(), encoding="utf-8")

    detected = detect_crs(path)
    assert detected is not None
    assert detected.epsg == 32618


def test_las_native_never_overridden_by_empty_prj(tmp_path: Path) -> None:
    """Distinct from the existing 'disagreeing .prj' test: this specifically covers an EMPTY sidecar, not just a differing one."""
    path = tmp_path / "survey.las"
    _write_las(path, crs=pyproj.CRS.from_epsg(32618))
    (tmp_path / "survey.prj").write_text("", encoding="utf-8")

    detected = detect_crs(path)
    assert detected is not None
    assert detected.epsg == 32618


def test_facade_respects_the_same_prj_naming_convention_as_the_detector(
    tmp_path: Path,
) -> None:
    """
    ExternalCRSDetector.detect() itself already confirms only the
    exact with_suffix('.prj') candidate is consulted (see
    test_external.py) -- this test re-confirms the SAME outcome
    through the facade specifically, since the facade's own dispatch
    logic is a separate code path that could in principle have
    diverged.
    """
    path = tmp_path / "data.xyz"
    path.write_text("1.0 2.0 3.0\n", encoding="utf-8")
    (tmp_path / "data.xyz.prj").write_text(
        pyproj.CRS.from_epsg(32618).to_wkt(), encoding="utf-8"
    )  # wrong candidate name

    assert detect_crs(path) is None


def test_never_reads_point_data_confirmed_with_a_real_spy(tmp_path: Path) -> None:
    """
    Confirmed directly, not assumed: chunk_iterator (the method that
    actually reads LAS/LAZ point data) is never invoked, and the main
    file is opened exactly once, regardless of point count.
    """
    from unittest.mock import patch

    path = tmp_path / "large.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_crs(pyproj.CRS.from_epsg(32618))
    data = laspy.LasData(header)
    n = 1000
    rng = np.random.default_rng(0)
    data.x = rng.uniform(500000, 500100, n)
    data.y = rng.uniform(4000000, 4000100, n)
    data.z = rng.uniform(100, 200, n)
    data.write(str(path))

    chunk_iterator_called = {"value": False}
    original_chunk_iterator = laspy.LasReader.chunk_iterator

    def spy_chunk_iterator(self: object, *args: object, **kwargs: object) -> object:
        chunk_iterator_called["value"] = True
        return original_chunk_iterator(self, *args, **kwargs)  # type: ignore[arg-type]

    open_call_count = {"value": 0}
    original_open = laspy.open

    def spy_open(*args: object, **kwargs: object) -> object:
        open_call_count["value"] += 1
        return original_open(*args, **kwargs)  # type: ignore[arg-type]

    with (
        patch.object(laspy.LasReader, "chunk_iterator", spy_chunk_iterator),
        patch("laspy.open", spy_open),
    ):
        detected = detect_crs(path)

    assert detected is not None
    assert detected.epsg == 32618
    assert chunk_iterator_called["value"] is False
    assert open_call_count["value"] == 1
