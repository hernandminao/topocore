"""
Tests for topocore.io.landxml.crs_detection.detect_crs.

Same public contract as the LAS/E57 detectors (path -> CRS | None,
absence is never an error), built on top of LandXMLReader's own
already-verified <CoordinateSystem> extraction (LandXMLDocument.crs)
rather than re-parsing XML independently. Every fixture is a real,
hand-constructed LandXML file exercising one of the 3 real shapes
<CoordinateSystem>'s value can take (name="EPSG:...", epsgCode="...",
or desc="..." free text), confirmed directly against
LandXMLReader._read_units_and_crs()'s own priority order.
"""

from __future__ import annotations

from pathlib import Path

import pyproj
import pytest
from topocore.geodesy import CRS
from topocore.io.landxml.crs_detection import detect_crs

_BASE = """<?xml version="1.0"?>
<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
  <Units><Metric linearUnit="meter"/></Units>
{crs_element}
</LandXML>
"""


def _write(tmp_path: Path, name: str, crs_element: str) -> Path:
    path = tmp_path / name
    path.write_text(_BASE.format(crs_element=crs_element), encoding="utf-8")
    return path


@pytest.fixture
def landxml_with_epsg_name(tmp_path: Path) -> Path:
    return _write(tmp_path, "epsg_name.xml", '  <CoordinateSystem name="EPSG:32617"/>')


@pytest.fixture
def landxml_with_bare_epsg_code(tmp_path: Path) -> Path:
    return _write(tmp_path, "bare_epsg.xml", '  <CoordinateSystem epsgCode="2274"/>')


@pytest.fixture
def landxml_with_free_text_desc(tmp_path: Path) -> Path:
    return _write(
        tmp_path,
        "desc_only.xml",
        '  <CoordinateSystem desc="NAD83 State Plane Texas"/>',
    )


@pytest.fixture
def landxml_without_coordinate_system(tmp_path: Path) -> Path:
    return _write(tmp_path, "no_crs.xml", "")


@pytest.fixture
def landxml_with_nonexistent_epsg(tmp_path: Path) -> Path:
    return _write(tmp_path, "bad_epsg.xml", '  <CoordinateSystem name="EPSG:999999999"/>')


@pytest.fixture
def malformed_xml_file(tmp_path: Path) -> Path:
    path = tmp_path / "malformed.xml"
    path.write_text("<LandXML><Surfaces>", encoding="utf-8")  # unclosed, not well-formed
    return path


@pytest.fixture
def wrong_root_element_file(tmp_path: Path) -> Path:
    path = tmp_path / "wrong_root.xml"
    path.write_text("<NotLandXML></NotLandXML>", encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# The 3 real shapes <CoordinateSystem> takes, verified exactly.
# ----------------------------------------------------------------------


def test_epsg_prefixed_name_is_resolved(landxml_with_epsg_name: Path) -> None:
    detected = detect_crs(landxml_with_epsg_name)
    assert detected is not None
    assert isinstance(detected, CRS)
    assert detected.epsg == 32617


def test_bare_epsg_code_is_resolved(landxml_with_bare_epsg_code: Path) -> None:
    """epsgCode alone (no 'EPSG:' prefix) -- confirmed this is how LandXMLReader itself extracts it."""
    detected = detect_crs(landxml_with_bare_epsg_code)
    assert detected is not None
    assert detected.epsg == 2274


def test_free_text_desc_returns_none_not_a_guess(
    landxml_with_free_text_desc: Path,
) -> None:
    """A human-readable-only description ('NAD83 State Plane Texas') is not an EPSG code or WKT -- never guessed."""
    assert detect_crs(landxml_with_free_text_desc) is None


# ----------------------------------------------------------------------
# Absence and failure cases.
# ----------------------------------------------------------------------


def test_no_coordinate_system_element_returns_none(
    landxml_without_coordinate_system: Path,
) -> None:
    assert detect_crs(landxml_without_coordinate_system) is None


def test_epsg_shaped_but_nonexistent_code_returns_none(
    landxml_with_nonexistent_epsg: Path,
) -> None:
    """Correctly shaped as EPSG:NNNN, but not a real registered code -- resolved to None, not an exception."""
    assert detect_crs(landxml_with_nonexistent_epsg) is None


def test_malformed_xml_returns_none(malformed_xml_file: Path) -> None:
    assert detect_crs(malformed_xml_file) is None


def test_wrong_root_element_returns_none(wrong_root_element_file: Path) -> None:
    assert detect_crs(wrong_root_element_file) is None


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        detect_crs(tmp_path / "does_not_exist.xml")


def test_accepts_str_path_as_well_as_path_object(landxml_with_epsg_name: Path) -> None:
    detected = detect_crs(str(landxml_with_epsg_name))
    assert detected is not None
    assert detected.epsg == 32617


# ----------------------------------------------------------------------
# Reviewer-identified gaps, closed with real, verified evidence.
# ----------------------------------------------------------------------


def test_name_wins_over_conflicting_epsg_code(tmp_path: Path) -> None:
    """
    Confirmed directly: name and epsgCode present simultaneously,
    with genuinely conflicting values -- name always wins, inherited
    from LandXMLReader's own `name or epsgCode or desc` priority.
    """
    path = _write(
        tmp_path,
        "conflict.xml",
        '  <CoordinateSystem name="EPSG:32617" epsgCode="4326"/>',
    )
    detected = detect_crs(path)
    assert detected is not None
    assert detected.epsg == 32617


def test_full_wkt_embedded_in_name_is_detected(tmp_path: Path) -> None:
    """A full WKT string in `name` (not just a short EPSG code) is detected via the WKT fallback path."""
    wkt = pyproj.CRS.from_epsg(32617).to_wkt().replace('"', "&quot;")
    path = _write(tmp_path, "full_wkt.xml", f'  <CoordinateSystem name="{wkt}"/>')
    detected = detect_crs(path)
    assert detected is not None
    assert detected.epsg == 32617


@pytest.mark.parametrize(
    "namespace",
    [
        "http://www.landxml.org/schema/LandXML-1.0",
        "http://www.landxml.org/schema/LandXML-1.2",
        None,  # no namespace declared at all
    ],
)
def test_namespace_version_does_not_affect_detection(tmp_path: Path, namespace: str | None) -> None:
    """Confirmed directly: element matching is namespace-stripped throughout LandXMLReader -- version doesn't matter."""
    xmlns_attr = f'xmlns="{namespace}" ' if namespace else ""
    content = (
        f'<?xml version="1.0"?>\n<LandXML {xmlns_attr}version="1.2">\n'
        '  <Units><Metric linearUnit="meter"/></Units>\n'
        '  <CoordinateSystem name="EPSG:32617"/>\n</LandXML>\n'
    )
    path = tmp_path / f"ns_{namespace is not None}.xml"
    path.write_text(content, encoding="utf-8")

    detected = detect_crs(path)
    assert detected is not None
    assert detected.epsg == 32617


def test_nested_coordinate_system_is_a_real_confirmed_limitation(
    tmp_path: Path,
) -> None:
    """
    Confirmed directly: <CoordinateSystem> nested inside another
    element (e.g. <Project>) is missed -- children() only searches
    direct children of the <LandXML> root. This is a genuine,
    current limitation of the underlying reader, not a bug in
    detect_crs() itself; this test documents and locks in the
    observed behavior rather than silently letting it drift.
    """
    path = _write(
        tmp_path,
        "nested.xml",
        '  <Project name="Test">\n    <CoordinateSystem name="EPSG:32617"/>\n  </Project>',
    )
    assert detect_crs(path) is None


def test_correctly_declared_non_ascii_encoding_is_parsed_correctly(
    tmp_path: Path,
) -> None:
    """A genuinely non-UTF-8 file (ISO-8859-1, with non-ASCII content) whose XML declaration matches its real encoding."""
    content = (
        '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
        '<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">\n'
        '  <Units><Metric linearUnit="meter"/></Units>\n'
        '  <CoordinateSystem name="EPSG:32617" desc="Proyecto en Bogotá, ñoño"/>\n'
        "</LandXML>\n"
    )
    path = tmp_path / "latin1.xml"
    path.write_bytes(content.encode("iso-8859-1"))

    detected = detect_crs(path)
    assert detected is not None
    assert detected.epsg == 32617


def test_encoding_mismatch_fails_safely_returns_none(tmp_path: Path) -> None:
    """
    Confirmed directly: a file whose XML declaration implies UTF-8
    (the default, no encoding attribute) but whose actual bytes are
    ISO-8859-1 (with genuine non-ASCII content) raises
    LandXMLParseError internally -- caught here and returned as
    None, never a crash.
    """
    content = (
        '<?xml version="1.0"?>\n'
        '<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">\n'
        '  <Units><Metric linearUnit="meter"/></Units>\n'
        '  <CoordinateSystem name="EPSG:32617" desc="Proyecto en Bogotá, ñoño"/>\n'
        "</LandXML>\n"
    )
    path = tmp_path / "mismatched_encoding.xml"
    path.write_bytes(content.encode("iso-8859-1"))  # bytes don't match the implicit UTF-8 declaration

    assert detect_crs(path) is None
