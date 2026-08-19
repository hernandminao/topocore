"""
Round-trip tests: LandXMLDocument -> file -> LandXMLDocument.

This is the core correctness guarantee of PR18B: what LandXMLWriter
writes, LandXMLReader must read back as an equivalent document --
including preserving the exact (possibly non-Delaunay) <Faces>
connectivity, per TIN.from_mesh().
"""

from __future__ import annotations

from pathlib import Path

import pytest

from topocore.io.landxml.models import LandXMLDocument
from topocore.io.landxml.reader import LandXMLReader
from topocore.io.landxml.writer import LandXMLWriter


def test_roundtrip_preserves_surface_name_and_desc(
    sample_document: LandXMLDocument,
    tmp_landxml_path: Path,
) -> None:
    LandXMLWriter(tmp_landxml_path).write(sample_document)
    result = LandXMLReader(tmp_landxml_path).read()

    assert len(result.surfaces) == 1
    assert result.surfaces[0].name == "Existing"
    assert result.surfaces[0].desc == "Terreno natural"


def test_roundtrip_preserves_tin_vertex_count_and_coordinates(
    sample_document: LandXMLDocument,
    tmp_landxml_path: Path,
) -> None:
    LandXMLWriter(tmp_landxml_path).write(sample_document)
    result = LandXMLReader(tmp_landxml_path).read()

    original_tin = sample_document.surfaces[0].tin
    round_tripped_tin = result.surfaces[0].tin

    assert round_tripped_tin.vertex_count == original_tin.vertex_count

    for original_vertex, round_tripped_vertex in zip(original_tin.vertices, round_tripped_tin.vertices, strict=True):
        assert round_tripped_vertex.x == pytest.approx(original_vertex.x)
        assert round_tripped_vertex.y == pytest.approx(original_vertex.y)
        assert round_tripped_vertex.z == pytest.approx(original_vertex.z)


def test_roundtrip_preserves_exact_non_delaunay_face_connectivity(
    sample_document: LandXMLDocument,
    tmp_landxml_path: Path,
) -> None:
    """
    The whole reason TIN.from_mesh() exists: the specific diagonal
    chosen in the fixture (0,1,2)/(1,3,2) must survive a full
    write -> read cycle unchanged, not be replaced by whatever a
    fresh Delaunay triangulation of the same 4 points would pick.
    """
    LandXMLWriter(tmp_landxml_path).write(sample_document)
    result = LandXMLReader(tmp_landxml_path).read()

    original_tin = sample_document.surfaces[0].tin
    round_tripped_tin = result.surfaces[0].tin

    for index in range(original_tin.triangle_count):
        original_triangle = original_tin.triangle_vertices(index)
        round_tripped_triangle = round_tripped_tin.triangle_vertices(index)

        for original_vertex, round_tripped_vertex in zip(original_triangle, round_tripped_triangle, strict=True):
            assert round_tripped_vertex.x == pytest.approx(original_vertex.x)
            assert round_tripped_vertex.y == pytest.approx(original_vertex.y)
            assert round_tripped_vertex.z == pytest.approx(original_vertex.z)


def test_roundtrip_preserves_point_group(
    sample_document: LandXMLDocument,
    tmp_landxml_path: Path,
) -> None:
    LandXMLWriter(tmp_landxml_path).write(sample_document)
    result = LandXMLReader(tmp_landxml_path).read()

    assert len(result.point_groups) == 1
    group = result.point_groups[0]
    original_group = sample_document.point_groups[0]

    assert group.name == "Control"
    assert group.desc == "Puntos de control"
    assert len(group.points) == len(original_group.points)

    for original_point, round_tripped_point in zip(original_group.points, group.points, strict=True):
        assert round_tripped_point.id == original_point.id
        assert round_tripped_point.code == original_point.code
        assert round_tripped_point.x == pytest.approx(original_point.x)
        assert round_tripped_point.y == pytest.approx(original_point.y)
        assert round_tripped_point.z == pytest.approx(original_point.z)


def test_roundtrip_preserves_point_order(
    sample_document: LandXMLDocument,
    tmp_landxml_path: Path,
) -> None:
    """
    SurveyPointSet order is meaningful (topocore.features linework
    grouping) -- must not be silently reordered.
    """
    LandXMLWriter(tmp_landxml_path).write(sample_document)
    result = LandXMLReader(tmp_landxml_path).read()

    original_ids = [point.id for point in sample_document.point_groups[0].points]
    round_tripped_ids = [point.id for point in result.point_groups[0].points]

    assert round_tripped_ids == original_ids


def test_write_returns_accurate_report(
    sample_document: LandXMLDocument,
    tmp_landxml_path: Path,
) -> None:
    report = LandXMLWriter(tmp_landxml_path).write(sample_document)

    assert report.surface_count == 1
    assert report.point_group_count == 1
    assert report.triangle_count == 2
    assert report.point_count == 4 + 3  # 4 TIN vertices + 3 survey points
    assert report.output_path == tmp_landxml_path


def test_read_with_report_returns_accurate_counts(
    sample_document: LandXMLDocument,
    tmp_landxml_path: Path,
) -> None:
    LandXMLWriter(tmp_landxml_path).write(sample_document)
    document, report = LandXMLReader(tmp_landxml_path).read_with_report()

    assert report.surface_count == 1
    assert report.point_group_count == 1
    assert report.triangle_count == 2
    assert document.surfaces[0].tin.triangle_count == 2
