"""
Tests confirming LandXMLWriter validates before writing anything to
disk -- required per the frozen PR18B contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from topocore.io.landxml.exceptions import LandXMLValidationError
from topocore.io.landxml.models import LandXMLDocument, NamedSurface
from topocore.io.landxml.writer import LandXMLWriter
from topocore.terrain.tin import TIN


def test_write_rejects_duplicated_surface_names_before_touching_disk(
    two_triangle_square_tin: TIN,
    tmp_landxml_path: Path,
) -> None:
    document = LandXMLDocument(
        surfaces=(
            NamedSurface(name="Dup", tin=two_triangle_square_tin),
            NamedSurface(name="Dup", tin=two_triangle_square_tin),
        )
    )

    with pytest.raises(LandXMLValidationError):
        LandXMLWriter(tmp_landxml_path).write(document)

    assert not tmp_landxml_path.exists()


def test_write_empty_document_produces_valid_minimal_file(
    tmp_landxml_path: Path,
) -> None:
    document = LandXMLDocument()

    report = LandXMLWriter(tmp_landxml_path).write(document)

    assert tmp_landxml_path.exists()
    assert report.surface_count == 0
    assert report.point_group_count == 0
