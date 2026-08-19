"""
Shared fixtures for topocore.io.landxml tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from topocore.geometry.point3d import Point3D
from topocore.io.landxml.models import LandXMLDocument, NamedPointGroup, NamedSurface
from topocore.survey.models import SurveyPoint, SurveyPointSet
from topocore.terrain.tin import TIN


@pytest.fixture
def two_triangle_square_tin() -> TIN:
    """
    Same fixed non-Delaunay-ambiguous diagonal used in the
    TIN.from_mesh() tests: triangle 0=(0,1,2), triangle 1=(1,3,2).
    """
    vertices = (
        Point3D(0.0, 0.0, 10.0),
        Point3D(10.0, 0.0, 11.0),
        Point3D(0.0, 10.0, 12.0),
        Point3D(10.0, 10.0, 13.0),
    )
    simplices = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return TIN.from_mesh(vertices, simplices)


@pytest.fixture
def sample_document(two_triangle_square_tin: TIN) -> LandXMLDocument:
    surface = NamedSurface(name="Existing", tin=two_triangle_square_tin, desc="Terreno natural")

    points = SurveyPointSet(
        points=(
            SurveyPoint(id="1", x=100.0, y=200.0, z=50.0, code="MOJON"),
            SurveyPoint(id="2", x=101.5, y=201.5, z=50.5, code="ARBOL"),
            SurveyPoint(id="3", x=102.0, y=199.0, z=49.8, code=None),
        )
    )
    group = NamedPointGroup(name="Control", points=points, desc="Puntos de control")

    return LandXMLDocument(surfaces=(surface,), point_groups=(group,))


@pytest.fixture
def tmp_landxml_path(tmp_path: Path) -> Path:
    return tmp_path / "sample.xml"
