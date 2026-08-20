"""Shared fixtures for topocore.gpkg tests -- PR19."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from topocore.features.models import (
    Feature,
    FeatureCategory,
    FeatureGeometry,
    FeatureType,
    GeometryType,
)


def make_feature(
    fid: int,
    ftype: FeatureType,
    category: FeatureCategory,
    geom: FeatureGeometry,
    *,
    attributes: Mapping[str, Any] = MappingProxyType({}),
) -> Feature:
    return Feature(
        feature_id=fid,
        category=category,
        feature_type=ftype,
        geometry=geom,
        attributes=attributes,
    )


@pytest.fixture
def point_feature() -> Feature:
    geom = FeatureGeometry(
        geometry_type=GeometryType.POINT,
        vertices=np.array([[500123.456, 4000456.789, 100.5]]),
    )
    return make_feature(
        1,
        FeatureType.TREE,
        FeatureCategory.VEGETATION,
        geom,
        attributes=MappingProxyType({"survey_code": "TR01", "height_m": 5.5}),
    )


@pytest.fixture
def polygon_feature() -> Feature:
    verts = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0]])
    geom = FeatureGeometry(geometry_type=GeometryType.POLYGON, vertices=verts, closed=True)
    return make_feature(2, FeatureType.BUILDING, FeatureCategory.BUILDING, geom)


@pytest.fixture
def polyline_feature() -> Feature:
    verts = np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 5.0], [20.0, 0.0, 2.0]])
    geom = FeatureGeometry(geometry_type=GeometryType.POLYLINE, vertices=verts)
    return make_feature(3, FeatureType.BREAKLINE, FeatureCategory.TERRAIN, geom)


@pytest.fixture
def mesh_feature() -> Feature:
    verts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 1.0], [0.0, 10.0, 2.0], [10.0, 10.0, 3.0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    geom = FeatureGeometry(geometry_type=GeometryType.MESH, vertices=verts, faces=faces)
    return make_feature(
        4,
        FeatureType.ROOF,
        FeatureCategory.BUILDING,
        geom,
        attributes=MappingProxyType(
            {
                "survey_code": "RF01",
                "survey_name": "Roof A",
                "cad_layer": "ROOFS",
                "material": "tile",
                "area_m2": 123.4,
            }
        ),
    )
