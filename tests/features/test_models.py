"""
Coverage audit tests for topocore.features.models (FeatureGeometry,
FeatureMetadata, Feature, FeatureCollection).

PR22 coverage hardening. Confirmed via grep before writing anything:
this is a genuinely central, widely-adopted data model -- consumers
span gpkg/, dxf/, geodesy/transform.py, and essentially every
subdirectory of features/ (infrastructure, catalogs, drainage,
buildings, vegetation, terrain) -- unlike several other modules
audited in this session, there is no "orphaned" question here at all.

Confirmed directly, before writing tests, that _EXPECTED_GEOMETRY
covers all 84 FeatureType values with no gaps -- the geometry-type
validation in Feature.__post_init__ is genuinely active for every
feature type, not silently skipped for any of them.

WALL/RETAINING_WALL/ROOF/TREE/SHRUB are documented as deliberately
accepting more than one GeometryType (e.g. WALL accepts both
POLYLINE and POLYGON, since PR15's detector and a field-surveyed
MURO/MURCONT represent the same physical object differently) --
verified directly that both representations are genuinely accepted
for WALL, confirmed via real construction of both.

feature_id=None, metadata=None (both defaults), and
source_point_indices are confirmed genuinely used in production (via
grep: features/_shared.py, buildings/walls.py,
buildings/retaining_walls.py, buildings/roofs.py all construct
Feature with source_point_indices; feature_id=None is the natural
pre-normalize_ids() state for freshly detected features).
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.features.exceptions import GeometryError
from topocore.features.models import (
    Feature,
    FeatureCategory,
    FeatureCollection,
    FeatureGeometry,
    FeatureMetadata,
    FeatureType,
    GeometryType,
)


def _polygon_geometry() -> FeatureGeometry:
    return FeatureGeometry(
        geometry_type=GeometryType.POLYGON,
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
    )


def _polyline_geometry() -> FeatureGeometry:
    return FeatureGeometry(
        geometry_type=GeometryType.POLYLINE,
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )


# ----------------------------------------------------------------------
# FeatureGeometry -- happy paths for all 4 GeometryType values.
# ----------------------------------------------------------------------


def test_point_geometry_happy_path() -> None:
    geometry = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[1.0, 2.0, 3.0]]))
    assert geometry.vertex_count == 1
    assert geometry.bounds == (1.0, 2.0, 3.0, 1.0, 2.0, 3.0)


def test_polygon_geometry_happy_path() -> None:
    geometry = _polygon_geometry()
    assert geometry.vertex_count == 3


def test_mesh_geometry_happy_path() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    geometry = FeatureGeometry(geometry_type=GeometryType.MESH, vertices=vertices, faces=faces)
    assert geometry.vertex_count == 3


# ----------------------------------------------------------------------
# FeatureGeometry -- all 6 validation branches.
# ----------------------------------------------------------------------


def test_geometry_rejects_wrong_shape() -> None:
    with pytest.raises(GeometryError, match=r"shape \(n, 3\)"):
        FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([0.0, 0.0, 0.0]))


def test_geometry_rejects_too_few_vertices_for_type() -> None:
    with pytest.raises(GeometryError, match="at least 3 vertices"):
        FeatureGeometry(
            geometry_type=GeometryType.POLYGON,
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )


def test_geometry_rejects_non_finite_vertices() -> None:
    with pytest.raises(GeometryError, match="finite coordinates"):
        FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[np.nan, 0.0, 0.0]]))


def test_mesh_requires_faces() -> None:
    with pytest.raises(GeometryError, match="requires `faces`"):
        FeatureGeometry(geometry_type=GeometryType.MESH, vertices=_polygon_geometry().vertices)


def test_mesh_rejects_wrong_faces_shape() -> None:
    with pytest.raises(GeometryError, match=r"shape \(m, 3\)"):
        FeatureGeometry(
            geometry_type=GeometryType.MESH,
            vertices=_polygon_geometry().vertices,
            faces=np.array([[0, 1]], dtype=np.int64),
        )


def test_mesh_rejects_out_of_range_face_indices() -> None:
    with pytest.raises(GeometryError, match="out of range"):
        FeatureGeometry(
            geometry_type=GeometryType.MESH,
            vertices=_polygon_geometry().vertices,
            faces=np.array([[0, 1, 10]], dtype=np.int64),
        )


def test_non_mesh_rejects_faces() -> None:
    with pytest.raises(GeometryError, match="only valid for MESH"):
        FeatureGeometry(
            geometry_type=GeometryType.POLYGON,
            vertices=_polygon_geometry().vertices,
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )


# ----------------------------------------------------------------------
# FeatureMetadata -- extra becomes immutable.
# ----------------------------------------------------------------------


def test_metadata_extra_becomes_immutable() -> None:
    metadata = FeatureMetadata(detector="TestDetector", extra={"note": "test"})
    with pytest.raises(TypeError):
        metadata.extra["note"] = "changed"  # type: ignore[index]


# ----------------------------------------------------------------------
# Feature -- happy path, dual-geometry acceptance, validation, and
# genuinely-used defaults (feature_id=None, metadata=None,
# source_point_indices).
# ----------------------------------------------------------------------


def test_feature_happy_path() -> None:
    feature = Feature(
        feature_id=1,
        category=FeatureCategory.BUILDING,
        feature_type=FeatureType.BUILDING,
        geometry=_polygon_geometry(),
    )
    assert feature.confidence == 1.0
    assert feature.metadata is None


def test_wall_accepts_both_polyline_and_polygon_representations() -> None:
    """PR15's detector builds a POLYGON footprint; a field-surveyed MURO/MURCONT is a POLYLINE -- same object, both valid."""
    wall_polygon = Feature(
        feature_id=1,
        category=FeatureCategory.BUILDING,
        feature_type=FeatureType.WALL,
        geometry=_polygon_geometry(),
    )
    wall_polyline = Feature(
        feature_id=2,
        category=FeatureCategory.BUILDING,
        feature_type=FeatureType.WALL,
        geometry=_polyline_geometry(),
    )
    assert wall_polygon.geometry.geometry_type == GeometryType.POLYGON
    assert wall_polyline.geometry.geometry_type == GeometryType.POLYLINE


def test_feature_rejects_disallowed_geometry_type() -> None:
    with pytest.raises(GeometryError, match="expects one of"):
        Feature(
            feature_id=1,
            category=FeatureCategory.BUILDING,
            feature_type=FeatureType.BUILDING,
            geometry=_polyline_geometry(),
        )


def test_feature_rejects_confidence_out_of_range() -> None:
    with pytest.raises(GeometryError, match="confidence must be in"):
        Feature(
            feature_id=1,
            category=FeatureCategory.BUILDING,
            feature_type=FeatureType.BUILDING,
            geometry=_polygon_geometry(),
            confidence=1.5,
        )


def test_feature_attributes_become_immutable() -> None:
    feature = Feature(
        feature_id=1,
        category=FeatureCategory.BUILDING,
        feature_type=FeatureType.BUILDING,
        geometry=_polygon_geometry(),
        attributes={"height": 5.0},
    )
    with pytest.raises(TypeError):
        feature.attributes["height"] = 10.0  # type: ignore[index]


def test_feature_id_none_is_valid_pre_normalization_state() -> None:
    feature = Feature(
        feature_id=None,
        category=FeatureCategory.BUILDING,
        feature_type=FeatureType.BUILDING,
        geometry=_polygon_geometry(),
    )
    assert feature.feature_id is None


def test_feature_source_point_indices() -> None:
    indices = np.array([0, 5, 10], dtype=np.int64)
    feature = Feature(
        feature_id=1,
        category=FeatureCategory.BUILDING,
        feature_type=FeatureType.BUILDING,
        geometry=_polygon_geometry(),
        source_point_indices=indices,
    )
    np.testing.assert_array_equal(feature.source_point_indices, indices)


def test_all_feature_types_have_expected_geometry_mapping() -> None:
    """Confirms _EXPECTED_GEOMETRY has no gaps -- every FeatureType's own geometry validation is genuinely active."""
    from topocore.features.models import _EXPECTED_GEOMETRY

    assert set(_EXPECTED_GEOMETRY.keys()) == set(FeatureType)


# ----------------------------------------------------------------------
# FeatureCollection.
# ----------------------------------------------------------------------


def _make_collection() -> FeatureCollection:
    fc = FeatureCollection()
    fc.add(
        Feature(
            feature_id=None,
            category=FeatureCategory.BUILDING,
            feature_type=FeatureType.BUILDING,
            geometry=_polygon_geometry(),
        )
    )
    fc.add(
        Feature(
            feature_id=None,
            category=FeatureCategory.BUILDING,
            feature_type=FeatureType.WALL,
            geometry=_polyline_geometry(),
        )
    )
    return fc


def test_collection_len_and_iter() -> None:
    fc = _make_collection()
    assert len(fc) == 2
    assert len(list(fc)) == 2


def test_collection_by_type_and_by_category() -> None:
    fc = _make_collection()
    assert len(fc.by_type(FeatureType.BUILDING)) == 1
    assert len(fc.by_category(FeatureCategory.BUILDING)) == 2


def test_collection_confidence_array() -> None:
    fc = _make_collection()
    np.testing.assert_array_equal(fc.confidence_array(), [1.0, 1.0])


def test_collection_normalize_ids_assigns_sequential_ids_from_one() -> None:
    fc = _make_collection()
    fc.normalize_ids()
    assert [f.feature_id for f in fc] == [1, 2]


def test_collection_extend() -> None:
    fc = _make_collection()
    other = _make_collection()
    fc.extend(other)
    assert len(fc) == 4


def test_collection_bounds() -> None:
    fc = _make_collection()
    bounds = fc.bounds
    assert bounds is not None
    assert bounds[0] <= 0.0
    assert bounds[3] >= 1.0


def test_empty_collection_bounds_is_none() -> None:
    assert FeatureCollection().bounds is None
