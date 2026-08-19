"""
Regression suite for topocore.geodesy.transform -- transform_survey,
transform_feature_collection, transform_point_cloud -- PR19.

All three verified against the SAME source point (500000, 4500000,
100.0 in UTM 17N) transformed to WGS84, confirming they agree with
each other (and with the isolated CoordinateTransformer tests in
test_transformer_from_operation.py) -- not just that each one
"doesn't crash" in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.features.models import (
    Feature,
    FeatureCategory,
    FeatureCollection,
    FeatureGeometry,
    FeatureType,
    GeometryType,
)
from topocore.geodesy.crs import CRS
from topocore.geodesy.transform import (
    transform_feature_collection,
    transform_point_cloud,
    transform_survey,
)
from topocore.geodesy.transformer import CoordinateTransformer
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.survey.models import SurveyPoint, SurveyPointSet

# Shared reference: the same physical point, expressed in UTM 17N,
# transformed to WGS84. Independently confirmed (session audit) to
# match across all three functions below.
_SOURCE_X, _SOURCE_Y, _SOURCE_Z = 500000.0, 4500000.0, 100.0
_EXPECTED_LON = pytest.approx(-81.0, abs=1e-6)
_EXPECTED_LAT = pytest.approx(40.65085651557158, abs=1e-6)


@pytest.fixture
def utm_to_wgs84() -> CoordinateTransformer:
    return CoordinateTransformer(CRS.from_epsg(32617), CRS.from_epsg(4326))


# ----------------------------------------------------------------------
# transform_survey
# ----------------------------------------------------------------------


def test_transform_survey_transforms_coordinates(
    utm_to_wgs84: CoordinateTransformer,
) -> None:
    survey = SurveyPointSet(points=(SurveyPoint(id="P1", x=_SOURCE_X, y=_SOURCE_Y, z=_SOURCE_Z, code="MOJON"),))

    result = transform_survey(survey, utm_to_wgs84)

    assert result.points[0].x == _EXPECTED_LON
    assert result.points[0].y == _EXPECTED_LAT
    assert result.points[0].z == pytest.approx(_SOURCE_Z)


def test_transform_survey_preserves_id_and_code(
    utm_to_wgs84: CoordinateTransformer,
) -> None:
    survey = SurveyPointSet(points=(SurveyPoint(id="P42", x=_SOURCE_X, y=_SOURCE_Y, z=_SOURCE_Z, code="ARBOL"),))

    result = transform_survey(survey, utm_to_wgs84)

    assert result.points[0].id == "P42"
    assert result.points[0].code == "ARBOL"


def test_transform_survey_preserves_point_order(
    utm_to_wgs84: CoordinateTransformer,
) -> None:
    survey = SurveyPointSet(
        points=(
            SurveyPoint(id="A", x=_SOURCE_X, y=_SOURCE_Y, z=_SOURCE_Z),
            SurveyPoint(id="B", x=_SOURCE_X + 100, y=_SOURCE_Y + 100, z=_SOURCE_Z + 5),
            SurveyPoint(id="C", x=_SOURCE_X + 200, y=_SOURCE_Y + 200, z=_SOURCE_Z + 10),
        )
    )

    result = transform_survey(survey, utm_to_wgs84)

    assert [p.id for p in result] == ["A", "B", "C"]


def test_transform_survey_does_not_mutate_original(
    utm_to_wgs84: CoordinateTransformer,
) -> None:
    survey = SurveyPointSet(points=(SurveyPoint(id="P1", x=_SOURCE_X, y=_SOURCE_Y, z=_SOURCE_Z),))

    transform_survey(survey, utm_to_wgs84)

    assert survey.points[0].x == _SOURCE_X
    assert survey.points[0].y == _SOURCE_Y


def test_transform_survey_empty_set() -> None:
    empty = SurveyPointSet(points=())
    transformer = CoordinateTransformer(CRS.from_epsg(32617), CRS.from_epsg(4326))

    result = transform_survey(empty, transformer)

    assert len(result) == 0


def test_transform_survey_matches_direct_transformer_call(
    utm_to_wgs84: CoordinateTransformer,
) -> None:
    survey = SurveyPointSet(points=(SurveyPoint(id="P1", x=_SOURCE_X, y=_SOURCE_Y, z=_SOURCE_Z),))

    result = transform_survey(survey, utm_to_wgs84)
    direct_x, direct_y, direct_z = utm_to_wgs84.transform_point(_SOURCE_X, _SOURCE_Y, _SOURCE_Z)

    assert result.points[0].x == direct_x
    assert result.points[0].y == direct_y
    assert result.points[0].z == direct_z


# ----------------------------------------------------------------------
# transform_feature_collection
# ----------------------------------------------------------------------


def _make_point_feature(x: float, y: float, z: float, feature_id: int = 1) -> Feature:
    geometry = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[x, y, z]]))
    return Feature(
        feature_id=feature_id,
        category=FeatureCategory.CONTROL,
        feature_type=FeatureType.CONTROL_POINT,
        geometry=geometry,
    )


def test_transform_feature_collection_transforms_vertices(
    utm_to_wgs84: CoordinateTransformer,
) -> None:
    collection = FeatureCollection(features=[_make_point_feature(_SOURCE_X, _SOURCE_Y, _SOURCE_Z)])

    result = transform_feature_collection(collection, utm_to_wgs84)

    vertex = result.features[0].geometry.vertices[0]
    assert vertex[0] == _EXPECTED_LON
    assert vertex[1] == _EXPECTED_LAT
    assert vertex[2] == pytest.approx(_SOURCE_Z)


def test_transform_feature_collection_matches_transform_survey_for_same_point(
    utm_to_wgs84: CoordinateTransformer,
) -> None:
    """
    The same physical point, expressed as a SurveyPointSet vs. as a
    FeatureCollection, must transform to the same coordinates.
    """
    survey_result = transform_survey(
        SurveyPointSet(points=(SurveyPoint(id="P1", x=_SOURCE_X, y=_SOURCE_Y, z=_SOURCE_Z),)),
        utm_to_wgs84,
    )
    feature_result = transform_feature_collection(
        FeatureCollection(features=[_make_point_feature(_SOURCE_X, _SOURCE_Y, _SOURCE_Z)]),
        utm_to_wgs84,
    )

    survey_point = survey_result.points[0]
    feature_vertex = feature_result.features[0].geometry.vertices[0]

    assert survey_point.x == pytest.approx(feature_vertex[0])
    assert survey_point.y == pytest.approx(feature_vertex[1])
    assert survey_point.z == pytest.approx(feature_vertex[2])


def test_transform_feature_collection_does_not_mutate_original(
    utm_to_wgs84: CoordinateTransformer,
) -> None:
    collection = FeatureCollection(features=[_make_point_feature(_SOURCE_X, _SOURCE_Y, _SOURCE_Z)])
    original_vertex = collection.features[0].geometry.vertices.copy()

    transform_feature_collection(collection, utm_to_wgs84)

    np.testing.assert_array_equal(collection.features[0].geometry.vertices, original_vertex)


def test_transform_feature_collection_preserves_feature_count_and_order(
    utm_to_wgs84: CoordinateTransformer,
) -> None:
    collection = FeatureCollection(
        features=[
            _make_point_feature(_SOURCE_X, _SOURCE_Y, _SOURCE_Z, feature_id=1),
            _make_point_feature(_SOURCE_X + 100, _SOURCE_Y + 100, _SOURCE_Z, feature_id=2),
        ]
    )

    result = transform_feature_collection(collection, utm_to_wgs84)

    assert len(result.features) == 2
    assert [f.feature_id for f in result.features] == [1, 2]


def test_transform_feature_collection_preserves_mesh_faces_unchanged(
    utm_to_wgs84: CoordinateTransformer,
) -> None:
    """
    faces (MESH triangle indices into vertices) must pass through
    unchanged -- they are indices, not coordinates.
    """
    geometry = FeatureGeometry(
        geometry_type=GeometryType.MESH,
        vertices=np.array(
            [
                [_SOURCE_X, _SOURCE_Y, _SOURCE_Z],
                [_SOURCE_X + 1, _SOURCE_Y, _SOURCE_Z],
                [_SOURCE_X, _SOURCE_Y + 1, _SOURCE_Z],
            ]
        ),
        faces=np.array([[0, 1, 2]]),
    )
    feature = Feature(
        feature_id=1,
        category=FeatureCategory.BUILDING,
        feature_type=FeatureType.ROOF,
        geometry=geometry,
    )
    collection = FeatureCollection(features=[feature])

    result = transform_feature_collection(collection, utm_to_wgs84)

    np.testing.assert_array_equal(result.features[0].geometry.faces, np.array([[0, 1, 2]]))
    assert result.features[0].geometry.vertices.shape == (3, 3)
    # Vertices genuinely transformed, not left in source CRS.
    assert result.features[0].geometry.vertices[0, 0] == _EXPECTED_LON


# ----------------------------------------------------------------------
# transform_point_cloud
# ----------------------------------------------------------------------


def _make_point_cloud(xs: list[float], ys: list[float], zs: list[float], intensities: list[int]) -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(
        size=len(xs),
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
            PointAttribute.INTENSITY,
        ],
    )
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    chunk[PointAttribute.INTENSITY][:] = intensities
    cloud.add_chunk(chunk)
    cloud.crs = "EPSG:32617"
    return cloud


def test_transform_point_cloud_transforms_xyz(
    utm_to_wgs84: CoordinateTransformer,
) -> None:
    cloud = _make_point_cloud([_SOURCE_X], [_SOURCE_Y], [_SOURCE_Z], [50])

    result = transform_point_cloud(cloud, utm_to_wgs84)
    result_chunk = next(iter(result))

    assert result_chunk[PointAttribute.X][0] == _EXPECTED_LON
    assert result_chunk[PointAttribute.Y][0] == _EXPECTED_LAT
    assert result_chunk[PointAttribute.Z][0] == pytest.approx(_SOURCE_Z)


def test_transform_point_cloud_matches_transform_survey_for_same_point(
    utm_to_wgs84: CoordinateTransformer,
) -> None:
    survey_result = transform_survey(
        SurveyPointSet(points=(SurveyPoint(id="P1", x=_SOURCE_X, y=_SOURCE_Y, z=_SOURCE_Z),)),
        utm_to_wgs84,
    )
    cloud_result = transform_point_cloud(_make_point_cloud([_SOURCE_X], [_SOURCE_Y], [_SOURCE_Z], [50]), utm_to_wgs84)
    cloud_chunk = next(iter(cloud_result))

    assert survey_result.points[0].x == pytest.approx(float(cloud_chunk[PointAttribute.X][0]))
    assert survey_result.points[0].y == pytest.approx(float(cloud_chunk[PointAttribute.Y][0]))


def test_transform_point_cloud_preserves_non_coordinate_attributes(
    utm_to_wgs84: CoordinateTransformer,
) -> None:
    cloud = _make_point_cloud(
        [_SOURCE_X, _SOURCE_X + 10],
        [_SOURCE_Y, _SOURCE_Y + 10],
        [_SOURCE_Z, _SOURCE_Z + 1],
        [50, 77],
    )

    result = transform_point_cloud(cloud, utm_to_wgs84)
    result_chunk = next(iter(result))

    np.testing.assert_array_equal(result_chunk[PointAttribute.INTENSITY], [50, 77])


def test_transform_point_cloud_does_not_mutate_original(
    utm_to_wgs84: CoordinateTransformer,
) -> None:
    cloud = _make_point_cloud([_SOURCE_X], [_SOURCE_Y], [_SOURCE_Z], [50])

    transform_point_cloud(cloud, utm_to_wgs84)

    original_chunk = next(iter(cloud))
    assert original_chunk[PointAttribute.X][0] == _SOURCE_X
    assert original_chunk[PointAttribute.Y][0] == _SOURCE_Y


def test_transform_point_cloud_updates_crs_to_target() -> None:
    transformer = CoordinateTransformer(CRS.from_epsg(32617), CRS.from_epsg(4326))
    cloud = _make_point_cloud([_SOURCE_X], [_SOURCE_Y], [_SOURCE_Z], [50])

    result = transform_point_cloud(cloud, transformer)

    assert result.crs == "EPSG:4326"


def test_transform_point_cloud_recomputes_bounds() -> None:
    transformer = CoordinateTransformer(CRS.from_epsg(32617), CRS.from_epsg(4326))
    cloud = _make_point_cloud(
        [_SOURCE_X, _SOURCE_X + 1000],
        [_SOURCE_Y, _SOURCE_Y + 1000],
        [_SOURCE_Z, _SOURCE_Z + 10],
        [50, 60],
    )

    result = transform_point_cloud(cloud, transformer)

    # Bounds must reflect the NEW (WGS84) coordinate range, not the
    # stale UTM-range bounds carried over from the source cloud.
    assert result.bounds is not None
    min_x, max_x = result.bounds.min_x, result.bounds.max_x
    assert -82.0 < min_x < -80.0  # plausible longitude range, not UTM meters
    assert max_x > min_x


def test_transform_point_cloud_handles_multiple_chunks() -> None:
    transformer = CoordinateTransformer(CRS.from_epsg(32617), CRS.from_epsg(4326))
    cloud = PointCloud()
    for i in range(2):
        chunk = Chunk(size=1, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
        chunk[PointAttribute.X][:] = [_SOURCE_X + i * 1000]
        chunk[PointAttribute.Y][:] = [_SOURCE_Y]
        chunk[PointAttribute.Z][:] = [_SOURCE_Z]
        cloud.add_chunk(chunk)

    result = transform_point_cloud(cloud, transformer)

    chunks = list(result)
    assert len(chunks) == 2
    assert chunks[0][PointAttribute.X][0] != chunks[1][PointAttribute.X][0]
