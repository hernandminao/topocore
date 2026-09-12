"""
Tests for topocore.geodesy.vertical.transformer.VerticalTransformer
and topocore.geodesy.vertical.transform's per-type functions.
"""

from __future__ import annotations

from pathlib import Path

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
from topocore.geodesy import VerticalDatum
from topocore.geodesy.vertical.exceptions import MissingGeoidGridError
from topocore.geodesy.vertical.geoid_grid import GeoidGrid
from topocore.geodesy.vertical.transform import (
    transform_feature_collection_vertical,
    transform_point_cloud_vertical,
    transform_survey_vertical,
)
from topocore.geodesy.vertical.transformer import VerticalTransformer
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.survey.models import SurveyPoint, SurveyPointSet


@pytest.fixture
def transformer(synthetic_geoid_path: Path) -> VerticalTransformer:
    return VerticalTransformer(
        source_datum=VerticalDatum(name="WGS84 ellipsoidal"),
        target_datum=VerticalDatum(name="Synthetic", geoid_model="SYNTHETIC"),
        geoid=GeoidGrid.from_geotiff(synthetic_geoid_path),
    )


# ----------------------------------------------------------------------
# VerticalTransformer itself.
# ----------------------------------------------------------------------


def test_ellipsoidal_to_orthometric_subtracts_undulation(
    transformer: VerticalTransformer,
) -> None:
    # N=50 exactly at (1,1); h=100 -> H = 100 - 50 = 50
    assert transformer.ellipsoidal_to_orthometric(1.0, 1.0, 100.0) == 50.0


def test_orthometric_to_ellipsoidal_adds_undulation(
    transformer: VerticalTransformer,
) -> None:
    assert transformer.orthometric_to_ellipsoidal(1.0, 1.0, 50.0) == 100.0


def test_round_trip_is_exact(transformer: VerticalTransformer) -> None:
    original = 123.456
    orthometric = transformer.ellipsoidal_to_orthometric(1.0, 1.0, original)
    back = transformer.orthometric_to_ellipsoidal(1.0, 1.0, orthometric)
    assert back == pytest.approx(original)


def test_never_returns_uncorrected_height_outside_grid(
    transformer: VerticalTransformer,
) -> None:
    """The core safety guarantee: a point outside the grid raises, never silently returns h unchanged."""
    with pytest.raises(MissingGeoidGridError):
        transformer.ellipsoidal_to_orthometric(100.0, 100.0, 2600.0)


# ----------------------------------------------------------------------
# transform_point_cloud_vertical
# ----------------------------------------------------------------------


def test_transform_point_cloud_vertical_shifts_only_z(
    transformer: VerticalTransformer,
) -> None:
    cloud = PointCloud()
    chunk = Chunk(size=2, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [1.0, 0.5]
    chunk[PointAttribute.Y][:] = [1.0, 1.5]
    chunk[PointAttribute.Z][:] = [100.0, 100.0]
    cloud.add_chunk(chunk)

    transformed = transform_point_cloud_vertical(cloud, transformer)
    new_chunk = next(iter(transformed))

    assert list(new_chunk[PointAttribute.Z]) == [50.0, 70.0]
    assert list(new_chunk[PointAttribute.X]) == [1.0, 0.5]
    assert list(new_chunk[PointAttribute.Y]) == [1.0, 1.5]


def test_transform_point_cloud_vertical_does_not_mutate_input(
    transformer: VerticalTransformer,
) -> None:
    cloud = PointCloud()
    chunk = Chunk(size=1, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [1.0]
    chunk[PointAttribute.Y][:] = [1.0]
    chunk[PointAttribute.Z][:] = [100.0]
    cloud.add_chunk(chunk)

    transform_point_cloud_vertical(cloud, transformer)

    assert next(iter(cloud))[PointAttribute.Z][0] == 100.0


def test_transform_point_cloud_vertical_fails_whole_operation_not_partially(
    transformer: VerticalTransformer,
) -> None:
    """One point inside the grid, one outside -- the WHOLE call must fail, not partially succeed."""
    cloud = PointCloud()
    chunk = Chunk(size=2, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [1.0, 100.0]
    chunk[PointAttribute.Y][:] = [1.0, 100.0]
    chunk[PointAttribute.Z][:] = [100.0, 100.0]
    cloud.add_chunk(chunk)

    with pytest.raises(MissingGeoidGridError):
        transform_point_cloud_vertical(cloud, transformer)


# ----------------------------------------------------------------------
# transform_survey_vertical
# ----------------------------------------------------------------------


def test_transform_survey_vertical(transformer: VerticalTransformer) -> None:
    survey = SurveyPointSet(points=(SurveyPoint(id="1", x=1.0, y=1.0, z=100.0, code="PT"),))
    transformed = transform_survey_vertical(survey, transformer)

    assert transformed.points[0].z == 50.0
    assert transformed.points[0].x == 1.0
    assert transformed.points[0].code == "PT"
    assert survey.points[0].z == 100.0  # original untouched


# ----------------------------------------------------------------------
# transform_feature_collection_vertical
# ----------------------------------------------------------------------


def test_transform_feature_collection_vertical(
    transformer: VerticalTransformer,
) -> None:
    geometry = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[1.0, 1.0, 100.0]]))
    feature = Feature(
        feature_id=1,
        category=FeatureCategory.TERRAIN,
        feature_type=FeatureType.CONTROL_POINT,
        geometry=geometry,
    )
    collection = FeatureCollection()
    collection.add(feature)

    transformed = transform_feature_collection_vertical(collection, transformer)
    new_vertex = next(iter(transformed)).geometry.vertices[0]

    assert new_vertex[0] == 1.0
    assert new_vertex[1] == 1.0
    assert new_vertex[2] == 50.0
    assert collection.features[0].geometry.vertices[0][2] == 100.0  # original untouched
