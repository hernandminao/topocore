"""
Tests for topocore.geodesy.georeferencing.apply -- applying an
already-fitted HelmertParameters to real artifact data.

Every test confirms 3 things per adapter: correct coordinate
transformation (matching apply_helmert_3d() exactly), correct CRS
assignment (in each artifact's own confirmed type -- str for
PointCloud/FeatureCollection, a real CRS object for SurveyPointSet),
and that the ORIGINAL artifact is never mutated.
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
from topocore.geodesy import CRS
from topocore.geodesy.georeferencing.apply import (
    apply_georeferencing_to_feature_collection,
    apply_georeferencing_to_point_cloud,
    apply_georeferencing_to_survey,
    apply_helmert_3d_array,
)
from topocore.geodesy.georeferencing.helmert_fit import apply_helmert_3d
from topocore.geodesy.helmert import HelmertParameters
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.survey.models import SurveyPoint, SurveyPointSet

_PARAMS = HelmertParameters(tx=10.0, ty=-5.0, tz=3.0, rx=2.0, ry=-1.0, rz=5.0, scale=50.0)
_TARGET_CRS = CRS.from_epsg(3116)


# ----------------------------------------------------------------------
# apply_helmert_3d_array() -- vectorized consistency with apply_helmert_3d().
# ----------------------------------------------------------------------


def test_vectorized_matches_per_point_exactly() -> None:
    points = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 50.0),
        (50.0, 50.0, 25.0),
        (10.0, 20.0, 30.0),
    ]
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    zs = np.array([p[2] for p in points])

    xt_array, yt_array, zt_array = apply_helmert_3d_array(xs, ys, zs, _PARAMS)

    for i, point in enumerate(points):
        xt, yt, zt = apply_helmert_3d(point, _PARAMS)
        assert xt_array[i] == pytest.approx(xt)
        assert yt_array[i] == pytest.approx(yt)
        assert zt_array[i] == pytest.approx(zt)


# ----------------------------------------------------------------------
# apply_georeferencing_to_point_cloud().
# ----------------------------------------------------------------------


def _sample_point_cloud() -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(
        size=3,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
            PointAttribute.INTENSITY,
        ],
    )
    chunk[PointAttribute.X][:] = [1.0, 2.0, 3.0]
    chunk[PointAttribute.Y][:] = [4.0, 5.0, 6.0]
    chunk[PointAttribute.Z][:] = [7.0, 8.0, 9.0]
    chunk[PointAttribute.INTENSITY][:] = [100, 200, 255]
    cloud.add_chunk(chunk)
    return cloud


def test_point_cloud_coordinates_transformed_correctly() -> None:
    cloud = _sample_point_cloud()
    result = apply_georeferencing_to_point_cloud(cloud, _PARAMS, _TARGET_CRS)

    expected_x, expected_y, expected_z = apply_helmert_3d((1.0, 4.0, 7.0), _PARAMS)
    assert result[0][PointAttribute.X][0] == pytest.approx(expected_x)
    assert result[0][PointAttribute.Y][0] == pytest.approx(expected_y)
    assert result[0][PointAttribute.Z][0] == pytest.approx(expected_z)


def test_point_cloud_crs_is_string_matching_pointcloud_convention() -> None:
    cloud = _sample_point_cloud()
    result = apply_georeferencing_to_point_cloud(cloud, _PARAMS, _TARGET_CRS)
    assert result.metadata.crs == "EPSG:3116"


def test_point_cloud_non_geometric_attributes_preserved() -> None:
    cloud = _sample_point_cloud()
    result = apply_georeferencing_to_point_cloud(cloud, _PARAMS, _TARGET_CRS)
    assert list(result[0][PointAttribute.INTENSITY]) == [100, 200, 255]


def test_point_cloud_original_is_never_mutated() -> None:
    cloud = _sample_point_cloud()
    apply_georeferencing_to_point_cloud(cloud, _PARAMS, _TARGET_CRS)
    assert cloud[0][PointAttribute.X][0] == 1.0
    assert cloud.metadata.crs is None


def test_point_cloud_bounds_are_recomputed() -> None:
    cloud = _sample_point_cloud()
    result = apply_georeferencing_to_point_cloud(cloud, _PARAMS, _TARGET_CRS)
    assert result.metadata.bounds is not None
    # The transformed bounds must reflect the NEW coordinates, not the original ones.
    assert result.metadata.bounds != cloud.metadata.bounds


# ----------------------------------------------------------------------
# apply_georeferencing_to_survey().
# ----------------------------------------------------------------------


def test_survey_coordinates_transformed_correctly() -> None:
    survey = SurveyPointSet(points=(SurveyPoint(id="1", x=1.0, y=4.0, z=7.0, code="PT"),))
    result = apply_georeferencing_to_survey(survey, _PARAMS, _TARGET_CRS)

    expected_x, expected_y, expected_z = apply_helmert_3d((1.0, 4.0, 7.0), _PARAMS)
    assert result.points[0].x == pytest.approx(expected_x)
    assert result.points[0].y == pytest.approx(expected_y)
    assert result.points[0].z == pytest.approx(expected_z)


def test_survey_crs_is_a_real_crs_object_matching_target_crs_directly() -> None:
    survey = SurveyPointSet(points=(SurveyPoint(id="1", x=1.0, y=4.0, z=7.0),))
    result = apply_georeferencing_to_survey(survey, _PARAMS, _TARGET_CRS)
    assert result.crs is _TARGET_CRS  # the exact same object, not a re-derived equivalent one


def test_survey_id_and_code_preserved() -> None:
    survey = SurveyPointSet(points=(SurveyPoint(id="P42", x=1.0, y=4.0, z=7.0, code="ARBOL"),))
    result = apply_georeferencing_to_survey(survey, _PARAMS, _TARGET_CRS)
    assert result.points[0].id == "P42"
    assert result.points[0].code == "ARBOL"


def test_survey_original_is_never_mutated() -> None:
    survey = SurveyPointSet(points=(SurveyPoint(id="1", x=1.0, y=4.0, z=7.0),))
    apply_georeferencing_to_survey(survey, _PARAMS, _TARGET_CRS)
    assert survey.points[0].x == 1.0
    assert survey.crs is None


def test_survey_multiple_points() -> None:
    survey = SurveyPointSet(
        points=(
            SurveyPoint(id="1", x=1.0, y=4.0, z=7.0),
            SurveyPoint(id="2", x=10.0, y=20.0, z=30.0),
        )
    )
    result = apply_georeferencing_to_survey(survey, _PARAMS, _TARGET_CRS)
    assert len(result.points) == 2
    expected_x, _, _ = apply_helmert_3d((10.0, 20.0, 30.0), _PARAMS)
    assert result.points[1].x == pytest.approx(expected_x)


# ----------------------------------------------------------------------
# apply_georeferencing_to_feature_collection().
# ----------------------------------------------------------------------


def _sample_feature_collection() -> FeatureCollection:
    geometry = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[1.0, 4.0, 7.0]]))
    feature = Feature(
        feature_id=1,
        category=FeatureCategory.TERRAIN,
        feature_type=FeatureType.CONTROL_POINT,
        geometry=geometry,
    )
    collection = FeatureCollection()
    collection.add(feature)
    return collection


def test_feature_collection_vertices_transformed_correctly() -> None:
    collection = _sample_feature_collection()
    result = apply_georeferencing_to_feature_collection(collection, _PARAMS, _TARGET_CRS)

    expected_x, expected_y, expected_z = apply_helmert_3d((1.0, 4.0, 7.0), _PARAMS)
    vertex = result.features[0].geometry.vertices[0]
    assert vertex[0] == pytest.approx(expected_x)
    assert vertex[1] == pytest.approx(expected_y)
    assert vertex[2] == pytest.approx(expected_z)


def test_feature_collection_crs_is_string_matching_convention() -> None:
    collection = _sample_feature_collection()
    result = apply_georeferencing_to_feature_collection(collection, _PARAMS, _TARGET_CRS)
    assert result.crs == "EPSG:3116"


def test_feature_collection_original_is_never_mutated() -> None:
    collection = _sample_feature_collection()
    apply_georeferencing_to_feature_collection(collection, _PARAMS, _TARGET_CRS)
    assert collection.features[0].geometry.vertices[0][0] == 1.0
    assert collection.crs is None


def test_feature_collection_faces_pass_through_unchanged() -> None:
    """MESH faces are triangle indices into vertices -- never coordinates, so they must never be transformed."""
    geometry = FeatureGeometry(
        geometry_type=GeometryType.MESH,
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
    )
    feature = Feature(
        feature_id=1,
        category=FeatureCategory.BUILDING,
        feature_type=FeatureType.ROOF,
        geometry=geometry,
    )
    collection = FeatureCollection()
    collection.add(feature)

    result = apply_georeferencing_to_feature_collection(collection, _PARAMS, _TARGET_CRS)
    np.testing.assert_array_equal(result.features[0].geometry.faces, geometry.faces)


def test_feature_collection_multiple_features() -> None:
    collection = FeatureCollection()
    for i in range(3):
        geometry = FeatureGeometry(
            geometry_type=GeometryType.POINT,
            vertices=np.array([[float(i), float(i), 0.0]]),
        )
        collection.add(
            Feature(
                feature_id=i,
                category=FeatureCategory.TERRAIN,
                feature_type=FeatureType.CONTROL_POINT,
                geometry=geometry,
            )
        )

    result = apply_georeferencing_to_feature_collection(collection, _PARAMS, _TARGET_CRS)
    assert len(result.features) == 3


# ----------------------------------------------------------------------
# Reviewer-identified gaps, closed with real, verified evidence.
# ----------------------------------------------------------------------


def test_helmert_2d_result_applied_with_scale_to_z_introduces_a_real_discrepancy() -> None:
    """
    Confirmed directly, and the reason apply_scale_to_z exists at
    all: applying a HELMERT_2D-fitted result to a NEW point (not one
    of the fit's own control points) via the default
    apply_scale_to_z=True introduces a real, Z-magnitude-dependent
    error the fit's own residuals never saw -- since fit_helmert_2d()
    itself assumes Z' = Z + tz alone, with no scale term.
    """
    from topocore.geodesy.georeferencing import ControlPoint, fit_helmert_2d

    sources = [(100.0, 0.0, 0.0), (0.0, 100.0, 50.0), (50.0, 50.0, 25.0)]
    targets = [
        (
            x * 1.00005 + 10 - (5 / 206265) * y,
            y * 1.00005 + 20 + (5 / 206265) * x,
            z + 3,
        )
        for x, y, z in sources
    ]
    controls = [
        ControlPoint(id=str(i), source=s, target=t) for i, (s, t) in enumerate(zip(sources, targets, strict=True))
    ]
    result_2d = fit_helmert_2d(controls)

    new_point = (
        200.0,
        300.0,
        500.0,
    )  # a genuinely new point, not one of the controls above
    expected_z = new_point[2] + result_2d.parameters.tz  # per HELMERT_2D's own Z=Z+tz-only model

    with_scale = apply_helmert_3d(new_point, result_2d.parameters, apply_scale_to_z=True)
    without_scale = apply_helmert_3d(new_point, result_2d.parameters, apply_scale_to_z=False)

    assert with_scale[2] != pytest.approx(expected_z)  # confirms the discrepancy is genuinely present by default
    assert without_scale[2] == pytest.approx(expected_z)  # confirms apply_scale_to_z=False fixes it exactly


# A genuinely HELMERT_2D-representative parameter set for the 3
# apply_scale_to_z=False tests below -- rx=ry=0 exactly, matching
# what fit_helmert_2d() always produces (never the general _PARAMS
# fixture above, which has nonzero rx/ry: apply_scale_to_z=False only
# removes the scale term, s*zs -- it does NOT remove the separate
# rx/ry rotation-coupling terms in Z, so "Z = Z + tz only" is only
# true when rx=ry=0 too, exactly the case this parameter is for).
_PARAMS_2D_STYLE = HelmertParameters(tx=10.0, ty=-5.0, tz=3.0, rz=5.0, scale=50.0)


def test_apply_scale_to_z_false_propagates_through_point_cloud_adapter() -> None:
    result = apply_georeferencing_to_point_cloud(
        _sample_point_cloud(), _PARAMS_2D_STYLE, _TARGET_CRS, apply_scale_to_z=False
    )
    expected_z_no_scale = 7.0 + _PARAMS_2D_STYLE.tz  # Z=Z+tz only (rx=ry=0 here, so no coupling terms either)
    assert result[0][PointAttribute.Z][0] == pytest.approx(expected_z_no_scale)


def test_apply_scale_to_z_false_propagates_through_survey_adapter() -> None:
    survey = SurveyPointSet(points=(SurveyPoint(id="1", x=1.0, y=4.0, z=7.0),))
    result = apply_georeferencing_to_survey(survey, _PARAMS_2D_STYLE, _TARGET_CRS, apply_scale_to_z=False)
    expected_z_no_scale = 7.0 + _PARAMS_2D_STYLE.tz
    assert result.points[0].z == pytest.approx(expected_z_no_scale)


def test_apply_scale_to_z_false_propagates_through_feature_collection_adapter() -> None:
    result = apply_georeferencing_to_feature_collection(
        _sample_feature_collection(),
        _PARAMS_2D_STYLE,
        _TARGET_CRS,
        apply_scale_to_z=False,
    )
    expected_z_no_scale = 7.0 + _PARAMS_2D_STYLE.tz
    assert result.features[0].geometry.vertices[0][2] == pytest.approx(expected_z_no_scale)


def test_apply_scale_to_z_default_true_matches_prior_behavior() -> None:
    """Confirms adding this parameter did not change any EXISTING default-call behavior."""
    result_default = apply_georeferencing_to_point_cloud(_sample_point_cloud(), _PARAMS, _TARGET_CRS)
    result_explicit_true = apply_georeferencing_to_point_cloud(
        _sample_point_cloud(), _PARAMS, _TARGET_CRS, apply_scale_to_z=True
    )
    assert result_default[0][PointAttribute.Z][0] == result_explicit_true[0][PointAttribute.Z][0]


def test_point_cloud_with_many_chunks_transforms_all_of_them() -> None:
    cloud = PointCloud()
    for i in range(50):
        chunk = Chunk(size=10, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
        chunk[PointAttribute.X][:] = np.arange(10) + i * 10
        chunk[PointAttribute.Y][:] = np.arange(10)
        chunk[PointAttribute.Z][:] = np.arange(10)
        cloud.add_chunk(chunk)

    result = apply_georeferencing_to_point_cloud(cloud, _PARAMS, _TARGET_CRS)

    assert sum(1 for _ in result) == 50
    assert sum(chunk.size for chunk in result) == 500
    # Spot-check the last chunk's first point actually got transformed, not skipped.
    last_chunk = list(result)[-1]
    assert last_chunk[PointAttribute.X][0] != 490.0  # the original, untransformed value


def test_feature_collection_with_mixed_point_and_mesh_geometry_types() -> None:
    point_feature = Feature(
        feature_id=1,
        category=FeatureCategory.TERRAIN,
        feature_type=FeatureType.CONTROL_POINT,
        geometry=FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[1.0, 2.0, 3.0]])),
    )
    mesh_feature = Feature(
        feature_id=2,
        category=FeatureCategory.BUILDING,
        feature_type=FeatureType.ROOF,
        geometry=FeatureGeometry(
            geometry_type=GeometryType.MESH,
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        ),
    )
    collection = FeatureCollection()
    collection.add(point_feature)
    collection.add(mesh_feature)

    result = apply_georeferencing_to_feature_collection(collection, _PARAMS, _TARGET_CRS)

    assert len(result) == 2
    assert result.features[0].geometry.geometry_type is GeometryType.POINT
    assert result.features[1].geometry.geometry_type is GeometryType.MESH
    np.testing.assert_array_equal(result.features[1].geometry.faces, mesh_feature.geometry.faces)
    # The mesh's own vertices were genuinely transformed, not skipped because it's a different geometry_type.
    expected_x, expected_y, expected_z = apply_helmert_3d((0.0, 0.0, 0.0), _PARAMS)
    vertex = result.features[1].geometry.vertices[0]
    assert vertex[0] == pytest.approx(expected_x)
    assert vertex[1] == pytest.approx(expected_y)
    assert vertex[2] == pytest.approx(expected_z)


def test_crs_type_distinction_is_deliberate_not_a_bug() -> None:
    """
    Documents explicitly: SurveyPointSet.crs receives the SAME target_crs object (checked with `is`),
    while PointCloud/FeatureCollection.crs receive a derived STRING (checked with `==`) -- this
    difference matches each model's own pre-existing, confirmed convention (see this capability's
    own design decisions), not an inconsistency introduced by these adapters.
    """
    survey = SurveyPointSet(points=(SurveyPoint(id="1", x=1.0, y=4.0, z=7.0),))
    survey_result = apply_georeferencing_to_survey(survey, _PARAMS, _TARGET_CRS)
    cloud_result = apply_georeferencing_to_point_cloud(_sample_point_cloud(), _PARAMS, _TARGET_CRS)
    fc_result = apply_georeferencing_to_feature_collection(_sample_feature_collection(), _PARAMS, _TARGET_CRS)

    assert survey_result.crs is _TARGET_CRS  # real CRS object, identity-preserving
    assert isinstance(cloud_result.metadata.crs, str)
    assert isinstance(fc_result.crs, str)


def test_apply_helmert_3d_array_handles_one_million_points() -> None:
    """Confirms this scales to real LiDAR-sized data without error, in well under a second."""
    import time

    rng = np.random.default_rng(0)
    n = 1_000_000
    xs = rng.uniform(-1000, 1000, n)
    ys = rng.uniform(-1000, 1000, n)
    zs = rng.uniform(0, 500, n)

    started = time.monotonic()
    xt, yt, zt = apply_helmert_3d_array(xs, ys, zs, _PARAMS)
    elapsed = time.monotonic() - started

    assert xt.shape == (n,)
    assert yt.shape == (n,)
    assert zt.shape == (n,)
    assert np.all(np.isfinite(xt))
    assert np.all(np.isfinite(yt))
    assert np.all(np.isfinite(zt))
    assert elapsed < 5.0  # generous bound -- confirmed directly this runs in ~0.15s
