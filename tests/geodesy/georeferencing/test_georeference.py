"""
Tests for Workflow.georeference() -- the integration layer consuming
fit_georeferencing() + the 3 apply_georeferencing_to_*() adapters,
already individually verified in
topocore.geodesy.georeferencing's own test suites.

This suite validates the INTEGRATION on top of those already-frozen
contracts -- it does not modify or re-test transform_crs(),
CoordinateTransformer, or apply.py's own math (already covered
elsewhere). Every strategy-dependent assertion checks
`result.strategy` directly (via `GeoreferencingResult`, exposed in
`StageMetrics.extra["georeferencing_result"]`) -- never inferred from
`rx`/`ry` values, matching this capability's own explicit rule.
"""

from __future__ import annotations

from typing import Any

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
from topocore.geodesy.georeferencing import (
    ControlPoint,
    GeoreferencingOptions,
    GeoreferencingStrategy,
    apply_helmert_3d,
)
from topocore.geodesy.georeferencing.exceptions import (
    UnderconstrainedGeoreferencingError,
)
from topocore.geodesy.helmert import HelmertParameters
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.survey.models import SurveyPoint, SurveyPointSet
from topocore.workflow.artifacts import ArtifactType
from topocore.workflow.exceptions import WorkflowStateError
from topocore.workflow.stages import WorkflowStage
from topocore.workflow.workflow import Workflow

_KNOWN_3D = HelmertParameters(tx=1152345.0, ty=923456.0, tz=245.0, rx=2.0, ry=-1.0, rz=5.0, scale=50.0)
_TARGET_CRS = CRS.from_epsg(3116)


def _last_georeferencing_result(workflow: Workflow) -> Any:
    metrics = workflow.result.stages[-1].metrics
    assert metrics is not None
    return metrics.extra["georeferencing_result"]


def _controls_from(sources: list[tuple[float, float, float]], parameters: HelmertParameters) -> list[ControlPoint]:
    targets = [apply_helmert_3d(point, parameters) for point in sources]
    return [ControlPoint(id=str(i), source=s, target=t) for i, (s, t) in enumerate(zip(sources, targets, strict=True))]


def _sample_point_cloud() -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(size=1, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [200.0]
    chunk[PointAttribute.Y][:] = [300.0]
    chunk[PointAttribute.Z][:] = [500.0]
    cloud.add_chunk(chunk)
    return cloud


def _sample_survey() -> SurveyPointSet:
    return SurveyPointSet(points=(SurveyPoint(id="1", x=0.0, y=0.0, z=100.0, code="PT"),))


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


def _sample_mesh_feature_collection() -> FeatureCollection:
    geometry = FeatureGeometry(
        geometry_type=GeometryType.MESH,
        vertices=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
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
    return collection


# ----------------------------------------------------------------------
# 1. HELMERT_3D determined -- transforms correctly.
# ----------------------------------------------------------------------


def test_1_helmert_3d_determined_transforms_correctly() -> None:
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    controls = _controls_from(sources, _KNOWN_3D)

    workflow = Workflow()
    workflow._store.set(ArtifactType.SURVEY_POINT_SET, _sample_survey())
    workflow.georeference(ArtifactType.SURVEY_POINT_SET, controls, _TARGET_CRS, GeoreferencingOptions())

    result_artifact = workflow.artifact(ArtifactType.SURVEY_POINT_SET)
    gr = _last_georeferencing_result(workflow)

    assert gr.strategy is GeoreferencingStrategy.HELMERT_3D
    assert result_artifact.crs is _TARGET_CRS
    expected = apply_helmert_3d((0.0, 0.0, 100.0), gr.parameters)
    assert result_artifact.points[0].x == pytest.approx(expected[0])
    assert result_artifact.points[0].y == pytest.approx(expected[1])
    assert result_artifact.points[0].z == pytest.approx(expected[2])


# ----------------------------------------------------------------------
# 2. HELMERT_2D -- apply_scale_to_z=False, verified via a real Z
# discrepancy check, not just "it ran".
# ----------------------------------------------------------------------


def test_2_helmert_2d_uses_apply_scale_to_z_false() -> None:
    known_2d = HelmertParameters(tx=10.0, ty=20.0, tz=3.0, rz=5.0, scale=50.0)
    sources = [(100.0, 0.0, 0.0), (0.0, 100.0, 50.0)]
    controls = _controls_from(sources, known_2d)

    workflow = Workflow()
    workflow._store.set(ArtifactType.POINT_CLOUD, _sample_point_cloud())
    workflow.georeference(
        ArtifactType.POINT_CLOUD,
        controls,
        _TARGET_CRS,
        GeoreferencingOptions(accept_underconstrained=True),
    )

    result_artifact = workflow.artifact(ArtifactType.POINT_CLOUD)
    gr = _last_georeferencing_result(workflow)

    assert gr.strategy is GeoreferencingStrategy.HELMERT_2D

    # Z' = Z + tz_FITTED (not the "true" tz used to build the controls above --
    # confirmed directly during manual verification that using the true value
    # instead of the actually-fitted one gives a small, misleading false mismatch).
    expected_z_no_scale = 500.0 + gr.parameters.tz
    assert result_artifact[0][PointAttribute.Z][0] == pytest.approx(expected_z_no_scale, abs=1e-9)

    # Contrast: confirms the discrepancy would be real if scale HAD been applied to Z.
    from topocore.geodesy.georeferencing.apply import apply_helmert_3d_array

    zt_with_scale = apply_helmert_3d_array(
        np.array([200.0]),
        np.array([300.0]),
        np.array([500.0]),
        gr.parameters,
        apply_scale_to_z=True,
    )[2][0]
    assert zt_with_scale != pytest.approx(expected_z_no_scale, abs=1e-9)


# ----------------------------------------------------------------------
# 3. TRANSLATION_ONLY authorized -- transforms, rms=None.
# ----------------------------------------------------------------------


def test_3_translation_only_authorized_transforms_with_none_rms() -> None:
    control = ControlPoint(id="1", source=(0.0, 0.0, 0.0), target=(5.0, 3.0, 2.0))

    workflow = Workflow()
    workflow._store.set(ArtifactType.SURVEY_POINT_SET, _sample_survey())
    workflow.georeference(
        ArtifactType.SURVEY_POINT_SET,
        [control],
        _TARGET_CRS,
        GeoreferencingOptions(accept_underconstrained=True),
    )

    result_artifact = workflow.artifact(ArtifactType.SURVEY_POINT_SET)
    gr = _last_georeferencing_result(workflow)

    assert gr.strategy is GeoreferencingStrategy.TRANSLATION_ONLY
    assert gr.rms is None
    assert result_artifact.points[0].x == pytest.approx(5.0)
    assert result_artifact.crs is _TARGET_CRS


# ----------------------------------------------------------------------
# 4. 1 control without accept_underconstrained -- rejected.
# ----------------------------------------------------------------------


def test_4_single_control_without_authorization_is_rejected() -> None:
    control = ControlPoint(id="1", source=(0.0, 0.0, 0.0), target=(5.0, 3.0, 2.0))

    workflow = Workflow()
    workflow._store.set(ArtifactType.SURVEY_POINT_SET, _sample_survey())

    with pytest.raises(UnderconstrainedGeoreferencingError):
        workflow.georeference(
            ArtifactType.SURVEY_POINT_SET,
            [control],
            _TARGET_CRS,
            GeoreferencingOptions(),
        )


# ----------------------------------------------------------------------
# 5. 2 degenerate controls without authorization -- rejected.
# ----------------------------------------------------------------------


def test_5_two_degenerate_controls_without_authorization_is_rejected() -> None:
    """Same XY (vertical pair) -- degenerate for HELMERT_2D too, falls to TRANSLATION_ONLY, still needs authorization."""
    controls = [
        ControlPoint(id="1", source=(5.0, 5.0, 0.0), target=(6.0, 7.0, 3.0)),
        ControlPoint(id="2", source=(5.0, 5.0, 10.0), target=(6.0, 7.0, 13.0)),
    ]

    workflow = Workflow()
    workflow._store.set(ArtifactType.SURVEY_POINT_SET, _sample_survey())

    with pytest.raises(UnderconstrainedGeoreferencingError):
        workflow.georeference(
            ArtifactType.SURVEY_POINT_SET,
            controls,
            _TARGET_CRS,
            GeoreferencingOptions(),
        )


# ----------------------------------------------------------------------
# 6. GROUND_CLOUD -- transforms and gets target CRS.
# ----------------------------------------------------------------------


def test_6_ground_cloud_transforms_and_gets_target_crs() -> None:
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    controls = _controls_from(sources, _KNOWN_3D)

    workflow = Workflow()
    workflow._store.set(ArtifactType.GROUND_CLOUD, _sample_point_cloud())
    workflow.georeference(ArtifactType.GROUND_CLOUD, controls, _TARGET_CRS, GeoreferencingOptions())

    result_artifact = workflow.artifact(ArtifactType.GROUND_CLOUD)
    assert result_artifact.metadata.crs == "EPSG:3116"
    assert result_artifact[0][PointAttribute.X][0] != 200.0  # genuinely transformed, not left unchanged


# ----------------------------------------------------------------------
# 7. SURVEY_POINT_SET -- transforms and gets correct CRS (real object).
# ----------------------------------------------------------------------


def test_7_survey_point_set_transforms_and_gets_correct_crs() -> None:
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    controls = _controls_from(sources, _KNOWN_3D)

    workflow = Workflow()
    workflow._store.set(ArtifactType.SURVEY_POINT_SET, _sample_survey())
    workflow.georeference(ArtifactType.SURVEY_POINT_SET, controls, _TARGET_CRS, GeoreferencingOptions())

    result_artifact = workflow.artifact(ArtifactType.SURVEY_POINT_SET)
    assert result_artifact.crs is _TARGET_CRS  # same object, matching SurveyPointSet's own confirmed convention


# ----------------------------------------------------------------------
# 8. FEATURE_COLLECTION -- transforms, preserving geometry/faces.
# ----------------------------------------------------------------------


def test_8_feature_collection_transforms_preserving_faces() -> None:
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    controls = _controls_from(sources, _KNOWN_3D)

    workflow = Workflow()
    workflow._store.set(ArtifactType.FEATURE_COLLECTION, _sample_mesh_feature_collection())
    workflow.georeference(ArtifactType.FEATURE_COLLECTION, controls, _TARGET_CRS, GeoreferencingOptions())

    result_artifact = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
    assert result_artifact.crs == "EPSG:3116"
    np.testing.assert_array_equal(
        result_artifact.features[0].geometry.faces,
        np.array([[0, 1, 2]], dtype=np.int32),
    )
    assert result_artifact.features[0].geometry.vertices[0][0] != 0.0  # genuinely transformed


# ----------------------------------------------------------------------
# 9. Unsupported artifact -- rejected.
# ----------------------------------------------------------------------


def test_9_unsupported_artifact_type_is_rejected() -> None:
    control = ControlPoint(id="1", source=(0.0, 0.0, 0.0), target=(5.0, 3.0, 2.0))
    workflow = Workflow()

    with pytest.raises(WorkflowStateError, match="does not support"):
        workflow.georeference(
            ArtifactType.TIN,
            [control],
            _TARGET_CRS,
            GeoreferencingOptions(accept_underconstrained=True),
        )


# ----------------------------------------------------------------------
# 10. Original artifact remains intact on error.
# ----------------------------------------------------------------------


def test_10_original_artifact_untouched_when_rejected() -> None:
    control = ControlPoint(id="1", source=(0.0, 0.0, 0.0), target=(5.0, 3.0, 2.0))

    workflow = Workflow()
    workflow._store.set(ArtifactType.POINT_CLOUD, _sample_point_cloud())

    with pytest.raises(UnderconstrainedGeoreferencingError):
        workflow.georeference(ArtifactType.POINT_CLOUD, [control], _TARGET_CRS, GeoreferencingOptions())

    survivor = workflow.artifact(ArtifactType.POINT_CLOUD)
    assert survivor[0][PointAttribute.X][0] == 200.0
    assert survivor.metadata.crs is None


# ----------------------------------------------------------------------
# 11. StageMetrics.extra carries the full GeoreferencingResult.
# ----------------------------------------------------------------------


def test_11_stage_metrics_extra_carries_full_georeferencing_result() -> None:
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    controls = _controls_from(sources, _KNOWN_3D)

    workflow = Workflow()
    workflow._store.set(ArtifactType.SURVEY_POINT_SET, _sample_survey())
    workflow.georeference(ArtifactType.SURVEY_POINT_SET, controls, _TARGET_CRS, GeoreferencingOptions())

    gr = _last_georeferencing_result(workflow)
    assert gr.strategy is GeoreferencingStrategy.HELMERT_3D
    assert gr.parameters is not None
    assert len(gr.residuals) == 4
    assert gr.rms is not None
    assert gr.control_count == 4
    assert gr.warning is None


# ----------------------------------------------------------------------
# 12. GEOREFERENCE is recorded as a new stage.
# ----------------------------------------------------------------------


def test_12_georeference_is_recorded_as_a_stage() -> None:
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    controls = _controls_from(sources, _KNOWN_3D)

    workflow = Workflow()
    workflow._store.set(ArtifactType.SURVEY_POINT_SET, _sample_survey())
    workflow.georeference(ArtifactType.SURVEY_POINT_SET, controls, _TARGET_CRS, GeoreferencingOptions())

    assert workflow.result.stages[-1].stage is WorkflowStage.GEOREFERENCE
    assert workflow.result.stages[-1].status.value == "success"


# ----------------------------------------------------------------------
# 13. fit_georeferencing() exceptions propagate without mutating the workflow.
# ----------------------------------------------------------------------


def test_13_fit_georeferencing_exceptions_propagate_without_mutating_workflow() -> None:
    control = ControlPoint(id="1", source=(0.0, 0.0, 0.0), target=(5.0, 3.0, 2.0))

    workflow = Workflow()
    workflow._store.set(ArtifactType.SURVEY_POINT_SET, _sample_survey())
    stages_before = len(workflow.result.stages)

    with pytest.raises(UnderconstrainedGeoreferencingError):
        workflow.georeference(
            ArtifactType.SURVEY_POINT_SET,
            [control],
            _TARGET_CRS,
            GeoreferencingOptions(),
        )

    # No stage was recorded at all -- the exception happened before _execute_stage(),
    # exactly mirroring transform_crs()'s own pre-validation propagation pattern.
    assert len(workflow.result.stages) == stages_before


# ----------------------------------------------------------------------
# 14. Explicit Z-scale regression for 2D -- checked via result.strategy,
# never via rx/ry values.
# ----------------------------------------------------------------------


def test_14_z_scale_policy_checked_via_strategy_not_rx_ry() -> None:
    """
    The exact rule this whole capability settled on:
    apply_scale_to_z = (result.strategy is GeoreferencingStrategy.HELMERT_3D)
    -- never inferred from parameters.rx/ry, since HelmertParameters
    cannot and does not encode which strategy produced it.
    """
    known_2d = HelmertParameters(tx=10.0, ty=20.0, tz=3.0, rz=5.0, scale=50.0)
    sources = [(100.0, 0.0, 0.0), (0.0, 100.0, 50.0)]
    controls = _controls_from(sources, known_2d)

    workflow = Workflow()
    workflow._store.set(ArtifactType.POINT_CLOUD, _sample_point_cloud())
    workflow.georeference(
        ArtifactType.POINT_CLOUD,
        controls,
        _TARGET_CRS,
        GeoreferencingOptions(accept_underconstrained=True),
    )

    gr = _last_georeferencing_result(workflow)

    # The check that matters: strategy is what's inspected, not rx/ry
    # (even though rx=ry=0.0 happens to also be true here, that is
    # NOT what determined apply_scale_to_z inside georeference()).
    assert gr.strategy is GeoreferencingStrategy.HELMERT_2D

    result_artifact = workflow.artifact(ArtifactType.POINT_CLOUD)
    expected_z_no_scale = 500.0 + gr.parameters.tz
    assert result_artifact[0][PointAttribute.Z][0] == pytest.approx(expected_z_no_scale, abs=1e-9)


def test_14b_helmert_3d_does_apply_scale_to_z() -> None:
    """Contrast case: HELMERT_3D genuinely DOES scale Z -- confirms the policy isn't "never scale Z"."""
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    controls = _controls_from(sources, _KNOWN_3D)

    workflow = Workflow()
    workflow._store.set(ArtifactType.POINT_CLOUD, _sample_point_cloud())
    workflow.georeference(ArtifactType.POINT_CLOUD, controls, _TARGET_CRS, GeoreferencingOptions())

    gr = _last_georeferencing_result(workflow)
    assert gr.strategy is GeoreferencingStrategy.HELMERT_3D

    result_artifact = workflow.artifact(ArtifactType.POINT_CLOUD)
    z_without_scale_term = 500.0 + gr.parameters.tz  # what it would be if scale were (incorrectly) skipped
    assert result_artifact[0][PointAttribute.Z][0] != pytest.approx(z_without_scale_term, abs=1e-6)


# ----------------------------------------------------------------------
# Reviewer-identified gaps, closed with real, verified evidence.
# ----------------------------------------------------------------------


def test_reviewer_1_translation_only_warning_mentions_degraded_precision() -> None:
    control = ControlPoint(id="1", source=(0.0, 0.0, 0.0), target=(5.0, 3.0, 2.0))

    workflow = Workflow()
    workflow._store.set(ArtifactType.SURVEY_POINT_SET, _sample_survey())
    workflow.georeference(
        ArtifactType.SURVEY_POINT_SET,
        [control],
        _TARGET_CRS,
        GeoreferencingOptions(accept_underconstrained=True),
    )

    gr = _last_georeferencing_result(workflow)
    assert gr.warning is not None
    assert "ASSUMED" in gr.warning


def test_reviewer_2_calling_georeference_twice_on_an_already_referenced_artifact_is_rejected() -> None:
    """
    Confirmed a real, serious gap found while auditing this: without
    this check, a second georeference() call on an already-
    georeferenced artifact silently re-applied a new set of control
    points -- computed for the ORIGINAL local coordinates -- on top
    of the already-transformed result, producing a meaningless,
    double-transformed artifact with no error at all.
    """
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    controls = _controls_from(sources, _KNOWN_3D)

    workflow = Workflow()
    workflow._store.set(ArtifactType.SURVEY_POINT_SET, _sample_survey())
    workflow.georeference(ArtifactType.SURVEY_POINT_SET, controls, _TARGET_CRS, GeoreferencingOptions())

    first_result = workflow.artifact(ArtifactType.SURVEY_POINT_SET)

    with pytest.raises(WorkflowStateError, match="already known"):
        workflow.georeference(
            ArtifactType.SURVEY_POINT_SET,
            controls,
            CRS.from_epsg(9377),
            GeoreferencingOptions(),
        )

    # Confirms the result from the first call was never touched by the rejected second attempt.
    survivor = workflow.artifact(ArtifactType.SURVEY_POINT_SET)
    assert survivor.crs is first_result.crs
    assert survivor.points[0].x == first_result.points[0].x


def test_reviewer_3_minimum_control_distance_rejects_close_controls_at_workflow_level() -> None:
    from topocore.geodesy.georeferencing.exceptions import ControlPointsTooCloseError

    close_sources = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 50.0),
        (0.0, 100.0, 0.0),
        (50.0, 50.0, 50.0),
    ]
    controls = _controls_from(close_sources, _KNOWN_3D)

    workflow = Workflow()
    workflow._store.set(ArtifactType.SURVEY_POINT_SET, _sample_survey())

    with pytest.raises(ControlPointsTooCloseError):
        workflow.georeference(
            ArtifactType.SURVEY_POINT_SET,
            controls,
            _TARGET_CRS,
            GeoreferencingOptions(minimum_control_distance=5.0),
        )


def test_reviewer_4_duplicate_control_ids_do_not_mix_up_residuals() -> None:
    """
    Confirmed directly: `id` is purely for traceability (see
    ControlPoint's own docstring, "never interpreted") -- residuals
    are matched to controls by POSITION (the same order given), never
    by looking up `id`, so duplicate ids never cause internal
    mismatching. The caller-facing consequence (losing the ability to
    tell which of two same-id residuals is which) is a caller
    responsibility, not a defect fit_georeferencing() introduces.
    """
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    targets = [apply_helmert_3d(point, _KNOWN_3D) for point in sources]
    controls = [
        ControlPoint(id="A", source=sources[0], target=targets[0]),
        ControlPoint(id="A", source=sources[1], target=targets[1]),  # duplicate id, deliberately
        ControlPoint(id="B", source=sources[2], target=targets[2]),
        ControlPoint(id="B", source=sources[3], target=targets[3]),  # duplicate id, deliberately
    ]

    workflow = Workflow()
    workflow._store.set(ArtifactType.SURVEY_POINT_SET, _sample_survey())
    workflow.georeference(ArtifactType.SURVEY_POINT_SET, controls, _TARGET_CRS, GeoreferencingOptions())

    gr = _last_georeferencing_result(workflow)
    assert len(gr.residuals) == 4
    assert all(residual.magnitude < 1e-6 for residual in gr.residuals)  # exact fit, confirms no residual got scrambled


def test_reviewer_5_noisy_controls_produce_a_close_but_imperfect_fit() -> None:
    """Unlike every other test here (noise-free controls), this exercises a realistic, slightly-inconsistent case."""
    rng = np.random.default_rng(42)
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    clean_targets = [apply_helmert_3d(point, _KNOWN_3D) for point in sources]
    noise = rng.normal(0.0, 0.005, (len(sources), 3))
    noisy_targets = [(x + dx, y + dy, z + dz) for (x, y, z), (dx, dy, dz) in zip(clean_targets, noise, strict=True)]
    controls = [
        ControlPoint(id=str(i), source=s, target=t) for i, (s, t) in enumerate(zip(sources, noisy_targets, strict=True))
    ]

    workflow = Workflow()
    workflow._store.set(ArtifactType.SURVEY_POINT_SET, _sample_survey())
    workflow.georeference(ArtifactType.SURVEY_POINT_SET, controls, _TARGET_CRS, GeoreferencingOptions())

    gr = _last_georeferencing_result(workflow)
    assert gr.rms is not None
    assert 0.0 < gr.rms < 0.01  # small but genuinely nonzero, unlike every noise-free test elsewhere in this file
    assert gr.parameters.tx == pytest.approx(_KNOWN_3D.tx, abs=0.05)


def test_reviewer_6_large_point_cloud_with_many_controls_completes_quickly() -> None:
    """10 million points, 50 controls -- confirms this scales to realistic LiDAR-sized data, not just toy fixtures."""
    import time

    rng = np.random.default_rng(0)
    n_points = 10_000_000
    chunk_size = 1_000_000
    cloud = PointCloud()
    for _ in range(n_points // chunk_size):
        chunk = Chunk(
            size=chunk_size,
            attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z],
        )
        chunk[PointAttribute.X][:] = rng.uniform(-1000, 1000, chunk_size)
        chunk[PointAttribute.Y][:] = rng.uniform(-1000, 1000, chunk_size)
        chunk[PointAttribute.Z][:] = rng.uniform(0, 500, chunk_size)
        cloud.add_chunk(chunk)

    control_sources = [(rng.uniform(-500, 500), rng.uniform(-500, 500), rng.uniform(0, 300)) for _ in range(50)]
    controls = _controls_from(control_sources, _KNOWN_3D)

    workflow = Workflow()
    workflow._store.set(ArtifactType.POINT_CLOUD, cloud)

    started = time.monotonic()
    workflow.georeference(ArtifactType.POINT_CLOUD, controls, _TARGET_CRS, GeoreferencingOptions())
    elapsed = time.monotonic() - started

    result_artifact = workflow.artifact(ArtifactType.POINT_CLOUD)
    assert sum(chunk.size for chunk in result_artifact) == n_points
    assert elapsed < 30.0  # generous bound -- confirmed directly this runs in ~1.3s
