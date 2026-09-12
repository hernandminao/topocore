"""
topocore.workflow.stages
============================

The two enums every other Workflow module depends on: `StageStatus`
(outcome of one stage execution) and `WorkflowStage` (which real
stage ran). Deliberately contract-only -- no logic lives here.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from enum import StrEnum


class StageStatus(StrEnum):
    """
    Outcome of a single stage execution within a Workflow.

    SUCCESS
        The stage ran and produced its artifact (or, for export
        stages, its external effect) without error.
    FAILED
        The stage ran but raised. The ArtifactStore is unaffected --
        versions only advance on SUCCESS (see ArtifactStore.set).
    SKIPPED
        The user deliberately chose not to run this stage (e.g. the
        input was already pre-classified, so CLASSIFY_GROUND was
        never needed).
    NOT_RUN
        The stage was never reached at all -- typically because an
        earlier stage in the same call chain FAILED and execution
        stopped before this stage could even be attempted.
    """

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_RUN = "not_run"


class WorkflowStage(StrEnum):
    """
    Every real stage the Workflow engine can execute, named after
    the artifact it reads or produces -- never after an algorithm
    variant or input origin.

    READ_POINT_CLOUD covers LAS, LAZ, PLY, E57, XYZ, CSV and PTS: all
    produce `PointCloud` through the same `PointCloudReader` base
    contract (see the Fase 1 audit and PR19's reader-dispatch fix),
    so the state they lead to -- not which format was on disk -- is
    what the engine tracks.

    CLASSIFY_GROUND and CLASSIFY_POINTS are two distinct stages, not
    one: they wrap genuinely different algorithms with different
    outputs (`GroundManager` -> binary ground/non-ground vs.
    `processing.classification` -> multi-class `ClassificationResult`)
    and must never share a name.

    BUILD_FEATURES_FROM_SURVEY and DETECT_FEATURES are likewise two
    distinct stages for the same reason (`FeatureBuilder` vs.
    `FeatureExtractionManager`), even though both produce
    `FEATURE_COLLECTION`.

    RESOLVE_SIDES both reads and re-produces `FEATURE_COLLECTION` --
    it does not introduce a new artifact type, since it enriches the
    existing collection's own Features (adding `side`/`side_method`
    to `PAVEMENT_EDGE`-type ones) rather than deriving a different
    kind of artifact from it. It is optional and must be explicitly
    chained; nothing else in the pipeline requires it, and running
    it produces no effect on collections that have no `CENTERLINE`/
    `PAVEMENT_EDGE`-type Features at all.

    TRANSFORM_CRS likewise reads and re-produces whichever of
    `POINT_CLOUD`/`GROUND_CLOUD`/`SURVEY_POINT_SET`/
    `FEATURE_COLLECTION` the caller names -- the only 4 types
    `topocore.geodesy.transform` provides a transform function for.
    `TIN`/`DTM`/`CONTOURS`/`CLASSIFICATION_RESULT` are not supported
    (no such function exists for them) and are rejected explicitly,
    not silently ignored. This stage never constructs a `CRS` or
    `CoordinateTransformer` itself -- it applies an already-built one,
    matching `topocore.geodesy.transform`'s own design. It is
    optional: nothing in the pipeline requires georeferencing, and no
    reader in `topocore.io` ever populates a CRS automatically (see
    the project's own geodesy documentation) -- transforming CRS is
    always something the caller opts into explicitly, for whichever
    artifact they choose, at whatever point in the chain they choose.

    TRANSFORM_VERTICAL is TRANSFORM_CRS's vertical-axis counterpart --
    same read-and-re-produce pattern, same 4 supported artifact types,
    but shifts `Z` via `topocore.geodesy.vertical.VerticalTransformer`
    (ellipsoidal <-> orthometric height through a real geoid grid)
    instead of reprojecting `X`/`Y`. It never silently leaves `Z`
    uncorrected: if the required geoid grid is missing, or any point
    falls outside the grid's own extent, the whole stage fails with
    `MissingGeoidGridError` (wrapped in `WorkflowExecutionError` like
    any other stage failure) rather than producing a result where
    some heights were shifted and others quietly were not.
    """

    READ_SURVEY = "read_survey"
    READ_POINT_CLOUD = "read_point_cloud"

    CLASSIFY_GROUND = "classify_ground"
    CLASSIFY_POINTS = "classify_points"

    BUILD_TIN = "build_tin"
    BUILD_DTM = "build_dtm"
    EXTRACT_CONTOURS = "extract_contours"

    BUILD_FEATURES_FROM_SURVEY = "build_features_from_survey"
    DETECT_FEATURES = "detect_features"
    RESOLVE_SIDES = "resolve_sides"
    TRANSFORM_CRS = "transform_crs"
    TRANSFORM_VERTICAL = "transform_vertical"
    GEOREFERENCE = "georeference"

    EXPORT_DXF = "export_dxf"
    EXPORT_GPKG = "export_gpkg"


__all__ = ["StageStatus", "WorkflowStage"]
