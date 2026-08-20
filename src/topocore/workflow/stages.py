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

    EXPORT_DXF = "export_dxf"
    EXPORT_GPKG = "export_gpkg"


__all__ = ["StageStatus", "WorkflowStage"]
