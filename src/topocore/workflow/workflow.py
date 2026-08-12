"""
topocore.workflow.workflow
==============================

`Workflow`: the orchestrator. Mutable, represents one execution.
Delegates every piece of infrastructure (artifact bookkeeping,
history, validation, progress) to the modules already frozen in
this package -- this file itself contains no infrastructure logic,
only the bridge between each public stage method and the real
TopoCore module it wraps (`SurveyTXTReader`, `LASReader`,
`GroundManager`, `TIN`, `DTM`, `ContourGenerator`, `FeatureBuilder`,
`FeatureExtractionManager`, `DXFExporter`, `GeoPackageExporter`).

Every public stage method follows the same shape:

    1. `WorkflowValidator.require*(...)` -- precondition check.
    2. Read whatever artifacts the stage needs from `self._store`.
    3. Call `self._execute_stage(...)`, passing a zero-argument
       callable that does the real work.
    4. Return `self`, for fluent chaining.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from topocore.workflow.artifacts import ArtifactStore, ArtifactType
from topocore.workflow.exceptions import WorkflowExecutionError
from topocore.workflow.history import (
    ArtifactDependency,
    StageMetrics,
    StageResult,
    WorkflowResult,
)
from topocore.workflow.progress import ProgressEvent, ProgressObserver
from topocore.workflow.stages import StageStatus, WorkflowStage
from topocore.workflow.validation import WorkflowValidator

logger = logging.getLogger(__name__)

_V = WorkflowValidator


class Workflow:
    """Mutable orchestrator representing one execution. See module docstring."""

    __slots__ = ("_history", "_progress", "_started_at", "_store")

    def __init__(self, *, progress: ProgressObserver | None = None) -> None:
        self._store = ArtifactStore()
        self._history: list[StageResult] = []
        self._progress = progress
        self._started_at = datetime.now(UTC)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def result(self) -> WorkflowResult:
        """
        A `WorkflowResult` snapshot of this Workflow's history so
        far. `finished_at` is the last stage's own `finished_at` (or
        `started_at` if no stage has run yet) -- not wall-clock "now"
        at access time, so two calls to `.result` without any stage
        running in between are identical and reproducible.
        """
        finished_at = self._history[-1].finished_at if self._history else self._started_at
        return WorkflowResult(
            stages=tuple(self._history),
            started_at=self._started_at,
            finished_at=finished_at,
        )

    @property
    def input_kind(self) -> ArtifactType | None:
        """
        Derived, never stored -- the artifact type of the very first
        stage's `produced` artifact, or `None` if nothing has run
        yet. See the Fase 1 audit: this must never be a separately
        assigned field, or it could silently disagree with the
        history it's supposed to describe.
        """
        if not self._history:
            return None
        first_produced = self._history[0].produced
        return first_produced.artifact if first_produced is not None else None

    # ------------------------------------------------------------------
    # Branching
    # ------------------------------------------------------------------

    def snapshot(self) -> Workflow:
        """
        Bifurcate this execution. O(depth of the pipeline so far),
        never O(size of the artifacts) -- see `ArtifactStore.snapshot()`.
        The two Workflows share the same `started_at` (same
        conceptual run, now diverging) and the same `ProgressObserver`
        by default. From this point on, they evolve completely
        independently: nothing done to one is visible to the other.
        """
        branch = Workflow.__new__(Workflow)
        branch._store = self._store.snapshot()
        branch._history = list(self._history)
        branch._progress = self._progress
        branch._started_at = self._started_at
        return branch

    # ------------------------------------------------------------------
    # Internal machinery -- infrastructure only, no domain knowledge
    # ------------------------------------------------------------------

    def _append_stage(self, stage_result: StageResult) -> None:
        self._history.append(stage_result)

    def _emit_progress(self, event: ProgressEvent) -> None:
        if self._progress is None:
            return
        try:
            self._progress.on_progress(event)
        except Exception:
            logger.warning("ProgressObserver raised an exception; ignoring.", exc_info=True)

    def _execute_stage(
        self,
        stage: WorkflowStage,
        work: Callable[[], Any],
        *,
        dependencies: tuple[ArtifactDependency, ...] = (),
        produces: ArtifactType | None = None,
        metrics_fn: Callable[[Any], StageMetrics] | None = None,
    ) -> Any:
        """
        The one place every stage's execution bracket lives: timing,
        exception handling, ArtifactStore writes, and history
        append-only bookkeeping. Never knows about GroundManager,
        TIN, DXF, or any other domain module -- `work` is an opaque
        zero-argument callable supplied by the caller.

        Raises
        ------
        WorkflowExecutionError
            If `work()` raises. The original exception is preserved
            via `__cause__`. A FAILED StageResult is still appended
            to history before raising -- append-only history never
            loses information, even on failure.
        """
        started_at = datetime.now(UTC)
        self._emit_progress(ProgressEvent(stage=stage, message=f"starting {stage.value}"))

        try:
            raw_result = work()
        except Exception as exc:
            finished_at = datetime.now(UTC)
            self._append_stage(
                StageResult(
                    stage=stage,
                    status=StageStatus.FAILED,
                    started_at=started_at,
                    finished_at=finished_at,
                    dependencies=dependencies,
                    produced=None,
                    error=exc,
                )
            )
            raise WorkflowExecutionError(f"Stage {stage.value} failed: {exc}") from exc

        produced_dependency: ArtifactDependency | None = None
        if produces is not None:
            version = self._store.set(produces, raw_result)
            produced_dependency = ArtifactDependency(artifact=produces, version=version, required=True)

        metrics = metrics_fn(raw_result) if metrics_fn is not None else None
        finished_at = datetime.now(UTC)
        self._append_stage(
            StageResult(
                stage=stage,
                status=StageStatus.SUCCESS,
                started_at=started_at,
                finished_at=finished_at,
                dependencies=dependencies,
                produced=produced_dependency,
                metrics=metrics,
            )
        )
        self._emit_progress(ProgressEvent(stage=stage, message=f"finished {stage.value}"))
        return raw_result

    def _dependency(self, artifact_type: ArtifactType, *, required: bool) -> ArtifactDependency:
        """Build an ArtifactDependency at the artifact's CURRENT version."""
        version = self._store.version_of(artifact_type)
        assert version is not None  # caller must have already validated presence via WorkflowValidator
        return ArtifactDependency(artifact=artifact_type, version=version, required=required)

    # ------------------------------------------------------------------
    # READ stages -- the two mutually exclusive roots of the DAG
    # ------------------------------------------------------------------

    def read_survey(self, path: str | Path, **reader_kwargs: Any) -> Workflow:
        """
        Requires
        --------
        Nothing -- but no prior READ stage may have run on this Workflow.

        Produces
        --------
        ArtifactType.SURVEY_POINT_SET
        """
        _V.require_absent(
            WorkflowStage.READ_SURVEY,
            self._store,
            ArtifactType.SURVEY_POINT_SET,
            ArtifactType.POINT_CLOUD,
        )

        def work() -> Any:
            from topocore.survey import SurveyTXTReader

            return SurveyTXTReader(path, **reader_kwargs).read()

        self._execute_stage(
            WorkflowStage.READ_SURVEY,
            work,
            produces=ArtifactType.SURVEY_POINT_SET,
            metrics_fn=lambda result: StageMetrics(output_count=len(result)),
        )
        return self

    def read_point_cloud(self, path: str | Path, **reader_kwargs: Any) -> Workflow:
        """
        Requires
        --------
        Nothing -- but no prior READ stage may have run on this Workflow.

        Produces
        --------
        ArtifactType.POINT_CLOUD

        Covers both LAS and LAZ: both produce `PointCloud` through
        the same `PointCloudReader.read()` base contract, selected
        here by file extension.
        """
        _V.require_absent(
            WorkflowStage.READ_POINT_CLOUD,
            self._store,
            ArtifactType.SURVEY_POINT_SET,
            ArtifactType.POINT_CLOUD,
        )

        def work() -> Any:
            from topocore.io.base import PointCloudReader

            suffix = Path(path).suffix.lower()
            reader_class: type[PointCloudReader]
            if suffix == ".laz":
                from topocore.io.laz import LAZReader

                reader_class = LAZReader
            else:
                from topocore.io.las import LASReader

                reader_class = LASReader

            with reader_class(path, **reader_kwargs) as reader:
                return reader.read()

        self._execute_stage(
            WorkflowStage.READ_POINT_CLOUD,
            work,
            produces=ArtifactType.POINT_CLOUD,
            metrics_fn=lambda result: StageMetrics(output_count=result.point_count),
        )
        return self

    # ------------------------------------------------------------------
    # Classification -- two genuinely distinct algorithms, never one method
    # ------------------------------------------------------------------

    def classify_ground(self, **manager_kwargs: Any) -> Workflow:
        """
        Requires
        --------
        ArtifactType.POINT_CLOUD

        Produces
        --------
        ArtifactType.GROUND_CLOUD

        Calls `GroundManager.extract()` exactly once -- never
        `.classify()` too, which would run the same algorithm twice
        just to also have the boolean mask.
        """
        _V.require(WorkflowStage.CLASSIFY_GROUND, self._store, ArtifactType.POINT_CLOUD)
        cloud = self._store.get(ArtifactType.POINT_CLOUD)
        dep = self._dependency(ArtifactType.POINT_CLOUD, required=True)

        def work() -> Any:
            from topocore.processing.ground import GroundManager

            return GroundManager(**manager_kwargs).extract(cloud)

        def metrics_fn(ground_cloud: Any) -> StageMetrics:
            ground_points = ground_cloud.point_count
            return StageMetrics(
                input_count=cloud.point_count,
                output_count=ground_points,
                extra={"non_ground_points": cloud.point_count - ground_points},
            )

        self._execute_stage(
            WorkflowStage.CLASSIFY_GROUND,
            work,
            dependencies=(dep,),
            produces=ArtifactType.GROUND_CLOUD,
            metrics_fn=metrics_fn,
        )
        return self

    def classify_points(self, *, method: str = "auto", **manager_kwargs: Any) -> Workflow:
        """
        Requires
        --------
        ArtifactType.POINT_CLOUD

        Produces
        --------
        ArtifactType.CLASSIFICATION_RESULT

        Distinct from `classify_ground()`: this is multi-class ASPRS
        classification (`processing.classification.ClassificationManager`),
        not the binary ground/non-ground split `GroundManager` performs.
        """
        _V.require(WorkflowStage.CLASSIFY_POINTS, self._store, ArtifactType.POINT_CLOUD)
        cloud = self._store.get(ArtifactType.POINT_CLOUD)
        dep = self._dependency(ArtifactType.POINT_CLOUD, required=True)

        def work() -> Any:
            from topocore.processing.classification import ClassificationManager

            return ClassificationManager(method=method, **manager_kwargs).classify(cloud)

        self._execute_stage(
            WorkflowStage.CLASSIFY_POINTS,
            work,
            dependencies=(dep,),
            produces=ArtifactType.CLASSIFICATION_RESULT,
            metrics_fn=lambda result: StageMetrics(input_count=cloud.point_count, output_count=len(result.labels)),
        )
        return self

    # ------------------------------------------------------------------
    # Terrain -- TIN is the root; DTM and CONTOURS are parallel, not sequential
    # ------------------------------------------------------------------

    def build_tin(self) -> Workflow:
        """
        Requires
        --------
        ArtifactType.GROUND_CLOUD

        Produces
        --------
        ArtifactType.TIN
        """
        _V.require(WorkflowStage.BUILD_TIN, self._store, ArtifactType.GROUND_CLOUD)
        ground_cloud = self._store.get(ArtifactType.GROUND_CLOUD)
        dep = self._dependency(ArtifactType.GROUND_CLOUD, required=True)

        def work() -> Any:
            from topocore.terrain.conversion import pointcloud_to_points
            from topocore.terrain.tin import TIN

            points = pointcloud_to_points(ground_cloud)
            return TIN.from_points(points)

        self._execute_stage(
            WorkflowStage.BUILD_TIN,
            work,
            dependencies=(dep,),
            produces=ArtifactType.TIN,
            metrics_fn=lambda tin: StageMetrics(input_count=ground_cloud.point_count, output_count=len(tin.vertices)),
        )
        return self

    def build_dtm(self, grid: Any, interpolator: Any) -> Workflow:
        """
        Requires
        --------
        ArtifactType.TIN

        Produces
        --------
        ArtifactType.DTM

        `grid` and `interpolator` are required, explicit parameters
        -- no implicit default: Workflow never decides scientific
        algorithm choices on the user's behalf.
        """
        _V.require(WorkflowStage.BUILD_DTM, self._store, ArtifactType.TIN)
        tin = self._store.get(ArtifactType.TIN)
        dep = self._dependency(ArtifactType.TIN, required=True)

        def work() -> Any:
            from topocore.terrain.dtm import DTM

            return DTM.from_tin(tin, grid, interpolator)

        self._execute_stage(
            WorkflowStage.BUILD_DTM,
            work,
            dependencies=(dep,),
            produces=ArtifactType.DTM,
        )
        return self

    def extract_contours(self, interval: float, *, base: float = 0.0) -> Workflow:
        """
        Requires
        --------
        ArtifactType.TIN

        Produces
        --------
        ArtifactType.CONTOURS

        Consumes TIN directly -- never DTM.
        """
        _V.require(WorkflowStage.EXTRACT_CONTOURS, self._store, ArtifactType.TIN)
        tin = self._store.get(ArtifactType.TIN)
        dep = self._dependency(ArtifactType.TIN, required=True)

        def work() -> Any:
            from topocore.terrain.contours import ContourGenerator

            return ContourGenerator(tin).generate(interval, base=base)

        self._execute_stage(
            WorkflowStage.EXTRACT_CONTOURS,
            work,
            dependencies=(dep,),
            produces=ArtifactType.CONTOURS,
            metrics_fn=lambda contours: StageMetrics(output_count=len(contours)),
        )
        return self

    # ------------------------------------------------------------------
    # Features -- two genuinely distinct algorithms, never one method
    # ------------------------------------------------------------------

    def build_features_from_survey(self, registry: Any = None) -> Workflow:
        """
        Requires
        --------
        ArtifactType.SURVEY_POINT_SET

        Produces
        --------
        ArtifactType.FEATURE_COLLECTION

        Uses `FeatureBuilder` (field-code interpretation) -- distinct
        from `detect_features()`, which uses `FeatureExtractionManager`.
        """
        _V.require(
            WorkflowStage.BUILD_FEATURES_FROM_SURVEY,
            self._store,
            ArtifactType.SURVEY_POINT_SET,
        )
        survey_points = self._store.get(ArtifactType.SURVEY_POINT_SET)
        dep = self._dependency(ArtifactType.SURVEY_POINT_SET, required=True)

        def work() -> Any:
            from topocore.features.feature_builder import FeatureBuilder
            from topocore.features.feature_codes import FeatureCodeRegistry

            active_registry = registry if registry is not None else FeatureCodeRegistry.default()
            build_result = FeatureBuilder(active_registry).build(survey_points)
            return build_result.features

        self._execute_stage(
            WorkflowStage.BUILD_FEATURES_FROM_SURVEY,
            work,
            dependencies=(dep,),
            produces=ArtifactType.FEATURE_COLLECTION,
            metrics_fn=lambda features: StageMetrics(input_count=len(survey_points), output_count=len(features)),
        )
        return self

    def detect_features(self, *, strict: bool = True) -> Workflow:
        """
        Requires
        --------
        ArtifactType.POINT_CLOUD

        Optional
        --------
        ArtifactType.TIN, ArtifactType.DTM, ArtifactType.CLASSIFICATION_RESULT,
        ArtifactType.GROUND_CLOUD -- whichever are available are
        passed through.

        Produces
        --------
        ArtifactType.FEATURE_COLLECTION
        """
        _V.require(WorkflowStage.DETECT_FEATURES, self._store, ArtifactType.POINT_CLOUD)
        cloud = self._store.get(ArtifactType.POINT_CLOUD)

        dependencies = [self._dependency(ArtifactType.POINT_CLOUD, required=True)]
        optional_types = (
            ArtifactType.TIN,
            ArtifactType.DTM,
            ArtifactType.CLASSIFICATION_RESULT,
            ArtifactType.GROUND_CLOUD,
        )
        for artifact_type in optional_types:
            if self._store.has(artifact_type):
                dependencies.append(self._dependency(artifact_type, required=False))

        tin = self._store.get_or_none(ArtifactType.TIN)
        dtm = self._store.get_or_none(ArtifactType.DTM)
        classification = self._store.get_or_none(ArtifactType.CLASSIFICATION_RESULT)

        def work() -> Any:
            from topocore.features.manager import FeatureExtractionManager
            from topocore.features.protocols import DetectionContext

            context = DetectionContext(cloud=cloud, tin=tin, dtm=dtm, classification=classification)
            return FeatureExtractionManager(strict=strict).detect_all(context)

        self._execute_stage(
            WorkflowStage.DETECT_FEATURES,
            work,
            dependencies=tuple(dependencies),
            produces=ArtifactType.FEATURE_COLLECTION,
            metrics_fn=lambda features: StageMetrics(input_count=cloud.point_count, output_count=len(features)),
        )
        return self

    # ------------------------------------------------------------------
    # Exports -- lateral operations, never write to the ArtifactStore
    # ------------------------------------------------------------------

    def export_dxf(self, path: str | Path, **exporter_kwargs: Any) -> Workflow:
        """
        Requires
        --------
        ArtifactType.FEATURE_COLLECTION

        Never writes to the ArtifactStore -- may be called any
        number of times, including alongside `export_gpkg()`.
        """
        _V.require(WorkflowStage.EXPORT_DXF, self._store, ArtifactType.FEATURE_COLLECTION)
        features = self._store.get(ArtifactType.FEATURE_COLLECTION)
        dep = self._dependency(ArtifactType.FEATURE_COLLECTION, required=True)

        def work() -> Any:
            from topocore.dxf import DXFExporter

            return DXFExporter(**exporter_kwargs).export(features, path)

        self._execute_stage(WorkflowStage.EXPORT_DXF, work, dependencies=(dep,), produces=None)
        return self

    def export_gpkg(self, path: str | Path, *, epsg: int, **exporter_kwargs: Any) -> Workflow:
        """
        Requires
        --------
        ArtifactType.FEATURE_COLLECTION

        Never writes to the ArtifactStore -- may be called any
        number of times, including alongside `export_dxf()`.
        """
        _V.require(WorkflowStage.EXPORT_GPKG, self._store, ArtifactType.FEATURE_COLLECTION)
        features = self._store.get(ArtifactType.FEATURE_COLLECTION)
        dep = self._dependency(ArtifactType.FEATURE_COLLECTION, required=True)

        def work() -> Any:
            from topocore.gpkg import GeoPackageExporter, GPKGExportOptions

            options = GPKGExportOptions(epsg=epsg, **exporter_kwargs)
            return GeoPackageExporter(options).export(features, path)

        self._execute_stage(WorkflowStage.EXPORT_GPKG, work, dependencies=(dep,), produces=None)
        return self


__all__ = ["Workflow"]
