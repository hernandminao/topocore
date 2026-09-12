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

import dataclasses
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from topocore.terrain.interpolation import InterpolationMethod

from topocore.workflow.artifacts import ArtifactStore, ArtifactType
from topocore.workflow.exceptions import (
    StaleArtifactError,
    WorkflowExecutionError,
    WorkflowStateError,
)
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

#: Default chunk_size supplied to E57Reader when the caller doesn't
#: give one -- E57Reader.chunk_size has no built-in default (unlike
#: every other point-cloud reader), confirmed directly. Matches the
#: default already used by LASReader/LAZReader/PLYReader/ASCII readers.
_DEFAULT_CHUNK_SIZE = 1_000_000

#: The only 4 ArtifactType members topocore.geodesy.transform provides
#: a transform function for -- transform_crs() rejects any other type
#: explicitly rather than silently ignoring it. GROUND_CLOUD shares
#: transform_point_cloud() with POINT_CLOUD, since both are the same
#: PointCloud type (confirmed directly).
_TRANSFORMABLE_ARTIFACT_TYPES = (
    ArtifactType.POINT_CLOUD,
    ArtifactType.GROUND_CLOUD,
    ArtifactType.SURVEY_POINT_SET,
    ArtifactType.FEATURE_COLLECTION,
)


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

    def artifact(self, artifact_type: ArtifactType) -> Any:
        """
        Returns the current value of `artifact_type`, safe to modify
        without ever silently corrupting this Workflow's own stored
        version -- the only way to advance a stored artifact's own
        version remains running the stage that produces it.

        Raises
        ------
        WorkflowStateError
            If `artifact_type` has never been produced by this
            Workflow.
        StaleArtifactError
            If `artifact_type` is present but was built (directly or
            transitively) from an input that has since been
            superseded -- the exact same staleness check every stage
            method already applies to its own inputs via
            `WorkflowValidator.require_current()`. `artifact()` is
            not itself a `WorkflowStage` (it executes nothing and
            never appends to history), so this check is applied
            directly rather than through that machinery's own
            stage-shaped error message.

        Copy strategy, decided per type from this project's own
        confirmed-by-execution mutability audit -- never a uniform
        "always copy" or "always return as-is":

        - `SURVEY_POINT_SET`, `CONTOURS`, `TIN`: returned directly,
          at no extra cost. Each is either genuinely immutable
          (frozen dataclass over tuples of frozen elements) or
          already self-protecting (`TIN.simplices`/`.neighbors` are
          properties that already return a defensive copy on every
          access, confirmed directly -- not by virtue of being
          frozen, since `TIN` itself is not).
        - `POINT_CLOUD`, `GROUND_CLOUD` (both `PointCloud`):
          `.clone()` -- an existing, already-verified deep copy.
        - `DTM`: reconstructed via `dataclasses.replace()` with only
          `raster` copied (`Raster.copy()`, existing and verified) --
          `.tin` needs no copy, since `TIN` is already safe.
        - `CLASSIFICATION_RESULT`: reconstructed via
          `dataclasses.replace()` with `labels`/`confidence` (numpy
          arrays, confirmed mutable even though the dataclass itself
          is frozen) and `cloud` (via its own `.clone()`) all copied.
        - `FEATURE_COLLECTION`: a new `FeatureCollection` wrapping a
          new list of the same `Feature` references -- safe because
          `Feature` is itself already frozen; only the containing
          list needed protecting, not each Feature.
        """
        if not self._store.has(artifact_type):
            raise WorkflowStateError(f"Artifact {artifact_type.value} has never been produced by this Workflow.")

        if WorkflowValidator._is_stale(self._store, self._history, artifact_type):
            raise StaleArtifactError(
                f"Artifact {artifact_type.value} was built from since-superseded input(s); "
                "rebuild it (and anything downstream of it) before retrieving it."
            )

        current = self._store.get(artifact_type)

        if artifact_type in (
            ArtifactType.SURVEY_POINT_SET,
            ArtifactType.CONTOURS,
            ArtifactType.TIN,
        ):
            return current

        if artifact_type in (ArtifactType.POINT_CLOUD, ArtifactType.GROUND_CLOUD):
            return current.clone()

        if artifact_type is ArtifactType.DTM:
            return dataclasses.replace(current, raster=current.raster.copy())

        if artifact_type is ArtifactType.CLASSIFICATION_RESULT:
            return dataclasses.replace(
                current,
                labels=current.labels.copy(),
                cloud=current.cloud.clone(),
                confidence=current.confidence.copy() if current.confidence is not None else None,
            )

        if artifact_type is ArtifactType.FEATURE_COLLECTION:
            from topocore.features.models import FeatureCollection

            # Found and fixed while wiring FeatureCollection.crs
            # (added after this defensive-copy branch was originally
            # written): the reconstruction below previously dropped
            # `.crs` back to its default (None) unconditionally --
            # confirmed directly, a FeatureCollection stored with a
            # real, detected CRS came back as crs=None through this
            # accessor, even though the stored object itself was
            # correct. `crs` is now carried through explicitly.
            return FeatureCollection(features=list(current.features), crs=current.crs)

        raise AssertionError(f"Unhandled ArtifactType in artifact(): {artifact_type.value}.")  # pragma: no cover

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

        Parameters
        ----------
        work
            Does the actual domain work and returns its raw result.
        dependencies
            Every artifact (required or optional) this stage reads,
            already resolved to `ArtifactDependency` by the caller
            (who knows which artifacts it read and at what version).
        produces
            If given, `work()`'s return value is written to
            `self._store` under this `ArtifactType` on success. `None`
            for stages that never write to the store (export stages).
        metrics_fn
            Optional: computes `StageMetrics` from `work()`'s result,
            only called on success.

        Returns
        -------
        Any
            Whatever `work()` returned.

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
        """Build an ArtifactDependency at the artifact's CURRENT version -- called right before _execute_stage, so the version recorded is exactly what the stage is about to consume."""
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

        Supports LAS, LAZ, PLY, E57, XYZ, CSV, and PTS -- selected by
        file extension (case-insensitive). All produce `PointCloud`
        through the same `PointCloudReader.read()` base contract.
        `**reader_kwargs` are passed straight through to whichever
        reader class is selected (e.g. `chunk_size` for most formats,
        `has_header` for CSV specifically) -- unrecognized kwargs for
        a given format raise the same `TypeError` they always would
        constructing that reader directly.

        Found and fixed in PR19: this previously only distinguished
        ".laz" from everything else, silently routing PLY/E57/XYZ/CSV/
        PTS files (and any unrecognized extension) to `LASReader`.
        """
        _V.require_absent(
            WorkflowStage.READ_POINT_CLOUD,
            self._store,
            ArtifactType.SURVEY_POINT_SET,
            ArtifactType.POINT_CLOUD,
        )

        def work() -> Any:
            from topocore.io.base import PointCloudReader

            input_path = Path(path)

            if not input_path.exists():
                raise WorkflowExecutionError(f"Point-cloud file does not exist: '{input_path}'.")

            if not input_path.is_file():
                raise WorkflowExecutionError(f"Point-cloud path is not a file: '{input_path}'.")

            suffix = input_path.suffix.lower()
            reader_class: type[PointCloudReader]
            reader_kwargs_final = dict(reader_kwargs)

            # Found and fixed in PR19: this previously only
            # distinguished ".laz" from everything else (defaulting
            # unconditionally to LASReader) -- confirmed directly
            # that ANY non-LAZ format (.ply, .e57, .xyz, .csv, .pts,
            # or an unrecognized extension) silently got routed to
            # LASReader, which would fail or misparse rather than use
            # the correct, already-existing reader for that format.
            #
            # Import paths verified against the real package
            # structure, not assumed: topocore.io.ply/__init__.py,
            # topocore.io.ascii.csv/__init__.py, and
            # topocore.io.ascii.pts/__init__.py do NOT re-export their
            # Reader classes (unlike las/laz/e57/ascii.xyz, which do)
            # -- those three must be imported from their `.reader`
            # submodule directly, confirmed by direct import testing.
            if suffix == ".las":
                from topocore.io.las import LASReader

                reader_class = LASReader
            elif suffix == ".laz":
                from topocore.io.laz import LAZReader

                reader_class = LAZReader
            elif suffix == ".ply":
                from topocore.io.ply.reader import PLYReader

                reader_class = PLYReader
            elif suffix == ".e57":
                from topocore.io.e57 import E57Reader

                reader_class = E57Reader
                # E57Reader.chunk_size is a REQUIRED keyword argument
                # (no default) -- confirmed directly, unlike every
                # other reader here, which all have a sensible
                # built-in default. Supplying one here keeps
                # read_point_cloud()'s own contract uniform across
                # formats: the caller shouldn't need to know which
                # one format has no built-in default.
                reader_kwargs_final.setdefault("chunk_size", _DEFAULT_CHUNK_SIZE)
            elif suffix == ".xyz":
                from topocore.io.ascii.xyz import XYZReader

                reader_class = XYZReader
            elif suffix == ".csv":
                from topocore.io.ascii.csv.reader import CSVReader

                reader_class = CSVReader
            elif suffix == ".pts":
                from topocore.io.ascii.pts.reader import PTSReader

                reader_class = PTSReader
            else:
                raise WorkflowExecutionError(
                    f"Unsupported point-cloud format: '{suffix or '<none>'}'. "
                    "Supported formats: LAS, LAZ, PLY, E57, XYZ, CSV, PTS."
                )

            with reader_class(input_path, **reader_kwargs_final) as reader:
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
        just to also have the boolean mask (see the Fase 2 audit:
        no real consumer needs the raw mask; ground/non-ground point
        counts are cheaper to derive from `.point_count`).
        """
        _V.require(WorkflowStage.CLASSIFY_GROUND, self._store, ArtifactType.POINT_CLOUD)
        _V.require_current(
            WorkflowStage.CLASSIFY_GROUND,
            self._store,
            self._history,
            ArtifactType.POINT_CLOUD,
        )
        cloud = self._store.get(ArtifactType.POINT_CLOUD)
        dep = self._dependency(ArtifactType.POINT_CLOUD, required=True)

        def work() -> Any:
            from topocore.processing.ground import GroundManager

            ground_cloud = GroundManager(**manager_kwargs).extract(cloud)
            # Confirmed a real gap found while adding transform_crs()'s
            # own new safety check: GroundManager.extract() constructs
            # a fresh PointCloud for the ground subset, and never
            # propagated the source cloud's own metadata.crs -- always
            # None, regardless of what the input actually had. Ground
            # classification is a pure geometric subset (points
            # removed, none moved or reprojected), so the CRS the
            # remaining points are expressed in is unchanged; not
            # propagating it here would have made
            # `transform_crs(GROUND_CLOUD, ...)` unusable in any
            # normal pipeline once that stage's own "CRS must be
            # known" check was added.
            ground_cloud.crs = cloud.crs
            return ground_cloud

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
        _V.require_current(
            WorkflowStage.CLASSIFY_POINTS,
            self._store,
            self._history,
            ArtifactType.POINT_CLOUD,
        )
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
        _V.require_current(
            WorkflowStage.BUILD_TIN,
            self._store,
            self._history,
            ArtifactType.GROUND_CLOUD,
        )
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

    def build_dtm(self, grid: Any, *, method: InterpolationMethod, power: float = 2.0) -> Workflow:
        """
        Requires
        --------
        ArtifactType.TIN

        Produces
        --------
        ArtifactType.DTM

        `grid` and `method` are required, explicit parameters -- no
        implicit default (Opción A, per the frozen contract):
        Workflow never decides scientific algorithm choices on the
        user's behalf. `power` (only meaningful for
        `InterpolationMethod.IDW`) keeps `TerrainInterpolator`'s own
        default, since it's a secondary tuning parameter, not a
        fundamental algorithm choice.

        Workflow constructs the `TerrainInterpolator` internally,
        bound to the `TIN` it already holds -- the user never needs
        to (and, with the public API alone, has no way to) obtain
        that `TIN` object to build one themselves. This mirrors
        every other stage: the user never constructs `GroundManager`,
        `ClassificationManager`, `DXFExporter`, or
        `GeoPackageExporter` directly either -- Workflow always does.
        """
        _V.require(WorkflowStage.BUILD_DTM, self._store, ArtifactType.TIN)
        _V.require_current(WorkflowStage.BUILD_DTM, self._store, self._history, ArtifactType.TIN)
        tin = self._store.get(ArtifactType.TIN)
        dep = self._dependency(ArtifactType.TIN, required=True)

        def work() -> Any:
            from topocore.terrain.dtm import DTM
            from topocore.terrain.interpolation import InterpolationMethod

            interpolator: Any
            if method == InterpolationMethod.LINEAR:
                from topocore.terrain.linear import LinearInterpolator

                interpolator = LinearInterpolator(tin)
            elif method == InterpolationMethod.BARYCENTRIC:
                from topocore.terrain.barycentric import BarycentricInterpolator

                interpolator = BarycentricInterpolator(tin)
            elif method == InterpolationMethod.IDW:
                from topocore.terrain.idw import IDWInterpolator

                interpolator = IDWInterpolator(tin, power=power)
            else:
                from topocore.terrain.nearest import NearestInterpolator

                interpolator = NearestInterpolator(tin)

            # DTM.from_tin() types `interpolator` as `BaseInterpolator`,
            # but none of the four concrete interpolator classes above
            # actually declare that inheritance (a pre-existing gap in
            # topocore.terrain, not introduced here) -- verified
            # correct at runtime (DTM.from_tin only ever calls
            # `.interpolate(...)` on it, duck-typed), reported to
            # Hernán rather than silently worked around with an
            # unrelated wrapper type.
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

        Consumes TIN directly -- never DTM (see the Fase 1 audit:
        `ContourGenerator` has no dependency on `DTM` at all).
        """
        _V.require(WorkflowStage.EXTRACT_CONTOURS, self._store, ArtifactType.TIN)
        _V.require_current(WorkflowStage.EXTRACT_CONTOURS, self._store, self._history, ArtifactType.TIN)
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
        _V.require_current(
            WorkflowStage.BUILD_FEATURES_FROM_SURVEY,
            self._store,
            self._history,
            ArtifactType.SURVEY_POINT_SET,
        )
        survey_points = self._store.get(ArtifactType.SURVEY_POINT_SET)
        dep = self._dependency(ArtifactType.SURVEY_POINT_SET, required=True)

        def work() -> Any:
            from topocore.features.feature_builder import FeatureBuilder
            from topocore.features.feature_codes import FeatureCodeRegistry

            active_registry = registry if registry is not None else FeatureCodeRegistry.default()
            build_result = FeatureBuilder(active_registry).build(survey_points)
            features = build_result.features
            # survey_points.crs is a real CRS object (SurveyPointSet's
            # own convention); FeatureCollection.crs is a string
            # (PointCloud's own convention -- see FeatureCollection's
            # own docstring for why). Converted here via the same
            # f"EPSG:{code}" or .name shape already established by
            # transform_point_cloud().
            if survey_points.crs is not None:
                source_crs = survey_points.crs
                features.crs = f"EPSG:{source_crs.epsg}" if source_crs.epsg is not None else source_crs.name
            return features

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
        passed through; `FeatureExtractionManager` itself decides
        which detectors can run with what's present (same behavior
        as calling it directly, non-strict skips unmet detectors).

        Produces
        --------
        ArtifactType.FEATURE_COLLECTION
        """
        _V.require(WorkflowStage.DETECT_FEATURES, self._store, ArtifactType.POINT_CLOUD)
        _V.require_current(
            WorkflowStage.DETECT_FEATURES,
            self._store,
            self._history,
            ArtifactType.POINT_CLOUD,
        )
        cloud = self._store.get(ArtifactType.POINT_CLOUD)

        dependencies = [self._dependency(ArtifactType.POINT_CLOUD, required=True)]
        # Found and fixed in PR20 coverage phase: this previously
        # also tracked ArtifactType.GROUND_CLOUD here (both for
        # staleness AND as a recorded dependency), but
        # DetectionContext (see topocore.features.protocols) has NO
        # `ground_cloud` field at all -- confirmed directly: its
        # only fields are cloud/tin/dtm/classification/normals/
        # pca_features. No detector reached through this pathway can
        # ever actually consume GROUND_CLOUD data. Tracking its
        # staleness anyway was a real, reproducible false-positive
        # bug: detect_features() incorrectly raised StaleArtifactError
        # whenever GROUND_CLOUD alone became stale, even though the
        # actual inputs detect_features() uses were fully fresh and
        # correct -- confirmed with a direct reproduction before
        # fixing. This was introduced earlier in this same session
        # when require_current() was first wired in, not pre-existing.
        optional_types = (
            ArtifactType.TIN,
            ArtifactType.DTM,
            ArtifactType.CLASSIFICATION_RESULT,
        )
        for artifact_type in optional_types:
            if self._store.has(artifact_type):
                _V.require_current(
                    WorkflowStage.DETECT_FEATURES,
                    self._store,
                    self._history,
                    artifact_type,
                )
                dependencies.append(self._dependency(artifact_type, required=False))

        tin = self._store.get_or_none(ArtifactType.TIN)
        dtm = self._store.get_or_none(ArtifactType.DTM)
        classification = self._store.get_or_none(ArtifactType.CLASSIFICATION_RESULT)

        def work() -> Any:
            from topocore.features.manager import FeatureExtractionManager
            from topocore.features.protocols import DetectionContext

            context = DetectionContext(cloud=cloud, tin=tin, dtm=dtm, classification=classification)
            result = FeatureExtractionManager(strict=strict).detect_all(context)
            # cloud is the same POINT_CLOUD still accessible in this
            # Workflow's own store (confirmed directly, regardless of
            # whether TIN/DTM were also used) -- FeatureCollection has
            # no CRS mechanism of its own, so this is the only source
            # it can ever get one from.
            result.crs = cloud.crs
            return result

        self._execute_stage(
            WorkflowStage.DETECT_FEATURES,
            work,
            dependencies=tuple(dependencies),
            produces=ArtifactType.FEATURE_COLLECTION,
            metrics_fn=lambda features: StageMetrics(input_count=cloud.point_count, output_count=len(features)),
        )
        return self

    def resolve_sides(self, **resolver_kwargs: Any) -> Workflow:
        """
        Requires
        --------
        ArtifactType.FEATURE_COLLECTION

        Produces
        --------
        ArtifactType.FEATURE_COLLECTION (re-produced, not a new
        artifact type -- see WorkflowStage's own docstring for why)

        Optional stage: resolves left/right laterality for linear
        features (`FeatureType.PAVEMENT_EDGE` by default) relative to
        a reference `FeatureType.CENTERLINE`, via
        `topocore.features.side.SideResolver`. Must be explicitly
        chained after `build_features_from_survey()` or
        `detect_features()` -- never runs implicitly. A collection
        with no CENTERLINE/PAVEMENT_EDGE-type Features passes through
        with no Features modified; this is not an error.

        `**resolver_kwargs` are forwarded to `SideResolver.__init__`
        (`target_types`, `max_distance`, `ambiguity_margin`,
        `cross_tolerance`) -- unrecognized kwargs raise the same
        `TypeError` constructing `SideResolver` directly would.
        """
        _V.require(WorkflowStage.RESOLVE_SIDES, self._store, ArtifactType.FEATURE_COLLECTION)
        _V.require_current(
            WorkflowStage.RESOLVE_SIDES,
            self._store,
            self._history,
            ArtifactType.FEATURE_COLLECTION,
        )
        features = self._store.get(ArtifactType.FEATURE_COLLECTION)
        dep = self._dependency(ArtifactType.FEATURE_COLLECTION, required=True)

        def work() -> Any:
            from topocore.features.side import SideResolver

            return SideResolver(**resolver_kwargs).resolve(features)

        self._execute_stage(
            WorkflowStage.RESOLVE_SIDES,
            work,
            dependencies=(dep,),
            produces=ArtifactType.FEATURE_COLLECTION,
            metrics_fn=lambda result: StageMetrics(input_count=len(features), output_count=len(result)),
        )
        return self

    def transform_crs(self, artifact_type: ArtifactType, transformer: Any) -> Workflow:
        """
        Requires
        --------
        `artifact_type` -- must currently be present, and must be one
        of `POINT_CLOUD`, `GROUND_CLOUD`, `SURVEY_POINT_SET`,
        `FEATURE_COLLECTION`. These are the only 4 types
        `topocore.geodesy.transform` provides a transform function
        for; any other `artifact_type` (`TIN`, `DTM`, `CONTOURS`,
        `CLASSIFICATION_RESULT`) is rejected explicitly with
        `WorkflowStateError`, not silently ignored or passed through
        unchanged.

        Produces
        --------
        `artifact_type` (re-produced in place -- same pattern as
        `RESOLVE_SIDES`; this does not introduce a new artifact type).

        Applies `transformer` (an already-built
        `topocore.geodesy.CoordinateTransformer`) to the current
        value of `artifact_type`, via whichever of
        `transform_point_cloud()`/`transform_survey()`/
        `transform_feature_collection()` matches. `GROUND_CLOUD` uses
        the same `transform_point_cloud()` as `POINT_CLOUD`, since
        both are the exact same `PointCloud` type -- confirmed
        directly, `transform_point_cloud()` has no special handling
        that would distinguish a "full" cloud from a ground-only one.

        This stage never constructs a `CRS` or `CoordinateTransformer`
        itself -- you build one yourself first (see
        `topocore.geodesy`), matching `topocore.geodesy.transform`'s
        own design (which never decides source/target CRS on the
        caller's behalf either). It is optional: no reader in
        `topocore.io` ever populates a CRS automatically, so nothing
        else in the pipeline requires this stage to have run.

        Confirmed directly: an artifact transformed here that has
        downstream consumers already produced from its pre-transform
        version (e.g. `FEATURE_COLLECTION` built from a `POINT_CLOUD`
        you then transform, or vice versa) does not automatically
        re-run -- the existing `ArtifactStore` staleness mechanism
        (`WorkflowValidator.require_current()`, `Workflow.artifact()`'s
        own `StaleArtifactError`) is what surfaces this the next time
        that downstream artifact is touched, exactly as it already
        does for every other stage. This stage does not need, and
        does not add, any new staleness logic of its own.

        Two safety checks, for `POINT_CLOUD`/`GROUND_CLOUD`/
        `SURVEY_POINT_SET`/`FEATURE_COLLECTION` (all 4 supported
        artifact types -- `FeatureCollection` gained its own `.crs`
        after these checks were first written; the earlier gap where
        it had no `.crs` concept at all is now closed):

        1. **The artifact's current CRS must be known.** `crs=None`
           means "this origin CRS is genuinely unknown" -- never
           "assume it's whatever CRS you're about to transform to".
           Confirmed directly: without this check, a `SurveyPointSet`
           of raw, local total-station coordinates (`crs=None`, no
           `.prj`) transformed silently, producing `inf` coordinate
           values while the result falsely declared the transformer's
           own target CRS as a legitimate georeferencing.
        2. **The artifact's current CRS must match
           `transformer.source_crs`.** A mismatch (e.g. the artifact
           is genuinely `EPSG:3116` but `transformer` was built for
           `EPSG:4326 -> EPSG:32618`) would be mathematically valid
           but semantically wrong -- rejected explicitly rather than
           silently producing a plausible-looking, incorrect result.
        """
        if artifact_type not in _TRANSFORMABLE_ARTIFACT_TYPES:
            supported = ", ".join(t.value for t in _TRANSFORMABLE_ARTIFACT_TYPES)
            raise WorkflowStateError(
                f"transform_crs() does not support {artifact_type.value}; supported types are: {supported}."
            )

        _V.require(WorkflowStage.TRANSFORM_CRS, self._store, artifact_type)
        _V.require_current(WorkflowStage.TRANSFORM_CRS, self._store, self._history, artifact_type)
        current = self._store.get(artifact_type)

        # Confirmed directly during this stage's own safety review: a
        # SurveyPointSet/PointCloud with crs=None (genuinely unknown
        # origin CRS -- e.g. raw total-station coordinates with no
        # .prj) previously transformed silently, producing infinite
        # (`inf`) coordinate values while the RESULT falsely declared
        # the transformer's own target CRS as if it were a legitimate
        # georeferencing. FEATURE_COLLECTION now has its own `.crs`
        # too (added after this check was first written -- see
        # FeatureCollection's own docstring), so this protection now
        # extends to it as well; the earlier documented gap ("no
        # `.crs` concept at all to check") is closed.
        if artifact_type in (
            ArtifactType.POINT_CLOUD,
            ArtifactType.GROUND_CLOUD,
            ArtifactType.SURVEY_POINT_SET,
            ArtifactType.FEATURE_COLLECTION,
        ):
            current_crs = current.crs

            if current_crs is None:
                raise WorkflowStateError(
                    f"transform_crs() cannot transform {artifact_type.value}: its current CRS is "
                    "unknown (crs=None). Transforming from an unknown origin CRS would silently "
                    "produce meaningless coordinates. Assign a known CRS first -- detected "
                    "automatically, declared explicitly, or established via control-point "
                    "georeferencing -- before transforming."
                )

            source_crs = transformer.source_crs
            if artifact_type is ArtifactType.SURVEY_POINT_SET:
                matches = current_crs == source_crs
            else:
                # PointCloud.crs and FeatureCollection.crs are both
                # str (the "EPSG:{code}" or .name convention
                # established by transform_point_cloud() itself), not
                # a CRS object -- converted here to the same string
                # shape for a like-for-like comparison.
                expected = f"EPSG:{source_crs.epsg}" if source_crs.epsg is not None else source_crs.name
                matches = current_crs == expected

            if not matches:
                raise WorkflowStateError(
                    f"transform_crs() refused: {artifact_type.value}'s current CRS ({current_crs!r}) "
                    f"does not match the transformer's own source CRS ({source_crs!r}). This would be "
                    "mathematically valid but semantically wrong -- build a CoordinateTransformer "
                    "whose source_crs genuinely matches this artifact's current CRS."
                )

        dep = self._dependency(artifact_type, required=True)

        def work() -> Any:
            from topocore.geodesy.transform import (
                transform_feature_collection,
                transform_point_cloud,
                transform_survey,
            )

            if artifact_type in (ArtifactType.POINT_CLOUD, ArtifactType.GROUND_CLOUD):
                return transform_point_cloud(current, transformer)
            if artifact_type is ArtifactType.SURVEY_POINT_SET:
                return transform_survey(current, transformer)
            return transform_feature_collection(current, transformer)

        self._execute_stage(
            WorkflowStage.TRANSFORM_CRS,
            work,
            dependencies=(dep,),
            produces=artifact_type,
            metrics_fn=lambda result: StageMetrics(input_count=len(current), output_count=len(result)),
        )
        return self

    def transform_vertical(
        self,
        artifact_type: ArtifactType,
        *,
        source_datum: Any,
        target_datum: Any,
        geoid: Any,
    ) -> Workflow:
        """
        Requires
        --------
        `artifact_type` -- same 4 supported types as `transform_crs()`
        (`POINT_CLOUD`, `GROUND_CLOUD`, `SURVEY_POINT_SET`,
        `FEATURE_COLLECTION`); any other type is rejected explicitly
        with `WorkflowStateError`.

        Produces
        --------
        `artifact_type` (re-produced in place -- same pattern as
        `TRANSFORM_CRS`/`RESOLVE_SIDES`).

        Shifts `Z` via `topocore.geodesy.vertical.VerticalTransformer`
        (constructed internally from `source_datum`/`target_datum`/
        `geoid`, matching this method's own agreed signature -- unlike
        `transform_crs()`, which receives an already-built
        `CoordinateTransformer` directly). Dispatches to whichever of
        `transform_point_cloud_vertical()`/`transform_survey_vertical()`/
        `transform_feature_collection_vertical()` matches, exactly
        mirroring `transform_crs()`'s own dispatch.

        `X`/`Y` (or longitude/latitude, for `SurveyPointSet`) must
        already be geographic degrees in the same CRS `geoid` itself
        is defined in -- this stage does not reproject and does not
        validate this, matching
        `topocore.geodesy.vertical.transform`'s own stated caller
        responsibility. If your data is in a projected CRS, run
        `transform_crs()` to a geographic CRS first.

        Never leaves `Z` uncorrected. If any point/vertex in the
        artifact falls outside `geoid`'s own extent (or on a nodata
        cell), the entire stage fails with `WorkflowExecutionError`
        wrapping `MissingGeoidGridError` -- not a partially-corrected
        result, and never a silently-unchanged height. This directly
        closes the exact danger confirmed during this feature's own
        design review: plain `pyproj.Transformer` was confirmed to
        silently return an unchanged height when its own required
        vertical grid was missing, rather than raising.
        """
        if artifact_type not in _TRANSFORMABLE_ARTIFACT_TYPES:
            supported = ", ".join(t.value for t in _TRANSFORMABLE_ARTIFACT_TYPES)
            raise WorkflowStateError(
                f"transform_vertical() does not support {artifact_type.value}; supported types are: {supported}."
            )

        _V.require(WorkflowStage.TRANSFORM_VERTICAL, self._store, artifact_type)
        _V.require_current(WorkflowStage.TRANSFORM_VERTICAL, self._store, self._history, artifact_type)
        current = self._store.get(artifact_type)
        dep = self._dependency(artifact_type, required=True)

        def work() -> Any:
            from topocore.geodesy.vertical.transform import (
                transform_feature_collection_vertical,
                transform_point_cloud_vertical,
                transform_survey_vertical,
            )
            from topocore.geodesy.vertical.transformer import VerticalTransformer

            vertical_transformer = VerticalTransformer(
                source_datum=source_datum, target_datum=target_datum, geoid=geoid
            )

            if artifact_type in (ArtifactType.POINT_CLOUD, ArtifactType.GROUND_CLOUD):
                return transform_point_cloud_vertical(current, vertical_transformer)
            if artifact_type is ArtifactType.SURVEY_POINT_SET:
                return transform_survey_vertical(current, vertical_transformer)
            return transform_feature_collection_vertical(current, vertical_transformer)

        self._execute_stage(
            WorkflowStage.TRANSFORM_VERTICAL,
            work,
            dependencies=(dep,),
            produces=artifact_type,
            metrics_fn=lambda result: StageMetrics(input_count=len(current), output_count=len(result)),
        )
        return self

    def georeference(
        self,
        artifact_type: ArtifactType,
        controls: Any,
        target_crs: Any,
        options: Any,
    ) -> Workflow:
        """
        Requires
        --------
        `artifact_type` -- must currently be present, and must be one
        of `POINT_CLOUD`, `GROUND_CLOUD`, `SURVEY_POINT_SET`,
        `FEATURE_COLLECTION` -- the same 4 types
        `transform_crs()`/`transform_vertical()` support, for the
        same reason: `GROUND_CLOUD` reuses
        `apply_georeferencing_to_point_cloud()`, since both are the
        exact same `PointCloud` type (confirmed directly, that
        function has no special handling that would distinguish a
        "full" cloud from a ground-only one -- identical to
        `transform_point_cloud()`'s own precedent).

        Produces
        --------
        `artifact_type` (re-produced in place, same pattern as
        `TRANSFORM_CRS`/`TRANSFORM_VERTICAL`).

        `controls: Sequence[ControlPoint]`, `target_crs: CRS`,
        `options: GeoreferencingOptions` -- see
        `topocore.geodesy.georeferencing` for all 3. This stage never
        constructs a `ControlPoint` or decides `target_crs`/`options`
        itself, matching `transform_crs()`'s own "you build the
        transformer yourself" design.

        Unlike `transform_crs()`, this stage does NOT require
        `artifact_type`'s own current CRS to be known or to match
        anything -- control-point georeferencing exists specifically
        for local/arbitrary coordinates with no CRS at all (`crs=None`
        is the expected, common starting point here, never rejected).

        It DOES require the opposite: `artifact_type`'s current CRS
        must genuinely be `None`. Confirmed directly a real, serious
        gap without this check: calling `georeference()` a second
        time on an already-georeferenced (or otherwise already-CRS-
        assigned) artifact would silently re-apply a new set of
        control points -- computed for the ORIGINAL local
        coordinates -- on top of the already-transformed result,
        producing a meaningless, double-transformed artifact with no
        error at all. Rejected explicitly instead, mirroring
        `transform_crs()`'s own "never silently re-transform an
        already-referenced artifact" principle (there enforced via a
        source-CRS match; here via requiring genuine `crs=None`).

        `fit_georeferencing(controls, options)` is called BEFORE
        `_execute_stage()`, exactly mirroring `transform_crs()`'s own
        pre-validation pattern (its 2-check CRS safety review also
        runs before `_execute_stage()`) -- so
        `InsufficientControlPointsError`/`DegenerateGeometryError`/
        `LinearizationInvalidError`/`UnderconstrainedGeoreferencingError`/
        `ControlPointsTooCloseError` all propagate directly, never
        wrapped as a recorded, failed stage.

        `apply_scale_to_z` is decided from `result.strategy` alone --
        `True` only for `HELMERT_3D`, `False` for both `HELMERT_2D`
        and `TRANSLATION_ONLY` -- never inferred from
        `result.parameters` itself (e.g. checking `rx == 0 and ry ==
        0` would be exactly the kind of dangerous inference this
        whole capability's own design has consistently rejected:
        `HelmertParameters` cannot and does not encode which strategy
        produced it -- only `GeoreferencingResult.strategy` does).

        The full `GeoreferencingResult` (strategy, parameters,
        residuals, rms, control_count, warning) is recorded in this
        stage's own `StageMetrics.extra["georeferencing_result"]` --
        not flattened into separate `extra` keys, so no information
        is lost and every field remains available to a caller
        (e.g. a future DXF/GPKG metadata export) without needing a
        second computation.
        """
        if artifact_type not in _TRANSFORMABLE_ARTIFACT_TYPES:
            supported = ", ".join(t.value for t in _TRANSFORMABLE_ARTIFACT_TYPES)
            raise WorkflowStateError(
                f"georeference() does not support {artifact_type.value}; supported types are: {supported}."
            )

        _V.require(WorkflowStage.GEOREFERENCE, self._store, artifact_type)
        _V.require_current(WorkflowStage.GEOREFERENCE, self._store, self._history, artifact_type)
        current = self._store.get(artifact_type)

        # Confirmed directly, a real and serious gap: without this
        # check, calling georeference() a second time on an artifact
        # that a PRIOR georeference() (or transform_crs(), or a real
        # detected/declared CRS) already gave a real CRS to silently
        # re-applies a NEW set of control points -- computed for the
        # ORIGINAL local coordinates -- on top of the ALREADY-
        # transformed result, producing a meaningless,
        # double-transformed artifact with no error at all. This
        # mirrors exactly the same "never silently re-transform an
        # already-referenced artifact" danger `transform_crs()`'s own
        # safety review already established (there, guarded by
        # requiring crs to MATCH transformer.source_crs; here,
        # guarded by requiring crs to genuinely be None, since
        # control-point georeferencing exists specifically for
        # coordinates with no CRS at all).
        if current.crs is not None:
            raise WorkflowStateError(
                f"georeference() refused: {artifact_type.value}'s current CRS is already known "
                f"({current.crs!r}), not None. Control-point georeferencing is for genuinely "
                "local/unreferenced coordinates -- running it again on an artifact that already "
                "has a real CRS would silently re-apply a new transformation on top of an "
                "already-transformed result. If you need a further CRS-to-CRS transformation, use "
                "transform_crs() instead."
            )

        dep = self._dependency(artifact_type, required=True)

        from topocore.geodesy.georeferencing import (
            GeoreferencingStrategy,
            fit_georeferencing,
        )

        result = fit_georeferencing(controls, options)
        apply_scale_to_z = result.strategy is GeoreferencingStrategy.HELMERT_3D

        def work() -> Any:
            from topocore.geodesy.georeferencing.apply import (
                apply_georeferencing_to_feature_collection,
                apply_georeferencing_to_point_cloud,
                apply_georeferencing_to_survey,
            )

            if artifact_type in (ArtifactType.POINT_CLOUD, ArtifactType.GROUND_CLOUD):
                return apply_georeferencing_to_point_cloud(
                    current,
                    result.parameters,
                    target_crs,
                    apply_scale_to_z=apply_scale_to_z,
                )
            if artifact_type is ArtifactType.SURVEY_POINT_SET:
                return apply_georeferencing_to_survey(
                    current,
                    result.parameters,
                    target_crs,
                    apply_scale_to_z=apply_scale_to_z,
                )
            return apply_georeferencing_to_feature_collection(
                current,
                result.parameters,
                target_crs,
                apply_scale_to_z=apply_scale_to_z,
            )

        self._execute_stage(
            WorkflowStage.GEOREFERENCE,
            work,
            dependencies=(dep,),
            produces=artifact_type,
            metrics_fn=lambda new_artifact: StageMetrics(
                input_count=len(current),
                output_count=len(new_artifact),
                extra={"georeferencing_result": result},
            ),
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
        number of times, including alongside `export_gpkg()` on the
        same FeatureCollection.
        """
        _V.require(WorkflowStage.EXPORT_DXF, self._store, ArtifactType.FEATURE_COLLECTION)
        _V.require_current(
            WorkflowStage.EXPORT_DXF,
            self._store,
            self._history,
            ArtifactType.FEATURE_COLLECTION,
        )
        features = self._store.get(ArtifactType.FEATURE_COLLECTION)
        dep = self._dependency(ArtifactType.FEATURE_COLLECTION, required=True)

        def work() -> Any:
            from topocore.dxf import DXFExporter
            from topocore.dxf.models import DXFExportOptions, ExportContext

            # Found and fixed in PR20 coverage phase: this previously
            # called DXFExporter(**exporter_kwargs) directly, but
            # DXFExporter.__init__ takes a single `context:
            # ExportContext | None` parameter -- it does NOT accept
            # arbitrary keyword arguments. Confirmed directly: calling
            # export_dxf(path, strict=False) crashed with
            # "DXFExporter.__init__() got an unexpected keyword
            # argument 'strict'" for ANY non-empty exporter_kwargs,
            # even though the method's own **exporter_kwargs: Any
            # signature advertises pass-through configuration.
            # export_gpkg() already gets this right (builds a
            # GPKGExportOptions from kwargs, then passes THAT single
            # object) -- mirrored here: build DXFExportOptions from
            # exporter_kwargs, wrap it in ExportContext, then pass
            # that.
            options = DXFExportOptions(**exporter_kwargs)
            # features.crs is already the exact string type
            # ExportContext.crs expects (both "EPSG:{code}" or a bare
            # .name) -- confirmed directly, no conversion needed.
            # None is passed through unchanged too: FeatureCollection
            # with no known CRS means the exported DXF simply carries
            # no "TopoCore CRS" header variable, never an invented one.
            context = ExportContext(crs=features.crs, options=options)
            return DXFExporter(context).export(features, path)

        self._execute_stage(WorkflowStage.EXPORT_DXF, work, dependencies=(dep,), produces=None)
        return self

    def export_gpkg(self, path: str | Path, *, epsg: int | None = None, **exporter_kwargs: Any) -> Workflow:
        """
        Requires
        --------
        ArtifactType.FEATURE_COLLECTION

        Never writes to the ArtifactStore -- may be called any
        number of times, including alongside `export_dxf()` on the
        same FeatureCollection.

        `epsg` is now optional (previously required in every call).
        Resolution policy, confirmed by direct testing for each case:

        1. `FeatureCollection.crs` gives a real EPSG code (starts
           with `"EPSG:"`) and `epsg` is not given -- that code is
           used automatically. TopoCore already knows it; the caller
           is not asked to repeat it.
        2. `FeatureCollection.crs` is `None`, or is a bare name with
           no EPSG code (e.g. a custom, unregistered CRS) -- `epsg`
           becomes required; `WorkflowStateError` if not given. Never
           invents one.
        3. Both are given and agree -- proceeds normally.
        4. Both are given and disagree -- `WorkflowStateError`,
           refusing to silently pick one over the other. This mirrors
           `transform_crs()`'s own "never silently resolve a CRS
           discrepancy" principle.
        """
        _V.require(WorkflowStage.EXPORT_GPKG, self._store, ArtifactType.FEATURE_COLLECTION)
        _V.require_current(
            WorkflowStage.EXPORT_GPKG,
            self._store,
            self._history,
            ArtifactType.FEATURE_COLLECTION,
        )
        features = self._store.get(ArtifactType.FEATURE_COLLECTION)
        dep = self._dependency(ArtifactType.FEATURE_COLLECTION, required=True)

        detected_epsg: int | None = None
        if features.crs is not None and features.crs.startswith("EPSG:"):
            suffix = features.crs.removeprefix("EPSG:")
            if suffix.isdigit():
                detected_epsg = int(suffix)

        if epsg is not None and detected_epsg is not None and epsg != detected_epsg:
            raise WorkflowStateError(
                f"export_gpkg() refused: FeatureCollection's own CRS is 'EPSG:{detected_epsg}', "
                f"but epsg={epsg} was explicitly provided and disagrees. TopoCore never silently "
                "picks one over the other -- pass the correct epsg, or transform_crs() the "
                "FeatureCollection to the CRS you actually want first."
            )

        resolved_epsg = epsg if epsg is not None else detected_epsg

        if resolved_epsg is None:
            raise WorkflowStateError(
                "export_gpkg() requires an explicit epsg: FeatureCollection has no known CRS "
                "(or its CRS has no EPSG code) to derive one from automatically."
            )

        def work() -> Any:
            from topocore.gpkg import GeoPackageExporter, GPKGExportOptions

            options = GPKGExportOptions(epsg=resolved_epsg, **exporter_kwargs)
            return GeoPackageExporter(options).export(features, path)

        self._execute_stage(WorkflowStage.EXPORT_GPKG, work, dependencies=(dep,), produces=None)
        return self


__all__ = ["Workflow"]
