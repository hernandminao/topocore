"""
topocore.workflow.history
=============================

The append-only execution log: `StageResult` (one stage execution)
and `WorkflowResult` (the whole run so far). Never modified after
creation, never has entries removed -- this is what makes it safe
for PR19 (QA) to inspect without racing against a live Workflow.

`ArtifactDependency` records exactly which artifact versions a stage
consumed (required or optional), enabling objective "was this built
from a stale input?" checks later, without Workflow ever needing to
know which artifact types derive from which others (see
`workflow.artifacts` -- the store deliberately isn't a dependency
graph, and this module doesn't turn it into one either; it just
records facts).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from topocore.workflow.artifacts import ArtifactType
from topocore.workflow.stages import StageStatus, WorkflowStage


@dataclass(frozen=True, slots=True)
class ArtifactDependency:
    """
    One artifact version a stage consumed.

    Parameters
    ----------
    artifact
        Which artifact type.
    version
        The version of that artifact at the moment this stage read
        it (from `ArtifactStore.version_of`).
    required
        Whether the stage would have failed to run without this
        artifact (True), or used it only if available, like
        `DETECT_FEATURES` with `TIN`/`DTM`/`CLASSIFICATION_RESULT`
        (False). Recorded for every artifact actually consumed,
        required or optional -- never only the required ones, so a
        staleness check later isn't blind to optional inputs.
    """

    artifact: ArtifactType
    version: int
    required: bool


@dataclass(frozen=True, slots=True)
class StageMetrics:
    """
    Numeric facts about one stage execution.

    Parameters
    ----------
    input_count
        Size of the input, in whatever unit is natural for the stage
        (points, features, vertices, ...). `None` if not applicable
        or not measured.
    output_count
        Size of the output, same convention.
    extra
        Anything stage-specific that doesn't warrant its own named
        field (e.g. ``{"ground_points": ..., "non_ground_points": ...}``
        for CLASSIFY_GROUND). Same pattern as `FeatureMetadata.extra`.
    """

    input_count: int | None = None
    output_count: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StageResult:
    """
    The outcome of one stage execution. Immutable once created --
    appended to `WorkflowResult.stages`, never edited afterward.

    Parameters
    ----------
    stage
        Which stage ran.
    status
        See `StageStatus`.
    started_at, finished_at
        Wall-clock timestamps -- must be timezone-aware (UTC;
        ``datetime.now(UTC)``), never naive. `Workflow` runs are
        expected to cross process/deployment boundaries over time
        (snapshots, long LiDAR jobs, CI logs), where a naive
        timestamp is ambiguous. `duration` is derived from these,
        not stored separately (nothing here should disagree with
        itself).
    dependencies
        Every artifact version this stage actually consumed,
        required or optional (see `ArtifactDependency`). Empty for
        stages with no artifact inputs (e.g. READ_SURVEY, READ_POINT_CLOUD).
    produced
        The artifact this stage wrote to the `ArtifactStore`, and at
        what version -- `None` for export stages (EXPORT_DXF,
        EXPORT_GPKG never write to the store) and for any stage that
        didn't finish with `StageStatus.SUCCESS`.
    metrics
        See `StageMetrics`. `None` if nothing was measured.
    warnings
        Non-fatal issues noticed during the stage (e.g. skipped
        detectors in non-strict `DETECT_FEATURES`).
    error
        The exception raised, if `status is StageStatus.FAILED`.
        `None` otherwise.
    """

    stage: WorkflowStage
    status: StageStatus
    started_at: datetime
    finished_at: datetime
    dependencies: tuple[ArtifactDependency, ...] = ()
    produced: ArtifactDependency | None = None
    metrics: StageMetrics | None = None
    warnings: tuple[str, ...] = ()
    error: BaseException | None = None

    @property
    def duration(self) -> float:
        """Wall-clock seconds elapsed, derived from started_at/finished_at."""
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def succeeded(self) -> bool:
        """Shorthand for `self.status is StageStatus.SUCCESS` -- used constantly enough (PR18, PR19, tests) to be worth a name."""
        return self.status is StageStatus.SUCCESS


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """
    The full execution history of a Workflow so far. Append-only:
    a new `WorkflowResult` (via `Workflow`'s internal bookkeeping) is
    what grows as stages run -- existing `StageResult`s already
    appended are never rewritten or removed.

    Two minimal helpers are accepted (`succeeded` on `StageResult`,
    `last_stage` here) -- both are one-line derivations with no
    design ambiguity. Deliberately still no `.stage(name)` lookup,
    no `.timeline()`: those need a real usage pattern (PR19/PR20/PR21)
    to shape their API correctly, not a guess made now.
    """

    stages: tuple[StageResult, ...]
    started_at: datetime
    finished_at: datetime

    @property
    def duration(self) -> float:
        """Wall-clock seconds elapsed, derived from started_at/finished_at."""
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def last_stage(self) -> StageResult | None:
        """The most recently appended StageResult, or None if no stage has run yet."""
        return self.stages[-1] if self.stages else None


__all__ = ["ArtifactDependency", "StageMetrics", "StageResult", "WorkflowResult"]
