"""
topocore.workflow.progress
==============================

A pure, read-only progress-observation contract: `ProgressEvent`
(immutable) and `ProgressObserver` (a `Protocol`, so any object with
an `on_progress` method qualifies -- no base class to inherit from).

Neither type does any fault isolation. The `try/except` around every
call to `observer.on_progress(...)` lives entirely in `workflow.py`,
not here -- keeping this module a pure contract, the same discipline
already applied to `stages.py`.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from topocore.workflow.stages import WorkflowStage


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """
    One progress notification. Carries no capability to affect
    execution -- `Workflow` never inspects a return value from
    `ProgressObserver.on_progress`, so nothing here can cancel,
    modify, or replace anything.

    Parameters
    ----------
    stage
        Which stage this event is about.
    phase
        Stable, categorical sub-step identifier for stages with
        internal phases (e.g. BUILD_TIN going through
        "building_kdtree" -> "triangulating" -> "building_adjacency"
        -> "validating_topology"). Unlike `message`, this stays
        constant across repeated events within the same sub-step --
        useful for a CLI/GUI progress bar to key off without parsing
        free text. `None` for stages with no meaningful sub-phases.
    message
        Short, human-readable description of what's happening right
        now (e.g. "classifying ground points").
    current, total
        Optional progress counters (e.g. points processed so far /
        total points), for stages that can report fine-grained
        progress. Both `None` for stages that only report
        start/finish, with nothing meaningful in between.
    """

    stage: WorkflowStage
    message: str
    phase: str | None = None
    current: int | None = None
    total: int | None = None


@runtime_checkable
class ProgressObserver(Protocol):
    """
    Structural contract: any object with a matching `on_progress`
    method satisfies this, no inheritance required.
    """

    def on_progress(self, event: ProgressEvent) -> None: ...


__all__ = ["ProgressEvent", "ProgressObserver"]
