"""
topocore.workflow
=====================

`Workflow`: the orchestrator for the whole engine -- Survey/LAS/LAZ
ingestion, terrain construction, feature extraction, and export --
built on top of the modules already frozen elsewhere in TopoCore
(`survey`, `io`, `processing`, `terrain`, `features`, `dxf`, `gpkg`).

Public surface is deliberately narrow: `ArtifactStore` and
`WorkflowValidator` are infrastructure `Workflow` uses internally,
never something a caller touches directly -- the same "orchestrator
exposes behavior, not machinery" principle applied throughout this
package's design (see the Fase 1/Fase 2 architectural audits).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.workflow.artifacts import ArtifactType
from topocore.workflow.exceptions import (
    WorkflowError,
    WorkflowExecutionError,
    WorkflowStateError,
    WorkflowValidationError,
)
from topocore.workflow.history import (
    ArtifactDependency,
    StageMetrics,
    StageResult,
    WorkflowResult,
)
from topocore.workflow.progress import ProgressEvent, ProgressObserver
from topocore.workflow.stages import StageStatus, WorkflowStage
from topocore.workflow.workflow import Workflow

__all__ = [
    "ArtifactDependency",
    "ArtifactType",
    "ProgressEvent",
    "ProgressObserver",
    "StageMetrics",
    "StageResult",
    "StageStatus",
    "Workflow",
    "WorkflowError",
    "WorkflowExecutionError",
    "WorkflowResult",
    "WorkflowStage",
    "WorkflowStateError",
    "WorkflowValidationError",
]
