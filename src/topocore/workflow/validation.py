"""
topocore.workflow.validation
================================

`WorkflowValidator`: the only place that inspects an `ArtifactStore`
to decide whether a stage is allowed to run. Every stage method in
`workflow.py` delegates its precondition checks here instead of
inlining `store.has(...)` calls -- keeping `workflow.py` itself a
thin orchestrator.

Deliberately narrow scope
--------------------------
- Only ever checks artifact *presence* (`ArtifactStore.has`) --
  never versions (`ArtifactStore.version_of`). Staleness checks
  belong to PR19, once `ArtifactDependency` has real consumers; this
  file is kept clean so that addition is trivial later, not made now.
- Never validates a stage's own parameters (`cell_size`, `interval`,
  `epsg`, `path`, ...) -- that stays local to each stage method.
  `WorkflowValidator` only ever answers questions about the
  `ArtifactStore`'s current contents.
- Stateless: every method takes the `ArtifactStore` to check as an
  explicit parameter. Never held as an instance reference, never
  cached -- the store's contents can change between two calls (e.g.
  across a `snapshot()` branch), so nothing here should assume
  otherwise.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.workflow.artifacts import ArtifactStore, ArtifactType
from topocore.workflow.exceptions import WorkflowStateError
from topocore.workflow.stages import WorkflowStage


class WorkflowValidator:
    """Stateless -- see module docstring. All methods are staticmethods; there is nothing to instantiate."""

    @staticmethod
    def require(stage: WorkflowStage, store: ArtifactStore, artifact: ArtifactType) -> None:
        """
        Raises
        ------
        WorkflowStateError
            If `artifact` isn't present in `store`.
        """
        if not store.has(artifact):
            raise WorkflowStateError(f"Stage {stage.value} requires artifact {artifact.value}.")

    @staticmethod
    def require_all(stage: WorkflowStage, store: ArtifactStore, *artifacts: ArtifactType) -> None:
        """
        Raises
        ------
        WorkflowStateError
            If any of `artifacts` is missing from `store`. Reports
            every missing artifact together, not just the first.
        """
        missing = [artifact for artifact in artifacts if not store.has(artifact)]
        if missing:
            names = ", ".join(artifact.value for artifact in missing)
            raise WorkflowStateError(f"Stage {stage.value} requires artifacts ({names}).")

    @staticmethod
    def require_any(stage: WorkflowStage, store: ArtifactStore, *artifacts: ArtifactType) -> None:
        """
        Raises
        ------
        WorkflowStateError
            If none of `artifacts` is present in `store`.
        """
        if not any(store.has(artifact) for artifact in artifacts):
            names = ", ".join(artifact.value for artifact in artifacts)
            raise WorkflowStateError(f"Stage {stage.value} requires at least one of ({names}).")

    @staticmethod
    def require_absent(stage: WorkflowStage, store: ArtifactStore, *artifacts: ArtifactType) -> None:
        """
        Raises
        ------
        WorkflowStateError
            If any of `artifacts` is already present in `store`.
            Used to enforce "a Workflow represents one execution" --
            e.g. READ_SURVEY and READ_POINT_CLOUD can never both run
            on the same Workflow.
        """
        present = [artifact for artifact in artifacts if store.has(artifact)]
        if present:
            names = ", ".join(artifact.value for artifact in present)
            raise WorkflowStateError(f"Stage {stage.value} requires artifacts to be absent ({names}).")


__all__ = ["WorkflowValidator"]
