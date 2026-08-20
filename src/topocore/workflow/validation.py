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
- Checks artifact *presence* (`ArtifactStore.has`) via `require*`,
  and *freshness* (transitively, via `WorkflowResult` history) via
  `require_current` -- added in PR19, now that `ArtifactDependency`
  has a real consumer. See `require_current`/`_is_stale`'s own
  docstrings for why the freshness check must walk the history
  transitively, not just compare one recorded version number.
- Never validates a stage's own parameters (`cell_size`, `interval`,
  `epsg`, `path`, ...) -- that stays local to each stage method.
  `WorkflowValidator` only ever answers questions about the
  `ArtifactStore`'s (and, for `require_current`, the history's)
  current contents.
- Stateless: every method takes the `ArtifactStore` (and, where
  needed, the history) to check as an explicit parameter. Never held
  as an instance reference, never cached -- the store's contents can
  change between two calls (e.g. across a `snapshot()` branch), so
  nothing here should assume otherwise.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Sequence

from topocore.workflow.artifacts import ArtifactStore, ArtifactType
from topocore.workflow.exceptions import StaleArtifactError, WorkflowStateError

# Imported for typing only -- avoids a circular import at module load
# time (topocore.workflow.history doesn't import validation.py, but
# importing StageResult directly here isn't actually circular; kept
# as a plain import for clarity since Sequence[StageResult] is used
# in a public method signature, not just under TYPE_CHECKING).
from topocore.workflow.history import StageResult
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

    @staticmethod
    def require_current(
        stage: WorkflowStage,
        store: ArtifactStore,
        history: Sequence[StageResult],
        artifact: ArtifactType,
    ) -> None:
        """
        Raises
        ------
        StaleArtifactError
            If `artifact` is present but was built (directly or
            transitively) from an input that has since been
            superseded -- e.g. `TIN` still at v1 after `POINT_CLOUD`
            was re-read to v2, even though `TIN` was never itself
            rebuilt (its own version number alone can't reveal
            this -- see `_is_stale`'s docstring).

            Does nothing if `artifact` isn't present at all -- that
            is `require`'s job, not this method's; call both
            together (`require` first) to cover presence and
            freshness.
        """
        if store.has(artifact) and WorkflowValidator._is_stale(store, history, artifact):
            raise StaleArtifactError(
                f"Stage {stage.value} requires a current {artifact.value}, but the one "
                "available was built from since-superseded input(s). Rebuild it (and "
                "everything downstream of the changed input) before continuing."
            )

    @staticmethod
    def _is_stale(
        store: ArtifactStore,
        history: Sequence[StageResult],
        artifact: ArtifactType,
    ) -> bool:
        """
        Whether `artifact`'s CURRENT value in `store` was built from
        input(s) that no longer match what `store` currently holds
        for them -- checked transitively, not just one hop back.

        Why transitive, not just a direct version comparison
        ------------------------------------------------------
        `ArtifactStore` only ever holds one CURRENT version per
        `ArtifactType` (see that module's own docstring: it's
        deliberately not a dependency graph). Comparing only
        `artifact`'s own immediate recorded dependency version
        against the store's current version is not enough: an
        artifact that was never itself rebuilt keeps its own
        version number even after one of ITS ancestors changed.

        Concretely: `POINT_CLOUD v1 -> TIN v1 -> DTM v1`, then
        `POINT_CLOUD` is re-read (now v2) without rebuilding `TIN`
        or `DTM`. `DTM`'s own recorded dependency (`TIN v1`) still
        matches `TIN`'s current version (still v1, since `TIN` was
        never rebuilt) -- a one-hop check would wrongly call `DTM`
        fresh. Only by recursing into `TIN`'s own staleness (whose
        recorded dependency, `POINT_CLOUD v1`, no longer matches the
        current `POINT_CLOUD v2`) does the check correctly find that
        `DTM` is stale too.

        How
        ---
        Walks `history` to find the `StageResult` whose `produced`
        matches `artifact` at its current version (there is at most
        one, since `ArtifactStore.set()` version numbers are
        monotonic and never reused within one Workflow/branch), then
        recurses into each of that stage's own `dependencies`.
        """
        current_version = store.version_of(artifact)

        if current_version is None:
            return False  # not present -- a presence question, not a staleness one

        producing_stage = next(
            (
                stage_result
                for stage_result in reversed(history)
                if stage_result.produced is not None
                and stage_result.produced.artifact == artifact
                and stage_result.produced.version == current_version
            ),
            None,
        )

        if producing_stage is None:
            # No matching production record (e.g. a hand-constructed
            # store in a test) -- nothing to compare against, so
            # there is no basis to call it stale.
            return False

        for dependency in producing_stage.dependencies:
            if store.version_of(dependency.artifact) != dependency.version:
                return True
            if WorkflowValidator._is_stale(store, history, dependency.artifact):
                return True

        return False


__all__ = ["WorkflowValidator"]
