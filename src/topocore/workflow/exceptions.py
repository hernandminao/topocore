"""
topocore.workflow.exceptions
================================

Same pattern as `dxf.exceptions`/`gpkg.exceptions`: one base
exception plus a small number of specific subclasses, so callers can
catch `WorkflowError` broadly without needing to know about every
underlying module's own exception hierarchy (`GroundError`,
`TerrainError`, `DetectionError`, `DXFError`, `GPKGError`, ...).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.core.exceptions import TopoCoreError


class WorkflowError(TopoCoreError):
    """Base exception for all Workflow errors."""


class WorkflowStateError(WorkflowError):
    """
    Raised when a stage is called without one of its required
    artifacts available in the `ArtifactStore` -- e.g. calling
    `build_tin()` before `classify_ground()` has produced
    `GROUND_CLOUD`. A precondition failure about pipeline progress,
    not about the parameters given to the call itself.
    """


class StaleArtifactError(WorkflowStateError):
    """
    Raised when a stage's required artifact -- or any artifact it
    was transitively built from -- no longer matches the version
    that was actually consumed to produce it.

    A `WorkflowStateError` subtype, not `WorkflowValidationError`:
    this is a precondition failure about pipeline progress/
    consistency (the same category as a missing artifact), never
    about a stage call's own parameters -- see
    `WorkflowValidationError`'s own docstring, which explicitly
    reserves that class for parameter validation, "never used for
    missing artifacts."

    Example
    -------
    ``POINT_CLOUD v1 -> TIN v1 -> DTM v1``, then ``POINT_CLOUD`` is
    re-read (now v2) without rebuilding ``TIN``/``DTM``. ``TIN v1``
    (and anything built from it) no longer reflects the current
    ``POINT_CLOUD`` -- calling `build_dtm()` again (or any stage
    consuming `TIN`) raises this, even though `TIN`'s own version
    number (v1) hasn't changed, because the check walks back
    transitively through `WorkflowResult`'s history, not just the
    artifact's immediate recorded dependency.
    """


class WorkflowValidationError(WorkflowError):
    """
    Raised when a stage's own parameters are invalid, independent of
    artifact availability -- e.g. `build_dtm(cell_size=-5)`. Never
    used for missing artifacts; see `WorkflowStateError` for that.
    """


class WorkflowExecutionError(WorkflowError):
    """
    Raised when the underlying module a stage delegates to
    (`GroundManager`, `TIN`, `FeatureExtractionManager`,
    `DXFExporter`, `GeoPackageExporter`, ...) itself fails. Always
    raised with the original exception attached via ``from`` (see
    `__cause__`) -- the specific failure is never silently discarded,
    only given a single, coherent type at the Workflow API surface.
    """


__all__ = [
    "StaleArtifactError",
    "WorkflowError",
    "WorkflowExecutionError",
    "WorkflowStateError",
    "WorkflowValidationError",
]
