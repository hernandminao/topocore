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
    "WorkflowError",
    "WorkflowExecutionError",
    "WorkflowStateError",
    "WorkflowValidationError",
]
