"""
topocore.processing.normals.models
====================================

Result container for normal/curvature estimation.

Kept separate from `base.py` (which defines the estimator ABCs:
`NormalEstimator`, `CurvatureEstimator`,
`NormalAndCurvatureEstimator`) so `NormalResult` can be imported
by downstream consumers — such as `topocore.features` — without
pulling in the estimator class hierarchy.

`NormalAndCurvatureEstimator.estimate_both()` keeps returning its
existing `tuple[FloatArray2D, FloatArray1D]` — that stays untouched
to avoid a breaking API change. `NormalResult` is an adapter for
callers that want a named, validated container instead of a bare
tuple:

    normals, curvature = normal_manager.estimate_both(cloud)
    result = NormalResult(normals=normals, curvature=curvature)

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from topocore.processing.exceptions import ProcessingError
from topocore.processing.types import FloatArray1D, FloatArray2D


@dataclass(frozen=True, slots=True)
class NormalResult:
    """
    Result of normal and curvature estimation.

    Parameters
    ----------
    normals
        Per-point unit normal vectors, shape ``(n, 3)``.
    curvature
        Per-point surface variation / curvature estimate, shape
        ``(n,)``. ``None`` if the estimator that produced this
        result doesn't compute curvature.

    Raises
    ------
    ProcessingError
        If `normals` isn't shaped ``(n, 3)``, or if `curvature` is
        given but its length doesn't match `normals`.
    """

    normals: FloatArray2D
    curvature: FloatArray1D | None = None

    def __post_init__(self) -> None:
        if self.normals.ndim != 2:
            raise ProcessingError("Normals must be a 2D array.")

        if self.normals.shape[1] != 3:
            raise ProcessingError(f"Normals must have shape (n,3); got {self.normals.shape}.")

        if self.curvature is not None and self.curvature.shape[0] != self.normals.shape[0]:
            raise ProcessingError("Curvature size does not match normals.")

    @property
    def point_count(self) -> int:
        return int(self.normals.shape[0])

    @property
    def has_curvature(self) -> bool:
        return self.curvature is not None


__all__ = ["NormalResult"]
