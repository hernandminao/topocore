"""
topocore.analysis.quality.correctness
=====================================

Correctness analysis.

Evaluates the correctness of a terrain model against reference data.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from topocore.analysis.exceptions import QualityError
from topocore.analysis.types import CorrectnessResult


class CorrectnessAnalysis:
    """
    Evaluates model correctness.
    """

    __slots__ = ("_tolerance",)

    def __init__(
        self,
        tolerance: float = 0.05,
    ) -> None:
        if tolerance <= 0.0:
            raise QualityError("Tolerance must be positive.")

        self._tolerance = float(tolerance)

    @property
    def tolerance(self) -> float:
        return self._tolerance

    def compute(
        self,
        model_grid: NDArray[np.float64],
        reference_grid: NDArray[np.float64],
    ) -> CorrectnessResult:
        """
        Evaluate correctness.

        Parameters
        ----------
        model_grid
            Model elevation grid.

        reference_grid
            Reference elevation grid.

        Returns
        -------
        CorrectnessResult
        """
        model_grid = np.asarray(
            model_grid,
            dtype=np.float64,
        )

        reference_grid = np.asarray(
            reference_grid,
            dtype=np.float64,
        )

        if model_grid.ndim != 2:
            raise QualityError("Model grid must be 2D.")

        if reference_grid.ndim != 2:
            raise QualityError("Reference grid must be 2D.")

        if model_grid.shape != reference_grid.shape:
            raise QualityError("Grid shapes must match.")

        model_valid = np.isfinite(model_grid)
        reference_valid = np.isfinite(reference_grid)

        if not np.any(model_valid) and not np.any(reference_valid):
            raise QualityError("Both grids contain no valid data.")

        common = model_valid & reference_valid

        differences = np.abs(model_grid - reference_grid)

        true_positives = int(np.count_nonzero(common & (differences <= self._tolerance)))

        commission = int(np.count_nonzero(model_valid & ~reference_valid))

        disagreement = int(np.count_nonzero(common & (differences > self._tolerance)))

        false_positives = commission + disagreement

        false_negatives = int(np.count_nonzero(~model_valid & reference_valid))

        precision = (
            true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        )

        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0

        return CorrectnessResult(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1,
        )

    def __call__(
        self,
        model_grid: NDArray[np.float64],
        reference_grid: NDArray[np.float64],
    ) -> CorrectnessResult:
        """Alias for :meth:`compute`."""
        return self.compute(
            model_grid,
            reference_grid,
        )


__all__ = [
    "CorrectnessAnalysis",
]
