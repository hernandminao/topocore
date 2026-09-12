"""
topocore.quality.analyzer -- PROPUESTA, Punto 2 (contrato acordado).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from topocore.analysis.exceptions import QualityError, StatisticsError
from topocore.analysis.quality.rmse import RMSEAnalysis
from topocore.analysis.statistics.manager import StatisticsAnalysis
from topocore.quality.models import (
    AccuracySection,
    ClassificationSection,
    CoverageSection,
    DensitySection,
    Diagnostic,
    DiagnosticSeverity,
    QualityResult,
    SpatialReference,
)

_MIN_CONTROL_POINTS = 2  # acordado: menos de esto -> diagnóstico, no se reporta accuracy


class QualityAnalyzer:
    def __init__(self, project_name: str) -> None:
        self._project_name = project_name
        self._diagnostics: list[Diagnostic] = []

    def _diagnose(self, code: str, severity: DiagnosticSeverity, message: str) -> None:
        self._diagnostics.append(Diagnostic(code=code, severity=severity, message=message))

    def analyze_accuracy(
        self,
        reference: NDArray[np.float64],
        observed: NDArray[np.float64],
    ) -> AccuracySection | None:
        """
        `observed` es típicamente el DTM/TIN de TopoCore interpolado
        en las mismas XY que `reference` (ver end-to-end-examples.md
        -- TIN.interpolate(x, y) es la integración real esperada).
        No valida que reference/observed compartan CRS -- acordado,
        pendiente para una versión futura.
        """
        reference = np.asarray(reference, dtype=np.float64)
        n = reference.shape[0] if reference.ndim > 0 else 0

        if n < _MIN_CONTROL_POINTS:
            self._diagnose(
                "ACCURACY_INSUFFICIENT_CONTROL_POINTS",
                DiagnosticSeverity.WARNING,
                f"Only {n} control point(s) given; at least {_MIN_CONTROL_POINTS} are needed to report accuracy.",
            )
            return None

        try:
            resultado = RMSEAnalysis.compute(reference, observed)
        except QualityError as exc:
            self._diagnose("ACCURACY_COMPUTE_FAILED", DiagnosticSeverity.ERROR, str(exc))
            return None

        return AccuracySection(
            rmse_horizontal=resultado.horizontal,
            rmse_vertical=resultado.vertical,
            rmse_total=resultado.total,
            control_point_count=resultado.count,
        )

    def analyze_density(
        self,
        points_xy: NDArray[np.float64],
        *,
        resolution: float = 1.0,
        crs: str | None = None,
    ) -> DensitySection | None:
        """
        `crs`, si se da, solo se usa para advertir (no bloquear) si
        parece geográfico -- acordado: "puntos/m²" no tiene sentido
        real en grados.

        Valida NaN/Inf ANTES de llamar a StatisticsAnalysis.density():
        confirmado por ejecución real, ese componente (ya cerrado en
        PR22, no se toca aquí) deja escapar un OverflowError crudo
        para entradas con Inf, en vez de su propio StatisticsError --
        un defecto real y preexistente, descubierto durante el Punto 4
        de este módulo. Esta validación explícita es la corrección en
        la frontera del Quality Engine, no un parche al componente.
        """
        if points_xy.size == 0:
            self._diagnose("DENSITY_NO_POINTS", DiagnosticSeverity.ERROR, "No points given.")
            return None
        if not np.all(np.isfinite(points_xy)):
            self._diagnose(
                "DENSITY_NON_FINITE_POINTS",
                DiagnosticSeverity.ERROR,
                "Points contain NaN or infinite values.",
            )
            return None

        if crs is not None:
            try:
                from topocore.geodesy.crs import CRS
                from topocore.geodesy.exceptions import CRSError

                crs_obj = CRS.from_epsg(int(crs.split(":", 1)[1])) if crs.upper().startswith("EPSG:") else None
                if crs_obj is not None and crs_obj.is_geographic:
                    self._diagnose(
                        "DENSITY_GEOGRAPHIC_CRS",
                        DiagnosticSeverity.WARNING,
                        f"CRS '{crs}' is geographic; points/m2 is not meaningful in degrees. "
                        "Reporting the value anyway, uninterpreted.",
                    )
            except (CRSError, ValueError):
                # No bloquear el análisis de densidad por un CRS que
                # no se pudo interpretar en forma "EPSG:xxxx" -- se
                # sigue calculando la densidad de todas formas.
                pass

        try:
            stats = StatisticsAnalysis().density(points_xy, resolution=resolution)
        except StatisticsError as exc:
            self._diagnose("DENSITY_COMPUTE_FAILED", DiagnosticSeverity.ERROR, str(exc))
            return None

        return DensitySection(
            mean=stats.mean_density,
            minimum=stats.minimum_density,
            maximum=stats.maximum_density,
            resolution=resolution,
        )

    def analyze_coverage(
        self,
        points_xy: NDArray[np.float64],
        bounds: tuple[float, float, float, float],
        *,
        cell_size: float,
        min_points_per_cell: int = 1,
    ) -> CoverageSection | None:
        min_x, min_y, max_x, max_y = bounds

        if cell_size <= 0:
            self._diagnose("COVERAGE_INVALID_CELL_SIZE", DiagnosticSeverity.ERROR, "cell_size must be positive.")
            return None
        if max_x <= min_x or max_y <= min_y:
            self._diagnose(
                "COVERAGE_INVALID_BOUNDS", DiagnosticSeverity.ERROR, "bounds must have positive width and height."
            )
            return None
        if points_xy.size == 0:
            self._diagnose("COVERAGE_NO_POINTS", DiagnosticSeverity.ERROR, "No points given.")
            return None
        if not np.all(np.isfinite(points_xy)):
            self._diagnose(
                "COVERAGE_NON_FINITE_POINTS", DiagnosticSeverity.ERROR, "Points contain NaN or infinite values."
            )
            return None

        n_cols = max(1, int(np.ceil((max_x - min_x) / cell_size)))
        n_rows = max(1, int(np.ceil((max_y - min_y) / cell_size)))
        counts = np.zeros((n_rows, n_cols), dtype=np.int64)
        cols = np.clip(((points_xy[:, 0] - min_x) / cell_size).astype(int), 0, n_cols - 1)
        rows = np.clip(((points_xy[:, 1] - min_y) / cell_size).astype(int), 0, n_rows - 1)
        np.add.at(counts, (rows, cols), 1)

        occupied = int(np.sum(counts >= min_points_per_cell))
        total = n_rows * n_cols

        return CoverageSection(
            total_cells=total,
            occupied_cells=occupied,
            coverage_percentage=100.0 * occupied / total,
            cell_size=cell_size,
            min_points_per_cell=min_points_per_cell,
        )

    def analyze_classification(
        self,
        codes: NDArray[np.int64],
        *,
        matched_code: int | None = None,
    ) -> ClassificationSection | None:
        """
        `matched_code` es opcional -- acordado. Sin él, se reporta
        el total disponible sin calcular un porcentaje (evita
        asumir una convención como ASPRS por defecto).
        """
        if codes.size == 0:
            self._diagnose("CLASSIFICATION_NO_DATA", DiagnosticSeverity.ERROR, "No classification codes given.")
            return None

        total = int(codes.size)
        if matched_code is None:
            return ClassificationSection(total_points=total, code_provided=False)

        matched = int(np.sum(codes == matched_code))
        if matched == 0:
            self._diagnose(
                "CLASSIFICATION_CODE_NOT_FOUND",
                DiagnosticSeverity.WARNING,
                f"matched_code={matched_code} does not appear anywhere in the given codes.",
            )
        return ClassificationSection(
            total_points=total,
            code_provided=True,
            matched_code=matched_code,
            matched_points=matched,
            ground_percentage=100.0 * matched / total,
        )

    def build(
        self,
        *,
        accuracy: AccuracySection | None = None,
        density: DensitySection | None = None,
        coverage: CoverageSection | None = None,
        classification: ClassificationSection | None = None,
        crs: str | None = None,
    ) -> QualityResult:
        if crs is None:
            self._diagnose(
                "SPATIAL_REFERENCE_MISSING",
                DiagnosticSeverity.WARNING,
                "No CRS was provided for this report.",
            )
        return QualityResult(
            project_name=self._project_name,
            accuracy=accuracy,
            density=density,
            coverage=coverage,
            classification=classification,
            spatial_reference=SpatialReference(crs=crs),
            diagnostics=tuple(self._diagnostics),
        )
