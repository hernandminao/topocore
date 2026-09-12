"""
topocore.quality.models -- PROPUESTA, Punto 2 (contrato acordado).

Reglas del contrato, decididas explícitamente (no asumidas):
- Ningún indicador implica un juicio "bueno/malo" -- solo reporta el
  valor calculado. No hay score agregado, no hay semáforo, no hay
  umbrales normativos en esta versión.
- Diagnósticos estructurados: code, severity, message -- pensando en
  una futura API JSON que necesite distinguir por código, no por texto.
- No se exige una única fuente de datos: cada sección se calcula de
  forma independiente y opcional.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    severity: DiagnosticSeverity
    message: str


@dataclass(frozen=True, slots=True)
class AccuracySection:
    """
    RMSE entre puntos de control conocidos y sus posiciones observadas
    (típicamente, el DTM/TIN de TopoCore interpolado en esas mismas
    XY). Reporta el valor calculado únicamente -- no afirma si ese
    RMSE es aceptable para ningún proyecto.
    """

    rmse_horizontal: float
    rmse_vertical: float
    rmse_total: float
    control_point_count: int
    units: str = "m"


@dataclass(frozen=True, slots=True)
class DensitySection:
    """Densidad de puntos -- indicador de adquisición, no de exactitud."""

    mean: float
    minimum: float
    maximum: float
    resolution: float
    units: str = "points/m2"


@dataclass(frozen=True, slots=True)
class CoverageSection:
    """
    Cobertura espacial según una grilla regular. El resultado
    depende de cell_size/bounds/min_points_per_cell -- no es una
    medida absoluta.
    """

    total_cells: int
    occupied_cells: int
    coverage_percentage: float
    cell_size: float
    min_points_per_cell: int


@dataclass(frozen=True, slots=True)
class ClassificationSection:
    """
    Porcentaje de puntos con un código de clasificación dado. El
    código es opcional -- si no se suministra, se reporta el
    conteo total disponible sin calcular un porcentaje específico
    (ground_percentage queda en None, no en 0.0).
    """

    total_points: int
    code_provided: bool
    matched_code: int | None = None
    matched_points: int | None = None
    ground_percentage: float | None = None


@dataclass(frozen=True, slots=True)
class SpatialReference:
    """CRS -- registrado si está disponible, nunca validado en esta versión."""

    crs: str | None = None


@dataclass(frozen=True, slots=True)
class QualityResult:
    project_name: str
    accuracy: AccuracySection | None = None
    density: DensitySection | None = None
    coverage: CoverageSection | None = None
    classification: ClassificationSection | None = None
    spatial_reference: SpatialReference | None = None
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
