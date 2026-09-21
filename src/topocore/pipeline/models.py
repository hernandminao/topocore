"""
topocore.pipeline.models -- PROPUESTA, no auditada todavia con la
disciplina completa de PR22.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from topocore.pipeline.exceptions import SurveyPipelineError
from topocore.survey.formats import SurveyFormat

if TYPE_CHECKING:
    from topocore.features.models import FeatureCollection
    from topocore.quality.models import QualityResult
    from topocore.terrain.dtm import DTM
    from topocore.terrain.models import ContourLine
    from topocore.terrain.tin import TIN


class SurveyType(StrEnum):
    """
    Que tipo de levantamiento es -- determina si se requiere un
    reference_code (ROAD) o no, y que puerta de validacion de
    limite aplica.
    """

    ROAD = "road"
    PROPERTY = "property"
    GENERIC = "generic"


class BoundaryStrategy(StrEnum):
    """
    Como se resuelve el limite/perimetro cerrado, explicito -- el
    pipeline nunca infiere silenciosamente cual quiso decir el
    usuario.
    """

    CODE = "code"
    """El survey ya trae los codigos con sintaxis grammar
    (BASE.FIGURA.S/E) -- no se aplica ningun clustering."""

    CLUSTER_HULL = "cluster_hull"
    """split_by_clustering() con ordering="hull" -- para un limite
    cerrado sin sintaxis grammar previa."""

    NONE = "none"
    """No hay limite que resolver (survey_type=GENERIC tipico, o un
    PROPERTY donde el limite no es parte de este analisis)."""


@dataclass(frozen=True, slots=True)
class SurveyPipelineConfig:
    """
    Contrato de configuracion completo para SurveyPipeline.run().

    Parameters
    ----------
    survey_type
        ROAD, PROPERTY, o GENERIC -- determina las puertas de
        validacion que aplican (ver modulo gates.py).
    survey_format
        Formato de columnas del CSV (ej. SurveyFormat.PNEZD).
    boundary_strategy
        Como se resuelve el limite/perimetro. NONE por defecto.
    reference_code
        Requerido si survey_type=ROAD -- el codigo del eje/centerline
        (ej. "EJE"). Su ausencia real en el survey es un
        MissingReferenceCodeError -- no se intenta adivinar otro eje.
    boundary_code
        Requerido si boundary_strategy != NONE -- el codigo del
        limite/cerca/lindero a resolver (ej. "CERCA").
    boundary_eps
        Solo aplica con boundary_strategy=CLUSTER_HULL -- pasado
        directo a ClusterSplitConfig.eps. Su valor correcto depende
        totalmente de la escala real del proyecto (confirmado con
        datos reales: ~1.0m para columnas de una estructura, ~1000m
        para mantener el perimetro completo de un predio en un solo
        cluster) -- no hay un valor universal seguro, se debe ajustar
        por proyecto.
    linear_codes / splittable_codes
        Solo para survey_type=ROAD -- los mismos parametros de
        MultilineSplitConfig (bordes/cercas relativos al eje).
    build_features
        Si construye Feature/FeatureCollection -- False para un
        pipeline que solo necesita terreno/curvas, sin el paso de
        catalogo de features en absoluto.
    build_terrain
        Si construye TIN/DTM/GeoTIFF/curvas de nivel. Confirmado
        necesario con datos reales: un levantamiento catastral puro
        (predio con estructuras, sin ningun disparo de terreno) puede
        tener 0 puntos GROUND -- exigir un TIN ahi seria un error
        real, no una condicion excepcional. Con build_terrain=False,
        el resultado trae tin=None, dtm=None, contours=() -- el resto
        del pipeline (features, DXF, GPKG) funciona igual.
    contour_interval
        Intervalo de curvas de nivel, en las unidades del CRS.
    dtm_resolution
        Resolucion de la grilla del DTM.
    crs_epsg
        EPSG real del levantamiento, o None para coordenadas locales
        -- None es un valor VALIDO, nunca un error (confirmado con
        datos reales: un levantamiento con estacion total y origen
        asumido no tiene CRS real, y no se le debe inventar uno).
        Sin CRS, no se genera GeoTIFF (no tiene sentido sin una
        proyeccion real) -- el resto del pipeline funciona igual.
    catalog_extra
        Ruta a un catalogo JSON adicional (topocore.features.catalogs
        .loaders.json_loader) para codigos de campo no estandar de
        este proyecto especifico. Mecanismo de extension, nunca
        requisito para correr el pipeline.
    use_field_code_grammar
        Pasado directo a FeatureBuilder.
    generate_quality_report
        Si construye un QualityResult (densidad, cobertura de los
        puntos de terreno) al final del pipeline.
    output_dir / output_name
        Carpeta y nombre base para los archivos exportados.
    point_labels / contour_labels
        Pasado directo a DXFExportOptions.
    generate_cross_sections
        Solo aplica con survey_type=ROAD -- genera secciones
        transversales reales a lo largo del EJE (CSV + DXF, solo
        terreno -- ver topocore.analysis.profile.dxf_export). Cada
        estacion se genera de forma individual y aislada -- una
        estacion que se sale del TIN real no descarta las demas
        (confirmado necesario con datos reales).
    cross_section_width
        Ancho de cada seccion transversal (a cada lado del eje).
        Confirmado con datos reales: el valor correcto depende
        totalmente de la densidad de puntos de terreno disponible
        -- no hay un valor universal seguro.
    cross_section_interval
        Cada cuantos metros se genera una seccion a lo largo del eje
        real -- interpolando posicion y direccion cuando la estacion
        no cae en un vertice real del eje (no una seccion por cada
        vertice, que producia estaciones en distancias irregulares
        segun donde el topografo tomo cada punto). Siempre incluye la
        estacion final del eje, aunque no sea multiplo exacto.
    """

    survey_type: SurveyType
    survey_format: SurveyFormat
    boundary_strategy: BoundaryStrategy = BoundaryStrategy.NONE
    reference_code: str | None = None
    boundary_code: str | None = None
    boundary_eps: float = 1000.0
    linear_codes: frozenset[str] = field(default_factory=frozenset)
    splittable_codes: frozenset[str] = field(default_factory=frozenset)
    build_features: bool = True
    build_terrain: bool = True
    contour_interval: float = 0.5
    dtm_resolution: float = 1.0
    crs_epsg: int | None = None
    catalog_extra: Path | None = None
    use_field_code_grammar: bool = True
    generate_quality_report: bool = True
    generate_cross_sections: bool = False
    cross_section_width: float = 5.0
    cross_section_interval: float = 20.0
    output_dir: Path = Path(".")
    output_name: str = "salida"
    point_labels: bool = False
    contour_labels: bool = True

    def __post_init__(self) -> None:
        if self.survey_type == SurveyType.ROAD and not self.reference_code:
            raise SurveyPipelineError(
                "survey_type=ROAD requires a non-empty reference_code (e.g. 'EJE'). "
                "TopoCore does not guess a centerline on its own."
            )
        if self.boundary_strategy != BoundaryStrategy.NONE and not self.boundary_code:
            raise SurveyPipelineError(
                f"boundary_strategy={self.boundary_strategy.value} requires a non-empty boundary_code."
            )
        if self.boundary_eps <= 0:
            raise SurveyPipelineError("boundary_eps must be positive.")
        if self.cross_section_width <= 0:
            raise SurveyPipelineError("cross_section_width must be positive.")
        if self.cross_section_interval <= 0:
            raise SurveyPipelineError("cross_section_interval must be positive.")
        if self.contour_interval <= 0:
            raise SurveyPipelineError("contour_interval must be positive.")
        if self.dtm_resolution <= 0:
            raise SurveyPipelineError("dtm_resolution must be positive.")


@dataclass(slots=True)
class SurveyPipelineResult:
    """Salida completa de SurveyPipeline.run() -- cada campo es None
    si el paso correspondiente no aplico o no se pidio."""

    points_read: int
    ground_count: int
    unmatched_codes: frozenset[str]
    boundary_area: float | None
    tin: TIN | None
    dtm: DTM | None
    contours: tuple[ContourLine, ...]
    features: FeatureCollection | None
    dxf_path: Path | None
    gpkg_path: Path | None
    geotiff_path: Path | None
    quality_report: QualityResult | None
    quality_report_path: Path | None
    quality_report_pdf_path: Path | None
    cross_sections_csv_path: Path | None
    cross_sections_dxf_path: Path | None


__all__ = [
    "BoundaryStrategy",
    "SurveyPipelineConfig",
    "SurveyPipelineResult",
    "SurveyType",
]
