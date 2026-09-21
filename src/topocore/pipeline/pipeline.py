"""
topocore.pipeline.pipeline -- PROPUESTA, no auditada todavia con la
disciplina completa de PR22. Construida y verificada con 4
levantamientos reales de campo (2 vias, 1 predio con estructuras, 1
predio con datos locales/arbitrarios) durante la validacion de PR24.

Consolida en un unico flujo configurable todo lo construido y
validado por separado en esta sesion:

    levantamiento -> limite -> area -> TIN -> DTM -> curvas de
    nivel -> DXF/GPKG -> Quality Report

con un unico principio rector: el pipeline se DETIENE ante cualquier
condicion geometricamente ambigua o invalida, en vez de adivinar y
generar una salida equivocada (ver topocore.pipeline.gates y
topocore.pipeline.exceptions).

Author
------
Hernán Mina

License
-------
MIT
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from topocore.analysis.types import ProfileResult
from topocore.dxf import DXFExporter
from topocore.dxf.models import DXFExportOptions, ExportContext
from topocore.features.catalogs.loaders.json_loader import load_json
from topocore.features.feature_builder import FeatureBuilder
from topocore.features.feature_codes import FeatureCodeRegistry
from topocore.features.models import (
    Feature,
    FeatureCategory,
    FeatureCollection,
    FeatureGeometry,
    FeatureType,
    GeometryType,
)
from topocore.geodesy.crs import CRS
from topocore.geometry.point3d import Point3D
from topocore.gpkg import GeoPackageExporter, GPKGExportOptions
from topocore.pipeline.exceptions import (
    InsufficientDataError,
    MissingReferenceCodeError,
    PropertyBoundaryError,
    SurveyPipelineError,
)
from topocore.pipeline.gates import validate_property_boundary
from topocore.pipeline.models import BoundaryStrategy, SurveyPipelineConfig, SurveyPipelineResult, SurveyType
from topocore.quality.analyzer import QualityAnalyzer
from topocore.survey.multiline import (
    ClusterSplitConfig,
    MultilineSplitConfig,
    split_by_clustering,
    split_multiline_codes,
)
from topocore.survey.reader import SurveyTXTReader
from topocore.terrain.contours import ContourGenerator
from topocore.terrain.dtm import DTM
from topocore.terrain.enums import InterpolationMethod
from topocore.terrain.grid import Grid
from topocore.terrain.interpolation import TerrainInterpolator
from topocore.terrain.models import ContourLine
from topocore.terrain.tin import TIN

_MIN_GROUND_POINTS = 3


def _interpolar_estacion(
    eje_xy: list[tuple[float, float]], estacion_objetivo: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Encuentra el punto real sobre la polilinea del eje que corresponde
    a ``estacion_objetivo`` (distancia acumulada desde el inicio),
    interpolando dentro del segmento que corresponda -- no solo en
    los vertices reales del eje.

    Devuelve (punto, origen_direccion, destino_direccion): el punto
    interpolado, y los 2 extremos del segmento que lo contiene (para
    usar como direccion local en .transversal()).
    """
    import math

    acumulado = 0.0
    for i in range(len(eje_xy) - 1):
        a, b = eje_xy[i], eje_xy[i + 1]
        largo_segmento = math.hypot(b[0] - a[0], b[1] - a[1])
        es_ultimo_segmento = i == len(eje_xy) - 2

        if acumulado + largo_segmento >= estacion_objetivo or es_ultimo_segmento:
            t = (estacion_objetivo - acumulado) / largo_segmento if largo_segmento > 0 else 0.0
            t = max(0.0, min(1.0, t))
            punto = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            return punto, a, b

        acumulado += largo_segmento

    return eje_xy[-1], eje_xy[-2], eje_xy[-1]


def _generar_secciones_por_intervalo(
    eje_xy: list[tuple[float, float]], tin: TIN, *, interval: float, width: float
) -> list[ProfileResult]:
    """
    Genera una seccion transversal cada ``interval`` metros a lo largo
    del eje real (interpolando posicion y direccion cuando la
    estacion no coincide con un vertice real), en vez de una por cada
    vertice del eje. Incluye siempre la estacion final del eje, aunque
    no caiga en un multiplo exacto de ``interval``. Cada estacion se
    genera de forma individual y aislada -- una estacion fuera del
    TIN real no descarta las demas.
    """
    import math

    from topocore.analysis.exceptions import ProfileError
    from topocore.analysis.profile.manager import ProfileAnalysis

    longitud_total = sum(
        math.hypot(eje_xy[i + 1][0] - eje_xy[i][0], eje_xy[i + 1][1] - eje_xy[i][1]) for i in range(len(eje_xy) - 1)
    )

    estaciones = [i * interval for i in range(int(longitud_total // interval) + 1)]
    if not estaciones or estaciones[-1] < longitud_total:
        estaciones.append(longitud_total)

    analizador = ProfileAnalysis()
    secciones: list[ProfileResult] = []
    for estacion in estaciones:
        _, origen, destino = _interpolar_estacion(eje_xy, estacion)
        try:
            secciones.append(analizador.transversal(origen, destino, estacion, tin, width=width))
        except ProfileError:
            pass

    return secciones


class SurveyPipeline:
    """
    ``SurveyPipeline(config).run(ruta_csv)`` ejecuta el flujo
    completo. Ver ``SurveyPipelineConfig`` para el contrato de
    configuracion, y ``topocore.pipeline.exceptions`` para las
    condiciones concretas que detienen el pipeline en vez de generar
    una salida equivocada.
    """

    def __init__(self, config: SurveyPipelineConfig) -> None:
        self._config = config

    def run(self, survey_path: str | Path) -> SurveyPipelineResult:
        config = self._config

        # 1. Leer el levantamiento
        survey_points = SurveyTXTReader(survey_path, format=config.survey_format).read()
        codigos_presentes = {p.code for p in survey_points.points}

        # 2. Resolver el limite, segun survey_type -- puerta de
        #    validacion #1: ROAD sin reference_code real es un error
        #    duro, nunca una suposicion silenciosa.
        if config.survey_type == SurveyType.ROAD:
            if config.reference_code not in codigos_presentes:
                raise MissingReferenceCodeError(
                    f"survey_type=ROAD requires the code '{config.reference_code}' "
                    f"to be present in the survey; it was not found. "
                    f"Codes present: {sorted(c for c in codigos_presentes if c is not None)}."
                )
            assert config.reference_code is not None  # garantizado por SurveyPipelineConfig.__post_init__
            multiline_config = MultilineSplitConfig(
                linear_codes=config.linear_codes,
                splittable_codes=config.splittable_codes,
                reference_code=config.reference_code,
            )
            survey_points = split_multiline_codes(survey_points, multiline_config)

        elif config.survey_type == SurveyType.PROPERTY and config.boundary_strategy == BoundaryStrategy.CLUSTER_HULL:
            assert config.boundary_code is not None  # garantizado por SurveyPipelineConfig.__post_init__
            cluster_config = ClusterSplitConfig(
                linear_codes=frozenset({config.boundary_code}),
                eps=config.boundary_eps,
                ordering="hull",
            )
            survey_points = split_by_clustering(survey_points, cluster_config)

        # 3. CRS -- None es valido (coordenadas locales), nunca un error
        if config.crs_epsg is not None:
            survey_points = dataclasses.replace(survey_points, crs=CRS.from_epsg(config.crs_epsg))

        # 4. Catalogo: por defecto + extension del proyecto
        registry = FeatureCodeRegistry.default()
        if config.catalog_extra is not None:
            registry.register_many(load_json(config.catalog_extra))

        # 5. Construir features
        result = FeatureBuilder(registry, use_field_code_grammar=config.use_field_code_grammar).build(survey_points)

        # 6. Limite de PROPERTY -- puerta de validacion #2: puntos
        #    suficientes, sin autointersecciones, area > 0. El
        #    pipeline se detiene aqui si alguna falla (ver gates.py).
        boundary_area: float | None = None
        boundary_features_replacement: dict[int, Feature] = {}
        if config.survey_type == SurveyType.PROPERTY and config.boundary_strategy != BoundaryStrategy.NONE:
            assert config.boundary_code is not None  # garantizado por SurveyPipelineConfig.__post_init__
            boundary_definition = registry.get(config.boundary_code)
            if boundary_definition is None:
                raise SurveyPipelineError(
                    f"boundary_code='{config.boundary_code}' is not registered in the catalog "
                    f"(default catalog + catalog_extra, if given). Register it before running the pipeline."
                )
            # Se busca por el codigo de campo real (attributes["survey_code"],
            # que FeatureBuilder adjunta siempre) -- confirmado necesario:
            # el MISMO codigo de campo (ej. "CERCA") puede quedar registrado
            # con un feature_type distinto segun la version/entorno del
            # catalogo (ej. "fence" en un catalogo, "boundary" en otro) --
            # buscar por survey_code es independiente de esa variacion.
            boundary_feature = next(
                (f for f in result.features if f.attributes.get("survey_code") == config.boundary_code),
                None,
            )
            if boundary_feature is None:
                tipos_presentes = sorted({f.feature_type.value for f in result.features if f.feature_type is not None})
                codigos_presentes_features = sorted(
                    {c for f in result.features if (c := f.attributes.get("survey_code")) is not None}
                )
                raise PropertyBoundaryError(
                    f"No feature built from boundary_code='{config.boundary_code}' was found. "
                    f"{len(result.features)} feature(s) were built, with types: {tipos_presentes} "
                    f"and survey_code(s): {codigos_presentes_features}. "
                    f"Check that '{config.boundary_code}' points are actually present and reach "
                    f"FeatureBuilder correctly (verify the intermediate/preprocessed file if one is used)."
                )
            vertices = [(float(v[0]), float(v[1])) for v in boundary_feature.geometry.vertices]
            boundary_area = validate_property_boundary(vertices)

            if boundary_feature is not None:
                closed_geometry = dataclasses.replace(boundary_feature.geometry, closed=True)
                boundary_features_replacement[id(boundary_feature)] = dataclasses.replace(
                    boundary_feature, geometry=closed_geometry
                )

        features_final = [boundary_features_replacement.get(id(f), f) for f in result.features]

        # 7. Terreno -- opcional (build_terrain=False para un
        #    levantamiento sin ningun disparo de terreno, ej. un
        #    predio catastral puro con solo estructuras/linderos).
        tin: TIN | None = None
        dtm: DTM | None = None
        geotiff_path: Path | None = None
        contours: tuple[ContourLine, ...] = ()
        cross_sections_csv_path: Path | None = None
        cross_sections_dxf_path: Path | None = None

        if config.build_terrain:
            if len(result.ground) < _MIN_GROUND_POINTS:
                raise InsufficientDataError(
                    f"Only {len(result.ground)} GROUND point(s) available; "
                    f"at least {_MIN_GROUND_POINTS} are needed to build a TIN."
                )
            ground_points = tuple(Point3D(p.x, p.y, p.z) for p in result.ground)
            tin = TIN.from_points(ground_points)

            # 8. DTM (+ GeoTIFF, solo si hay CRS real)
            min_x, min_y, max_x, max_y = tin.bounds
            grid = Grid(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, resolution=config.dtm_resolution)
            interpolator = TerrainInterpolator(tin, method=InterpolationMethod.LINEAR)
            # DTM.from_tin() declara su parametro como BaseInterpolator (ABC),
            # pero TerrainInterpolator -- la clase que todo el proyecto usa en
            # la practica -- no hereda de esa base. Inconsistencia real y
            # PREEXISTENTE del modulo terrain (ya cerrado en PR22) -- no se
            # corrige aqui, solo se documenta y suprime puntualmente.
            dtm = DTM.from_tin(tin, grid, interpolator)  # type: ignore[arg-type]

            config.output_dir.mkdir(parents=True, exist_ok=True)
            if config.crs_epsg is not None:
                from topocore.io.raster.geotiff import GeoTIFFWriteError, write_dtm_geotiff

                ruta_intentada = config.output_dir / f"{config.output_name}_dtm.tif"
                try:
                    write_dtm_geotiff(dtm, ruta_intentada, crs_wkt=CRS.from_epsg(config.crs_epsg).to_wkt())
                    geotiff_path = ruta_intentada
                except GeoTIFFWriteError as exc:
                    # GDAL ausente (u otro fallo real de escritura) no
                    # detiene el resto del pipeline -- el GeoTIFF es una
                    # salida adicional, no un requisito para terreno/
                    # curvas/DXF/GPKG. geotiff_path queda en None,
                    # visible en el resultado para que quien llame lo note.
                    print(f"Aviso: GeoTIFF omitido -- {exc}")

            # 9. Curvas de nivel
            contours = ContourGenerator(tin).generate(interval=config.contour_interval, base=0.0)

            # 9b. Secciones transversales (solo ROAD, si se pide)
            if config.survey_type == SurveyType.ROAD and config.generate_cross_sections:
                eje_features = [f for f in features_final if f.attributes.get("survey_code") == config.reference_code]
                if eje_features:
                    eje_xy = [(float(v[0]), float(v[1])) for v in eje_features[0].geometry.vertices]
                    secciones = _generar_secciones_por_intervalo(
                        eje_xy, tin, interval=config.cross_section_interval, width=config.cross_section_width
                    )

                    if secciones:
                        from topocore.analysis.profile.writer import write_profile_csv

                        cross_sections_csv_path = config.output_dir / f"{config.output_name}_secciones.csv"
                        write_profile_csv(secciones, cross_sections_csv_path)

                        secciones_validas = [s for s in secciones if len(s.points) >= 2]
                        if secciones_validas:
                            from topocore.analysis.profile.dxf_export import export_cross_sections_dxf

                            cross_sections_dxf_path = config.output_dir / f"{config.output_name}_secciones.dxf"
                            export_cross_sections_dxf(secciones_validas, cross_sections_dxf_path)
        else:
            config.output_dir.mkdir(parents=True, exist_ok=True)

        # 10. Coleccion de features (si se pide) -- incluye las curvas
        features_collection: FeatureCollection | None = None
        if config.build_features:
            features_collection = FeatureCollection()
            for f in features_final:
                features_collection.add(f)
            if config.crs_epsg is not None:
                features_collection.crs = f"EPSG:{config.crs_epsg}"

            for i, c in enumerate(contours):
                contour_vertices = np.array([[p.x, p.y, p.z] for p in c.points])
                geom = FeatureGeometry(geometry_type=GeometryType.POLYLINE, vertices=contour_vertices, closed=c.closed)
                curva = Feature(
                    feature_id=100000 + i,
                    category=FeatureCategory.TERRAIN,
                    feature_type=FeatureType.CONTOUR,
                    geometry=geom,
                    attributes={"elevation": c.elevation},
                )
                features_collection.add(curva)

        # 11. Exportar DXF + GPKG -- las rutas en el resultado solo
        #     se reportan si el archivo realmente se escribio;
        #     confirmado necesario: sin esto, el resultado podia
        #     reportar una ruta de GPKG inexistente cuando crs_epsg
        #     era None (GeoPackageExporter necesita un EPSG real).
        dxf_path: Path | None = None
        gpkg_path: Path | None = None
        if features_collection is not None:
            dxf_path = config.output_dir / f"{config.output_name}.dxf"
            options = DXFExportOptions(point_labels=config.point_labels, contour_labels=config.contour_labels)
            DXFExporter(ExportContext(options=options)).export(features_collection, dxf_path)
            if config.crs_epsg is not None:
                gpkg_path = config.output_dir / f"{config.output_name}.gpkg"
                GeoPackageExporter(GPKGExportOptions(epsg=config.crs_epsg)).export(features_collection, gpkg_path)

        # 12. Quality Report (mejor esfuerzo -- sin puntos de control
        #     independientes disponibles de forma generica, se omite
        #     accuracy; density y coverage si con los puntos de
        #     terreno reales)
        quality_report = None
        quality_report_path: Path | None = None
        quality_report_pdf_path: Path | None = None
        if config.generate_quality_report:
            analyzer = QualityAnalyzer(project_name=config.output_name)
            ground_xy = np.array([[p.x, p.y] for p in result.ground], dtype=np.float64)
            density = (
                analyzer.analyze_density(
                    ground_xy, resolution=config.dtm_resolution,
                    crs=f"EPSG:{config.crs_epsg}" if config.crs_epsg is not None else None,
                )
                if ground_xy.size > 0
                else None
            )
            coverage = (
                analyzer.analyze_coverage(ground_xy, tin.bounds, cell_size=config.dtm_resolution)
                if tin is not None and ground_xy.size > 0
                else None
            )
            quality_report = analyzer.build(
                density=density,
                coverage=coverage,
                crs=f"EPSG:{config.crs_epsg}" if config.crs_epsg is not None else None,
            )

            ruta_json = config.output_dir / f"{config.output_name}_quality.json"
            ruta_json.write_text(quality_report.to_json())
            quality_report_path = ruta_json

            try:
                from topocore.quality.pdf_renderer import render_pdf

                quality_report_pdf_path = render_pdf(
                    quality_report, config.output_dir / f"{config.output_name}_quality.pdf"
                )
            except ImportError:
                # reportlab es una dependencia opcional (extra "quality-report")
                # -- el JSON ya se escribio, el PDF simplemente no se genera.
                pass

        return SurveyPipelineResult(
            points_read=len(survey_points.points),
            ground_count=len(result.ground),
            unmatched_codes=frozenset(p.code for p in result.unmatched if p.code is not None),
            boundary_area=boundary_area,
            tin=tin,
            dtm=dtm,
            contours=contours,
            features=features_collection,
            dxf_path=dxf_path,
            gpkg_path=gpkg_path,
            geotiff_path=geotiff_path,
            quality_report=quality_report,
            quality_report_path=quality_report_path,
            quality_report_pdf_path=quality_report_pdf_path,
            cross_sections_csv_path=cross_sections_csv_path,
            cross_sections_dxf_path=cross_sections_dxf_path,
        )


__all__ = ["SurveyPipeline"]
