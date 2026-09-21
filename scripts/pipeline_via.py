"""
Pipeline completo: survey real -> CRS -> multiline (division
automatica de lados, geometrica respecto al eje) -> features ->
lateralidad -> terreno -> DTM (+ GeoTIFF real) -> secciones
transversales (CSV + DXF, solo terreno) -> contornos -> exportacion
DXF (con etiquetas) + GeoPackage (con columnas nativas).

Uso:
    python scripts/pipeline_via.py <entrada.csv> <FORMATO> <epsg> <carpeta_salida> <nombre_base> [intervalo_curvas]

Ejemplo:
    python scripts/pipeline_via.py data/real/via2/levantamiento_via2.csv \
        PENZD 9377 outputs/via2 salida_via2 0.5
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np
from topocore.analysis.exceptions import ProfileError
from topocore.analysis.profile.dxf_export import export_cross_sections_dxf
from topocore.analysis.profile.manager import ProfileAnalysis
from topocore.analysis.profile.writer import write_profile_csv
from topocore.analysis.statistics.manager import StatisticsAnalysis
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
from topocore.features.side.resolver import SideResolver
from topocore.geodesy.crs import CRS
from topocore.geometry.point3d import Point3D
from topocore.gpkg import GeoPackageExporter, GPKGExportOptions
from topocore.io.raster.geotiff import write_dtm_geotiff
from topocore.survey.formats import SurveyFormat
from topocore.survey.multiline import MultilineSplitConfig, split_multiline_codes
from topocore.survey.reader import SurveyTXTReader
from topocore.terrain.contours import ContourGenerator
from topocore.terrain.dtm import DTM
from topocore.terrain.enums import InterpolationMethod
from topocore.terrain.grid import Grid
from topocore.terrain.interpolation import TerrainInterpolator
from topocore.terrain.tin import TIN

MULTILINE_CONFIG = MultilineSplitConfig(
    linear_codes=frozenset({"EJE", "BORDE", "BORDEI", "CERCA"}),
    splittable_codes=frozenset({"BORDE", "BORDEI", "CERCA"}),
    reference_code="EJE",
)


def _direccion_local(eje_xy: list, indice: int) -> tuple:
    """
    Replica la logica interna real de CrossSectionProfile._local_direction()
    -- confirmado necesario para generar cada seccion de forma
    individual y resiliente (ver el comentario en el Paso 9), en vez
    de depender de ProfileAnalysis.cross_section() para todo el eje
    de una sola vez.
    """
    if indice == 0:
        return eje_xy[0], eje_xy[1]
    if indice == len(eje_xy) - 1:
        return eje_xy[-2], eje_xy[-1]

    anterior, actual, siguiente = eje_xy[indice - 1], eje_xy[indice], eje_xy[indice + 1]
    entrante = (actual[0] - anterior[0], actual[1] - anterior[1])
    saliente = (siguiente[0] - actual[0], siguiente[1] - actual[1])
    origen = (actual[0] - entrante[0], actual[1] - entrante[1])
    destino = (actual[0] + saliente[0], actual[1] + saliente[1])
    return origen, destino


def _generar_secciones_resilientes(eje_xy: list, tin, *, width: float):
    """
    Genera una seccion transversal por cada vertice real del EJE,
    aislando cada estacion con su propio try/except -- confirmado
    necesario con datos reales: ProfileAnalysis.cross_section() falla
    por completo si UNA sola estacion se sale del TIN, perdiendo
    todas las demas secciones aunque sean validas.
    """
    import math

    analizador = ProfileAnalysis()
    secciones = []
    estacion_acumulada = 0.0

    for indice, vertice in enumerate(eje_xy):
        origen, destino = _direccion_local(eje_xy, indice)
        try:
            secciones.append(analizador.transversal(origen, destino, estacion_acumulada, tin, width=width))
        except ProfileError:
            pass

        if indice < len(eje_xy) - 1:
            siguiente = eje_xy[indice + 1]
            estacion_acumulada += math.hypot(siguiente[0] - vertice[0], siguiente[1] - vertice[1])

    return secciones


def ejecutar_pipeline(
    ruta_entrada: str,
    formato: SurveyFormat,
    epsg: int,
    carpeta_salida: str,
    nombre_base: str,
    intervalo_curvas: float = 0.5,
) -> None:
    Path(carpeta_salida).mkdir(parents=True, exist_ok=True)

    # 1. Leer el levantamiento crudo -- sin preproceso externo
    survey_points = SurveyTXTReader(ruta_entrada, format=formato).read()
    print(f"Puntos leidos: {len(survey_points.points)}")

    # 2. Resolver automaticamente el eje de avance y los lados (BORDE/CERCA),
    #    de forma geometrica respecto al EJE -- nunca cruza el eje
    survey_points = split_multiline_codes(survey_points, MULTILINE_CONFIG)

    # 3. Asignar el CRS real
    survey_points = dataclasses.replace(survey_points, crs=CRS.from_epsg(epsg))
    print("CRS asignado:", survey_points.crs)

    # 4. Catalogo: por defecto + extension personal (ej. BORDEI)
    registry = FeatureCodeRegistry.default()
    registry.register_many(load_json("data/catalogs/catalogo_hernan.json"))

    # 5. Construir features (con grammar, ya confirmado que funciona)
    result = FeatureBuilder(registry, use_field_code_grammar=True).build(survey_points)
    print("Features construidas:", len(result.features))
    print("Puntos GROUND (terreno):", len(result.ground))
    print("Sin coincidencia:", len(result.unmatched))
    if result.unmatched:
        print("  Codigos sin coincidencia:", sorted({p.code for p in result.unmatched}))

    # 6. Lateralidad -- BORDE/BORDEI/CERCA respecto al EJE real
    coleccion = FeatureCollection()
    for f in result.features:
        coleccion.add(f)
    coleccion.crs = f"EPSG:{epsg}"
    coleccion_resuelta = SideResolver().resolve(coleccion)

    print("\n=== Lateralidad ===")
    for f in coleccion_resuelta:
        if f.feature_type.value == "pavement_edge":
            print(f"  side={f.attributes.get('side')} method={f.attributes.get('side_method')}")

    # 7. Terreno -- TIN desde los puntos GROUND
    puntos_terreno = tuple(Point3D(p.x, p.y, p.z) for p in result.ground)
    tin = TIN.from_points(puntos_terreno)
    print(f"\nTIN: {tin.vertex_count} vertices, {tin.triangle_count} triangulos")
    print("Bounds:", tin.bounds)

    stats = StatisticsAnalysis().elevation(tin)
    print(f"Elevacion: min={stats.minimum:.3f} max={stats.maximum:.3f} media={stats.mean:.3f} rango={stats.range:.3f}")

    # 8. DTM + GeoTIFF real
    min_x, min_y, max_x, max_y = tin.bounds
    grid = Grid(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, resolution=1.0)
    interpolador = TerrainInterpolator(tin, method=InterpolationMethod.LINEAR)
    dtm = DTM.from_tin(tin, grid, interpolador)

    ruta_dtm = f"{carpeta_salida}/{nombre_base}_dtm.tif"
    write_dtm_geotiff(dtm, ruta_dtm, crs_wkt=CRS.from_epsg(epsg).to_wkt())
    print(f"DTM exportado: {ruta_dtm}")

    # 9. Secciones transversales reales, a lo largo del EJE -- CSV + DXF
    #    (V1 honesta: solo terreno real, sin superficie de diseno,
    #    sin corte/relleno -- ver dxf_export.py)
    #
    #    Cada estacion se genera de forma INDIVIDUAL y aislada --
    #    confirmado necesario con datos reales: si se pide todo el eje
    #    de una sola vez (via ProfileAnalysis.cross_section()), UNA
    #    sola estacion que se salga del TIN hace fallar el calculo
    #    completo, perdiendo TODAS las demas secciones aunque si sean
    #    validas. Se replica aqui la misma logica interna real de
    #    cross_section() (direccion local por vertice), pero llamando
    #    a transversal() estacion por estacion, con su propio
    #    try/except -- una estacion que no cabe en el TIN (comun en
    #    los extremos de un levantamiento, o con densidad de puntos de
    #    terreno baja) se omite, sin perder las demas.
    eje_features = [f for f in coleccion_resuelta if f.feature_type == FeatureType.CENTERLINE]
    if eje_features:
        eje_xy = [(float(v[0]), float(v[1])) for v in eje_features[0].geometry.vertices]
        secciones = _generar_secciones_resilientes(eje_xy, tin, width=5.0)
        total_estaciones = len(eje_xy)
        print(f"\nSecciones transversales: {len(secciones)} de {total_estaciones} estaciones "
              f"generadas correctamente (ancho 5 m)")
        if len(secciones) < total_estaciones:
            print(f"  {total_estaciones - len(secciones)} estacion(es) omitida(s) -- "
                  "se salen del TIN real (extremos del levantamiento, o densidad de "
                  "puntos de terreno insuficiente en esa zona)")

        if secciones:
            ruta_secciones_csv = f"{carpeta_salida}/{nombre_base}_secciones.csv"
            write_profile_csv(secciones, ruta_secciones_csv)
            print(f"Secciones exportadas (CSV): {ruta_secciones_csv}")

            secciones_validas = [s for s in secciones if len(s.points) >= 2]
            if secciones_validas:
                ruta_secciones_dxf = f"{carpeta_salida}/{nombre_base}_secciones.dxf"
                export_cross_sections_dxf(secciones_validas, ruta_secciones_dxf)
                print(f"Secciones exportadas (DXF): {ruta_secciones_dxf}")
    else:
        print("\nSecciones transversales: omitidas -- no se encontro un EJE (centerline) en la coleccion")

    # 10. Contornos -- convertidos a Feature y agregados a la coleccion principal
    contornos = ContourGenerator(tin).generate(interval=intervalo_curvas, base=0.0)
    for i, c in enumerate(contornos):
        vertices = np.array([[p.x, p.y, p.z] for p in c.points])
        geom = FeatureGeometry(geometry_type=GeometryType.POLYLINE, vertices=vertices, closed=c.closed)
        curva = Feature(
            feature_id=100000 + i,
            category=FeatureCategory.TERRAIN,
            feature_type=FeatureType.CONTOUR,
            geometry=geom,
            attributes={"elevation": c.elevation},
        )
        coleccion_resuelta.add(curva)
    print(f"\nContornos agregados: {len(contornos)} (intervalo {intervalo_curvas} m)")

    # 11. Exportar -- DXF (con etiquetas de curvas y puntos) + GeoPackage
    #     (elevation/survey_id como columnas nativas)
    ruta_dxf = f"{carpeta_salida}/{nombre_base}.dxf"
    ruta_gpkg = f"{carpeta_salida}/{nombre_base}.gpkg"
    DXFExporter(ExportContext(options=DXFExportOptions(point_labels=True))).export(coleccion_resuelta, ruta_dxf)
    GeoPackageExporter(GPKGExportOptions(epsg=epsg)).export(coleccion_resuelta, ruta_gpkg)
    print(f"\nExportado: {ruta_dxf}")
    print(f"Exportado: {ruta_gpkg}")


if __name__ == "__main__":
    if len(sys.argv) not in (6, 7):
        print("Uso: python pipeline_via.py <entrada.csv> <FORMATO> <epsg> <carpeta_salida> <nombre_base> [intervalo_curvas]")
        sys.exit(1)

    intervalo = float(sys.argv[6]) if len(sys.argv) == 7 else 0.5
    ejecutar_pipeline(sys.argv[1], SurveyFormat[sys.argv[2]], int(sys.argv[3]), sys.argv[4], sys.argv[5], intervalo)
