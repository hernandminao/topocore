"""
Prueba completa de la VIA 2 con SurveyPipeline -- ROAD.

Ubicacion esperada:
    data/real/via2/levantamiento_via2.csv
"""

from pathlib import Path

from topocore.pipeline.models import SurveyPipelineConfig, SurveyType
from topocore.pipeline.pipeline import SurveyPipeline
from topocore.survey.formats import SurveyFormat

# ------------------------------------------------------------------
# CONFIGURACION -- ajusta estos valores segun tu proyecto real
# ------------------------------------------------------------------
RUTA_SURVEY = "data/real/via2/levantamiento_via2.csv"
FORMATO = SurveyFormat.PENZD  # via2 es Este,Norte,Z,Descripcion
EPSG = 9377  # MAGNA-SIRGAS 2018 / Origen-Nacional
INTERVALO_CURVAS = 0.5
RESOLUCION_DTM = 1.0
CARPETA_SALIDA = "outputs/via2"
NOMBRE_SALIDA = "salida_via2"
# ------------------------------------------------------------------

config = SurveyPipelineConfig(
    survey_type=SurveyType.ROAD,
    survey_format=FORMATO,
    reference_code="EJE",
    linear_codes=frozenset({"EJE", "BORDE", "BORDEI", "CERCA"}),
    splittable_codes=frozenset({"BORDE", "BORDEI", "CERCA"}),
    crs_epsg=EPSG,
    contour_interval=INTERVALO_CURVAS,
    dtm_resolution=RESOLUCION_DTM,
    point_labels=True,
    output_dir=Path(CARPETA_SALIDA),
    output_name=NOMBRE_SALIDA,
    generate_cross_sections=True,
    cross_section_width=4.0,  # ajusta según tu vía real
    cross_section_interval=5.0,  # cada 5m, en vez de por vértice
)

resultado = SurveyPipeline(config).run(RUTA_SURVEY)

print("Puntos leidos:", resultado.points_read)
print("Puntos GROUND (terreno):", resultado.ground_count)
print("Sin coincidencia:", sorted(resultado.unmatched_codes))
print(f"TIN: {resultado.tin.vertex_count} vertices, {resultado.tin.triangle_count} triangulos")
print(f"Contornos: {len(resultado.contours)} (intervalo {INTERVALO_CURVAS} m)")
print("DXF:", resultado.dxf_path)
print("GPKG:", resultado.gpkg_path)
print("GeoTIFF:", resultado.geotiff_path)
if resultado.quality_report:
    print("Quality Report: generado")
