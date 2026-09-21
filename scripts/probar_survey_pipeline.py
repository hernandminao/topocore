"""
Prueba completa del predio de Survey.csv -- PROPERTY, coordenadas
locales (sin CRS real).

"CERCA stake 2" se unifica bajo "CERCA" antes de correr el pipeline
-- confirmado con la documentacion oficial (Survey.pdf) que es una
estaca mas de la misma cerca, no un codigo distinto.

Ubicacion esperada:
    data/real/survey/Survey.csv
"""

import dataclasses
from pathlib import Path

from topocore.survey.reader import SurveyTXTReader
from topocore.survey.formats import SurveyFormat
from topocore.pipeline.pipeline import SurveyPipeline
from topocore.pipeline.models import SurveyPipelineConfig, SurveyType, BoundaryStrategy

# ------------------------------------------------------------------
# CONFIGURACION -- ajusta estos valores segun tu proyecto real
# ------------------------------------------------------------------
RUTA_SURVEY = "data/real/survey/Survey.csv"
FORMATO = SurveyFormat.PNEZD
EPSG = None  # coordenadas locales/arbitrarias -- no se inventa un CRS
BOUNDARY_EPS = 1000.0  # mantiene el perimetro completo en 1 solo cluster
INTERVALO_CURVAS = 0.5
CARPETA_SALIDA = "outputs/survey"
NOMBRE_SALIDA = "salida_survey"
RUTA_TEMPORAL = "outputs/survey/_unificado.csv"
# ------------------------------------------------------------------

# 1. Unificar "CERCA stake 2" -> "CERCA" antes de correr el pipeline
sp = SurveyTXTReader(RUTA_SURVEY, format=FORMATO).read()
puntos_unificados = tuple(dataclasses.replace(p, code="CERCA") if p.code == "CERCA stake 2" else p for p in sp.points)
sp = dataclasses.replace(sp, points=puntos_unificados)

Path(CARPETA_SALIDA).mkdir(parents=True, exist_ok=True)
lineas = [f"{p.id},{p.x},{p.y},{p.z},{p.code}" for p in sp.points]
Path(RUTA_TEMPORAL).write_text("\n".join(lineas), newline="\n")

# 2. Correr el pipeline completo -- PROPERTY, limite por clustering+hull
config = SurveyPipelineConfig(
    survey_type=SurveyType.PROPERTY,
    survey_format=FORMATO,
    boundary_strategy=BoundaryStrategy.CLUSTER_HULL,
    boundary_code="LINDERO",
    boundary_eps=BOUNDARY_EPS,
    crs_epsg=EPSG,
    contour_interval=INTERVALO_CURVAS,
    point_labels=True,
    output_dir=Path(CARPETA_SALIDA),
    output_name=NOMBRE_SALIDA,
)

resultado = SurveyPipeline(config).run(RUTA_TEMPORAL)

print("Puntos leidos:", resultado.points_read)
print("Puntos GROUND (terreno):", resultado.ground_count)
print("Area del limite (CERCA):", resultado.boundary_area)
print("Sin coincidencia:", sorted(resultado.unmatched_codes))
print(f"TIN: {resultado.tin.vertex_count} vertices, {resultado.tin.triangle_count} triangulos")
print(f"Contornos: {len(resultado.contours)} (intervalo {INTERVALO_CURVAS} m)")
print("DXF:", resultado.dxf_path)
print("GeoTIFF:", resultado.geotiff_path, "(None es correcto -- sin CRS real)")
