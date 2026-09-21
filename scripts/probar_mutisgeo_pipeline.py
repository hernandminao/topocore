"""
Prueba completa del predio con estructuras (mutisgeo1.csv).

Este levantamiento NO encaja en survey_type=PROPERTY del pipeline
por una razon real: necesita 2 codigos distintos (EST, AD) resueltos
por clustering, cada uno con su propio "eps" (EST=1.0m para columnas
apretadas, AD=3.0m para tramos de anden mas separados). El pipeline
actual solo admite UN boundary_code con UN eps -- por eso aqui se
preprocesa el survey manualmente ANTES de pasarlo al pipeline (que
corre como GENERIC, solo para terreno/curvas/export).

Ubicacion esperada:
    data/real/mutisgeo/mutisgeo1.csv
    data/catalogs/catalogo_mutisgeo.json
"""
from pathlib import Path

from topocore.survey.reader import SurveyTXTReader
from topocore.survey.formats import SurveyFormat
from topocore.survey.multiline import split_by_clustering, ClusterSplitConfig
from topocore.pipeline.pipeline import SurveyPipeline
from topocore.pipeline.models import SurveyPipelineConfig, SurveyType

# ------------------------------------------------------------------
# CONFIGURACION -- ajusta estos valores segun tu proyecto real
# ------------------------------------------------------------------
RUTA_SURVEY = "data/real/mutisgeo/mutisgeo1.csv"
RUTA_CATALOGO_EXTRA = "data/catalogs/catalogo_mutisgeo.json"
FORMATO = SurveyFormat.PNEZD
EPSG = 9377
EPS_EST = 1.0   # columnas -- ajusta si tu revision visual en QGIS lo pide
EPS_AD = 3.0    # tramos de anden
INTERVALO_CURVAS = 0.5
CARPETA_SALIDA = "outputs/mutisgeo"
NOMBRE_SALIDA = "salida_mutisgeo"
RUTA_TEMPORAL = "outputs/mutisgeo/_preprocesado.csv"
# ------------------------------------------------------------------

# 1. Preprocesar: resolver EST (columnas) y AD (andenes) por separado,
#    cada uno con su propio eps -- esto es lo que el pipeline no
#    hace por si solo para este caso.
survey_points = SurveyTXTReader(RUTA_SURVEY, format=FORMATO).read()
survey_points = split_by_clustering(
    survey_points, ClusterSplitConfig(linear_codes=frozenset({"EST"}), eps=EPS_EST, ordering="hull")
)
survey_points = split_by_clustering(
    survey_points, ClusterSplitConfig(linear_codes=frozenset({"AD"}), eps=EPS_AD, ordering="hull")
)

Path(CARPETA_SALIDA).mkdir(parents=True, exist_ok=True)
lineas = [f"{p.id},{p.x},{p.y},{p.z},{p.code}" for p in survey_points.points]
Path(RUTA_TEMPORAL).write_text("\n".join(lineas), newline="\n")

# 2. Correr el pipeline como GENERIC sobre el archivo ya preprocesado
#    -- se encarga de terreno, curvas, DXF/GPKG, Quality Report.
config = SurveyPipelineConfig(
    survey_type=SurveyType.GENERIC,
    survey_format=FORMATO,
    crs_epsg=EPSG,
    catalog_extra=Path(RUTA_CATALOGO_EXTRA),
    build_terrain=False,  # este predio no trae ningun disparo de terreno (TN)
    point_labels=True,
    output_dir=Path(CARPETA_SALIDA),
    output_name=NOMBRE_SALIDA,
)

resultado = SurveyPipeline(config).run(RUTA_TEMPORAL)

print("Puntos leidos:", resultado.points_read)
print("Puntos GROUND (terreno):", resultado.ground_count)
print("Sin coincidencia:", sorted(resultado.unmatched_codes))
print("DXF:", resultado.dxf_path)
print("GPKG:", resultado.gpkg_path)
