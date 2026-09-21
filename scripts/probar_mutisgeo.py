"""
Prueba del levantamiento de predio real (mutisgeo1.csv).

Ubicacion esperada de archivos (ajusta si tu estructura es distinta):
    data/real/mutisgeo/mutisgeo1.csv
    data/catalogs/catalogo_mutisgeo.json
"""
import dataclasses
from collections import Counter
from pathlib import Path

from topocore.dxf import DXFExporter
from topocore.dxf.models import DXFExportOptions, ExportContext
from topocore.features.catalogs.loaders.json_loader import load_json
from topocore.features.feature_builder import FeatureBuilder
from topocore.features.feature_codes import FeatureCodeRegistry
from topocore.features.models import FeatureCollection
from topocore.geodesy.crs import CRS
from topocore.survey.formats import SurveyFormat
from topocore.survey.multiline import ClusterSplitConfig, split_by_clustering
from topocore.survey.reader import SurveyTXTReader

RUTA_SURVEY = "data/real/mutisgeo/mutisgeo1.csv"
RUTA_CATALOGO = "data/catalogs/catalogo_mutisgeo.json"

# 1. Leer el levantamiento real
survey_points = SurveyTXTReader(RUTA_SURVEY, format=SurveyFormat.PNEZD).read()
print("Puntos leidos:", len(survey_points.points))

# 2. Agrupar EST (columnas, poligonos) y AD (andenes, lineas) --
#    ajusta 'eps' si tu revision visual en QGIS muestra que hace
#    falta separar mas o menos figuras
config_est = ClusterSplitConfig(
    linear_codes=frozenset({"EST"}), eps=1.0, min_samples=1, ordering="angular",
)
survey_points = split_by_clustering(survey_points, config_est)

config_ad = ClusterSplitConfig(
    linear_codes=frozenset({"AD"}), eps=3.0, min_samples=1, ordering="axis",
)
survey_points = split_by_clustering(survey_points, config_ad)

# 3. CRS real -- ajustalo si tu predio no usa MAGNA-SIRGAS Origen Nacional
survey_points = dataclasses.replace(survey_points, crs=CRS.from_epsg(9377))
print("CRS asignado:", survey_points.crs)

# 4. Catalogo: por defecto + los 4 codigos reales de este predio
registry = FeatureCodeRegistry.default()
registry.register_many(load_json(RUTA_CATALOGO))

# 5. Construir features -- grammar activado porque split_by_clustering
#    ya dejo los codigos con sufijo BASE.FIGURA.S/E
result = FeatureBuilder(registry, use_field_code_grammar=True).build(survey_points)

print("\nFeatures construidas:", len(result.features))
print("Puntos GROUND (terreno):", len(result.ground))
print("Sin coincidencia:", len(result.unmatched))
if result.unmatched:
    print("  Codigos sin coincidencia:", sorted({p.code for p in result.unmatched}))

print("\n=== Detalle por tipo ===")
conteo = Counter(f.feature_type.value for f in result.features)
for tipo, cantidad in conteo.items():
    print(f"  {tipo}: {cantidad}")

print("\n=== Detalle de estructuras (columnas) ===")
for f in result.features:
    if f.feature_type.value == "building":
        print(f"  {len(f.geometry.vertices)} vertices, cerrado={f.geometry.closed}")

# 6. Exportar a DXF -- capas correctas tomadas directamente del
#    catalogo (cad_layer), confirmado: PARAMENTOS, ANDENES,
#    EDIFICACIONES, SERVICIOS, CONTROL
coleccion = FeatureCollection()
for f in result.features:
    coleccion.add(f)
coleccion.crs = "EPSG:9377"

ruta_dxf = "outputs/mutisgeo/salida_mutisgeo.dxf"
Path(ruta_dxf).parent.mkdir(parents=True, exist_ok=True)
DXFExporter(ExportContext(options=DXFExportOptions(point_labels=True))).export(coleccion, ruta_dxf)
print(f"\nExportado: {ruta_dxf}")
