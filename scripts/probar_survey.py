"""
Procesa Survey.csv -- levantamiento con estacion total en
coordenadas LOCALES/ARBITRARIAS (sin CRS real conocido -- no se le
asigna ningun EPSG inventado).

Codigos NO reconocidos ("CERCA stake 2", "test loc 2", "test loc 3",
"res", "EOS", "wall") quedan visibles como no coincidentes -- su
significado no se conoce, y no se les inventa uno.

Ubicacion esperada:
    data/real/survey/Survey.csv
"""

import dataclasses
from pathlib import Path

from topocore.analysis.statistics.manager import StatisticsAnalysis
from topocore.dxf import DXFExporter
from topocore.dxf.models import DXFExportOptions, ExportContext
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
from topocore.geometry.point3d import Point3D
from topocore.survey.formats import SurveyFormat
from topocore.survey.multiline import ClusterSplitConfig, split_by_clustering
from topocore.survey.reader import SurveyTXTReader
from topocore.terrain.contours import ContourGenerator
from topocore.terrain.tin import TIN

RUTA_SURVEY = "data/real/survey/Survey.csv"
RUTA_SALIDA_DXF = "outputs/survey/salida_survey.dxf"

# 1. Leer el levantamiento
survey_points = SurveyTXTReader(RUTA_SURVEY, format=SurveyFormat.PNEZD).read()
print("Puntos leidos:", len(survey_points.points))
print("CRS:", survey_points.crs, "(coordenadas locales -- sin CRS real, a proposito)")

# 1b. "CERCA stake 2" es una estaca mas de la misma cerca -- confirmado
#     con la documentacion oficial del levantamiento (PDF): dice
#     literalmente "fence/stake 2", y esta a solo 12m del punto de
#     cerca mas cercano. Se unifica bajo "CERCA" antes de procesar.
puntos_unificados = tuple(
    dataclasses.replace(p, code="CERCA") if p.code == "CERCA stake 2" else p for p in survey_points.points
)
survey_points = dataclasses.replace(survey_points, points=puntos_unificados)

# 2. CERCA es 1 perimetro unico, confirmado -- reordenar con
#    "nearest" (no "angular"): el predio no es convexo respecto a
#    su propio centro, y "angular" produce saltos grandes
config_cerca = ClusterSplitConfig(
    linear_codes=frozenset({"CERCA"}),
    eps=1000.0,
    min_samples=1,
    ordering="hull",
)
survey_points = split_by_clustering(survey_points, config_cerca)

# 3. Construir features -- catalogo por defecto, sin inventar
#    significado para los codigos no reconocidos
registry = FeatureCodeRegistry.default()
result = FeatureBuilder(registry, use_field_code_grammar=True).build(survey_points)

print("\nFeatures construidas:", len(result.features))
print("Puntos GROUND (terreno):", len(result.ground))
print("Sin coincidencia:", len(result.unmatched))
if result.unmatched:
    print("  Codigos sin coincidencia:", sorted({p.code for p in result.unmatched}))

# 3b. Cerrar el perimetro de CERCA -- confirmado, la cerca es un
#     perimetro completo del predio (no cualquier fence generico
#     necesariamente lo es), asi que se marca closed=True aqui, en
#     este proyecto especifico -- no se cambia el catalogo global
#     (otro proyecto podria tener un tramo de cerca real que NO sea
#     un perimetro cerrado).
features_ajustadas = []
for f in result.features:
    if f.feature_type.value == "fence":
        geometria_cerrada = dataclasses.replace(f.geometry, closed=True)
        f = dataclasses.replace(f, geometry=geometria_cerrada)
    features_ajustadas.append(f)

# 4. Terreno -- TIN desde los puntos GROUND
puntos_terreno = tuple(Point3D(p.x, p.y, p.z) for p in result.ground)
tin = TIN.from_points(puntos_terreno)
print(f"\nTIN: {tin.vertex_count} vertices, {tin.triangle_count} triangulos")

stats = StatisticsAnalysis().elevation(tin)
print(f"Elevacion: min={stats.minimum:.3f} max={stats.maximum:.3f} media={stats.mean:.3f} rango={stats.range:.3f}")

# 5. Curvas de nivel
contornos = ContourGenerator(tin).generate(interval=0.25, base=0.0)
print(f"Contornos generados: {len(contornos)} (intervalo 0.25)")

# 6. Exportar -- sin CRS real, el DXF queda igual de valido
#    (DXF nunca guarda CRS de por si, ya lo confirmamos antes)
coleccion = FeatureCollection()
for f in features_ajustadas:
    coleccion.add(f)

for i, c in enumerate(contornos):
    import numpy as np

    vertices = np.array([[p.x, p.y, p.z] for p in c.points])
    geom = FeatureGeometry(geometry_type=GeometryType.POLYLINE, vertices=vertices, closed=c.closed)
    curva = Feature(
        feature_id=100000 + i,
        category=FeatureCategory.TERRAIN,
        feature_type=FeatureType.CONTOUR,
        geometry=geom,
        attributes={"elevation": c.elevation},
    )
    coleccion.add(curva)

Path(RUTA_SALIDA_DXF).parent.mkdir(parents=True, exist_ok=True)
DXFExporter(ExportContext(options=DXFExportOptions(point_labels=True))).export(coleccion, RUTA_SALIDA_DXF)
print(f"\nExportado: {RUTA_SALIDA_DXF}")
