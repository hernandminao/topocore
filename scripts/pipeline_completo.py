import dataclasses

from topocore.analysis.statistics.manager import StatisticsAnalysis
from topocore.dxf import DXFExporter
from topocore.features.catalogs.loaders.json_loader import load_json
from topocore.features.feature_builder import FeatureBuilder
from topocore.features.feature_codes import FeatureCodeRegistry
from topocore.features.models import FeatureCollection
from topocore.features.side.resolver import SideResolver
from topocore.geodesy.crs import CRS
from topocore.geometry.point3d import Point3D
from topocore.gpkg import GeoPackageExporter, GPKGExportOptions
from topocore.survey.formats import SurveyFormat
from topocore.survey.reader import SurveyTXTReader
from topocore.terrain.contours import ContourGenerator
from topocore.terrain.dtm import DTM
from topocore.terrain.enums import InterpolationMethod
from topocore.terrain.grid import Grid
from topocore.terrain.interpolation import TerrainInterpolator
from topocore.terrain.tin import TIN

# 1. Leer el levantamiento (ya preprocesado con grammar)
survey_points = SurveyTXTReader(
    "data/real/levantamiento_via1_grammar.csv",
    format=SurveyFormat.PNEZD,
).read()

# 2. Asignar el CRS real -- MAGNA-SIRGAS 2018 / Origen-Nacional
survey_points = dataclasses.replace(survey_points, crs=CRS.from_epsg(9377))
print("CRS asignado:", survey_points.crs)

# 3. Catálogo: por defecto + tu extensión personal (BORDEI)
registry = FeatureCodeRegistry.default()
registry.register_many(load_json("data/catalogs/catalogo_hernan.json"))

# 4. Construir features (con grammar, ya confirmado que funciona)
result = FeatureBuilder(registry, use_field_code_grammar=True).build(survey_points)
print("Features construidas:", len(result.features))
print("Puntos GROUND (terreno):", len(result.ground))
print("Sin coincidencia:", len(result.unmatched))

# 5. Lateralidad -- BORDE/BORDEI respecto al EJE real
coleccion = FeatureCollection()
for f in result.features:
    coleccion.add(f)
coleccion.crs = "EPSG:9377"  # FeatureCollection.crs usa string, no CRS

coleccion_resuelta = SideResolver().resolve(coleccion)

print("\n=== Lateralidad resuelta ===")
for f in coleccion_resuelta:
    if f.feature_type.value == "centerline":
        for v in f.geometry.vertices:
            print(f"  x={v[0]:.3f}  y={v[1]:.3f}  z={v[2]:.3f}")
        break

# 6. Terreno real -- TIN desde los puntos GROUND (TN)
puntos_terreno = tuple(Point3D(p.x, p.y, p.z) for p in result.ground)
print(f"\nConstruyendo TIN con {len(puntos_terreno)} puntos de terreno reales...")

tin = TIN.from_points(puntos_terreno)

print("TIN construido:")
print("  Vértices:", tin.vertex_count)
print("  Triángulos:", tin.triangle_count)
print("  Extensión (bounds):", tin.bounds)

stats = StatisticsAnalysis().elevation(tin)
print("\nEstadísticas reales de elevación:")
print(f"  Mínima:  {stats.minimum:.3f} m")
print(f"  Máxima:  {stats.maximum:.3f} m")
print(f"  Media:   {stats.mean:.3f} m")
print(f"  Rango:   {stats.range:.3f} m")

# 7. DTM real -- interpolado sobre una grilla cubriendo la extensión del TIN
min_x, min_y, max_x, max_y = tin.bounds
grid = Grid(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, resolution=1.0)
interpolador = TerrainInterpolator(tin, method=InterpolationMethod.LINEAR)
dtm = DTM.from_tin(tin, grid, interpolador)
print("\nDTM construido correctamente (resolución 1.0 m).")

# 8. Contornos reales -- intervalo 0.5 m, acorde al rango de 10.4 m del terreno
contornos = ContourGenerator(tin).generate(interval=0.5, base=0.0)
print(f"Contornos generados: {len(contornos)} líneas, intervalo 0.5 m")

DXFExporter().export(coleccion_resuelta, "salida_via_real.dxf")
GeoPackageExporter(GPKGExportOptions(epsg=9377)).export(coleccion_resuelta, "salida_via_real.gpkg")
print("Exportado -- ábrelo en Civil 3D/QGIS para verificar visualmente")
