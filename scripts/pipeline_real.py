import dataclasses

from topocore.survey.reader import SurveyTXTReader
from topocore.survey.formats import SurveyFormat
from topocore.geodesy.crs import CRS
from topocore.features.feature_codes import FeatureCodeRegistry
from topocore.features.catalogs.loaders.json_loader import load_json
from topocore.features.feature_builder import FeatureBuilder
from topocore.features.models import FeatureCollection
from topocore.features.side.resolver import SideResolver

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
coleccion.crs = "EPSG:9377"  # FeatureCollection.crs usa string, no CRS -- confirmado en geodesy.md

coleccion_resuelta = SideResolver().resolve(coleccion)

print("\n=== Lateralidad resuelta ===")
for f in coleccion_resuelta:
    if f.feature_type.value == "pavement_edge":
        print(f"  {f.feature_type.value}: side={f.attributes.get('side')}, method={f.attributes.get('side_method')}")
