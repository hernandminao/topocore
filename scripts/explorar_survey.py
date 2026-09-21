from topocore.features.catalogs.loaders.json_loader import load_json
from topocore.features.feature_builder import FeatureBuilder
from topocore.features.feature_codes import FeatureCodeRegistry
from topocore.survey.formats import SurveyFormat
from topocore.survey.reader import SurveyTXTReader

ruta = "data/real/levantamiento_via1_grammar.csv"
survey_points = SurveyTXTReader(ruta, format=SurveyFormat.PNEZD).read()

# Catálogo por defecto + tu extensión personal (BORDEI)
registry = FeatureCodeRegistry.default()
codigos_propios = load_json("data/catalogs/catalogo_hernan.json")
registry.register_many(codigos_propios)

result = FeatureBuilder(registry, use_field_code_grammar=True).build(survey_points)

print("Features construidas:", len(result.features))
for f in result.features:
    print(f" - {f.feature_type.value:20} {f.geometry.geometry_type.value:10} vértices={len(f.geometry.vertices)}")

print("\nPuntos GROUND (terreno):", len(result.ground))
print("Puntos sin coincidencia:", len(result.unmatched))
for u in result.unmatched:
    print(f"  id={u.id} code={u.code}")
