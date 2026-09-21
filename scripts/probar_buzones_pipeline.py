"""
Prueba de BUZONES.csv -- red de alcantarillado/drenaje.

Determinaciones aplicadas, con su nivel de confianza real:

    BZ1...BZ31 -> BZ (buzon/manhole)     CONFIANZA ALTA
        Normalizado antes de FeatureBuilder, ya que ningun catalogo
        puede tener 31 entradas identicas para 31 buzones numerados
        individualmente.

    E, N, D, VC, BV                       CONFIANZA BAJA -- NO se
        registran con un tipo especifico inventado. Quedan visibles
        como codigos sin coincidencia (no perdidos, no adivinados)
        hasta que se confirme su significado real de campo.

    DELTA                                  Marcador propio del
        topografo, confirmado (con datos reales) NO vinculado a BZ
        ni a PT -- se deja sin registrar, igual que los anteriores.

Ubicacion esperada:
    data/real/buzones/BUZONES.csv
    data/catalogs/catalogo_buzones.json
"""

from pathlib import Path

from topocore.survey.reader import SurveyTXTReader
from topocore.survey.formats import SurveyFormat
from topocore.survey.code_normalization import normalize_numbered_codes
from topocore.pipeline.pipeline import SurveyPipeline
from topocore.pipeline.models import SurveyPipelineConfig, SurveyType

# ------------------------------------------------------------------
# CONFIGURACION -- ajusta estos valores segun tu proyecto real
# ------------------------------------------------------------------
RUTA_SURVEY = "data/real/buzones/BUZONES.csv"
RUTA_CATALOGO_EXTRA = "data/catalogs/catalogo_buzones.json"
FORMATO = SurveyFormat.PNEZD
EPSG = 9377  # mismo sector que mutisgeo1.csv -- MAGNA-SIRGAS Origen Nacional
INTERVALO_CURVAS = 0.5
CARPETA_SALIDA = "outputs/buzones"
NOMBRE_SALIDA = "salida_buzones"
RUTA_TEMPORAL = "outputs/buzones/_normalizado.csv"
# ------------------------------------------------------------------

# 1. Normalizar BZ1...BZ31 -> BZ (E, N tambien se dejan preparadas
#    aqui, comentadas, para cuando confirmes su significado real)
sp = SurveyTXTReader(RUTA_SURVEY, format=FORMATO).read()
sp = normalize_numbered_codes(sp, frozenset({"BZ"}))
# sp = normalize_numbered_codes(sp, frozenset({"BZ", "E", "N"}))  # activar cuando confirmes E/N

Path(CARPETA_SALIDA).mkdir(parents=True, exist_ok=True)
lineas = [f"{p.id},{p.x},{p.y},{p.z},{p.code}" for p in sp.points]
Path(RUTA_TEMPORAL).write_text("\n".join(lineas), newline="\n")

# 2. Correr el pipeline -- GENERIC: esta red de alcantarillado no
#    tiene un EJE (no es una via) ni un limite/perimetro de predio
#    que resolver (no es ese el objeto de este levantamiento).
#    PT (264 puntos) es terreno real -- build_terrain=True (por
#    defecto) es correcto aqui.
config = SurveyPipelineConfig(
    survey_type=SurveyType.GENERIC,
    survey_format=FORMATO,
    crs_epsg=EPSG,
    catalog_extra=Path(RUTA_CATALOGO_EXTRA),
    contour_interval=INTERVALO_CURVAS,
    point_labels=True,
    output_dir=Path(CARPETA_SALIDA),
    output_name=NOMBRE_SALIDA,
)

resultado = SurveyPipeline(config).run(RUTA_TEMPORAL)

print("Puntos leidos:", resultado.points_read)
print("Puntos GROUND (terreno, PT):", resultado.ground_count)
print("Sin coincidencia:", sorted(resultado.unmatched_codes))
print(f"TIN: {resultado.tin.vertex_count if resultado.tin else 0} vertices")
print(f"Contornos: {len(resultado.contours)} (intervalo {INTERVALO_CURVAS} m)")
print("DXF:", resultado.dxf_path)
print("GPKG:", resultado.gpkg_path)
print("GeoTIFF:", resultado.geotiff_path)
