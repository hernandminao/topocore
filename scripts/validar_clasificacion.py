"""
Valida CSF (clasificacion de terreno de TopoCore) contra una
clasificacion REAL ya embebida en el archivo (ej. ASPRS codigo 2 =
Ground) -- confirmado con datos reales que autzen_classified.laz
trae esta clasificacion completa.

Metricas reales calculadas:
    precision = de lo que CSF dijo que era terreno, cuanto realmente lo era
    exhaustividad (recall) = de lo que realmente era terreno, cuanto CSF detecto
    F1 = balance entre las 2 anteriores
    exactitud (accuracy) = aciertos totales / puntos totales

Uso:
    python validar_clasificacion.py archivo.laz [--cloth-resolution N]
"""

import argparse
from pathlib import Path

import numpy as np
from topocore.io.las.reader import LASReader
from topocore.io.laz.reader import LAZReader
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.classification import PointClassification
from topocore.processing.ground.csf import CSFGroundClassifier


def validar(ruta_archivo: str, cloth_resolution: float) -> None:
    ruta = Path(ruta_archivo)
    print(f"\n=== {ruta.name} ===")

    reader = LASReader(ruta) if ruta.suffix.lower() == ".las" else LAZReader(ruta)
    nube = reader.read()
    print(f"Puntos leidos: {nube.point_count:,}")

    # 1. Extraer la verdad de campo REAL (codigo ASPRS 2 = Ground)
    es_terreno_real = np.empty(nube.point_count, dtype=bool)
    offset = 0
    for chunk in nube:
        codigos = np.asarray(chunk[PointAttribute.CLASSIFICATION])
        n = len(codigos)
        es_terreno_real[offset : offset + n] = codigos == PointClassification.GROUND
        offset += n

    if not es_terreno_real.any():
        print("Este archivo no trae ningun punto con codigo GROUND (2) -- no hay verdad de campo para comparar.")
        return

    print(f"Terreno real (verdad de campo): {es_terreno_real.sum():,} de {nube.point_count:,} puntos")

    # 2. Prediccion de CSF (a ciegas, sin usar la clasificacion real)
    clasificador = CSFGroundClassifier(cloth_resolution=cloth_resolution)
    es_terreno_predicho = clasificador.classify(nube)
    print(f"Terreno predicho por CSF (cloth_resolution={cloth_resolution}): {es_terreno_predicho.sum():,} puntos")

    # 3. Matriz de confusion real
    verdaderos_positivos = int(np.sum(es_terreno_real & es_terreno_predicho))
    falsos_positivos = int(np.sum(~es_terreno_real & es_terreno_predicho))
    falsos_negativos = int(np.sum(es_terreno_real & ~es_terreno_predicho))
    verdaderos_negativos = int(np.sum(~es_terreno_real & ~es_terreno_predicho))

    precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos)
    exhaustividad = verdaderos_positivos / (verdaderos_positivos + falsos_negativos)
    f1 = 2 * precision * exhaustividad / (precision + exhaustividad)
    exactitud = (verdaderos_positivos + verdaderos_negativos) / nube.point_count

    print("\n=== Resultado real de la validacion ===")
    print(f"Verdaderos positivos (CSF acerto que era terreno):     {verdaderos_positivos:,}")
    print(f"Falsos positivos (CSF dijo terreno, no lo era):        {falsos_positivos:,}")
    print(f"Falsos negativos (era terreno, CSF no lo detecto):     {falsos_negativos:,}")
    print(f"Verdaderos negativos (CSF acerto que NO era terreno):  {verdaderos_negativos:,}")
    print(f"\nPrecision:      {precision:.4f}")
    print(f"Exhaustividad:  {exhaustividad:.4f}")
    print(f"F1:             {f1:.4f}")
    print(f"Exactitud:      {exactitud:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archivo", help="Ruta al archivo .las/.laz (debe traer clasificacion GROUND real)")
    parser.add_argument("--cloth-resolution", type=float, default=0.5)
    args = parser.parse_args()

    validar(args.archivo, args.cloth_resolution)
