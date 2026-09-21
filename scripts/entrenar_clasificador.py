"""
Entrena un clasificador multi-clase real (Random Forest) usando datos
YA ETIQUETADOS -- no un modelo pre-entrenado incluido con TopoCore,
sino el flujo real para que cada usuario entrene el suyo con sus
propios datos reales.

Requiere scikit-learn (dependencia opcional de TopoCore):
    pip install scikit-learn

Uso:
    python entrenar_clasificador.py archivo_etiquetado.laz [--test-size 0.3] [--n-estimators 100]

Divide los puntos del archivo en train/test (division ALEATORIA por
punto, no espacial -- ver nota de limitaciones abajo), entrena sobre
train, evalua sobre test con metricas reales por clase.

QUE ARCHIVOS SIRVEN Y CUALES NO -- confirmado con el codigo real de
TopoCore, no una suposicion:

  .laz / .las  -- SI, unicamente si YA traen el campo CLASSIFICATION
                  poblado con MAS DE UNA clase real (ej. Ground +
                  Vegetacion + Edificio). Verificalo primero con
                  verificar_clasificacion.py -- si dice "NO viene
                  clasificado" o solo trae 1 clase, no sirve para
                  entrenar (no hay nada que aprender).

  .e57         -- NO. Confirmado en el codigo: el formato E57, tal
                  como lo lee TopoCore, solo trae X/Y/Z -- nunca un
                  campo de clasificacion, ni siquiera vacio. Este
                  script detecta esto y avisa con un mensaje claro
                  en vez de fallar con un error crudo.

  archivo clasificado SOLO por CSF/PMF propio (2 clases:
  Ground/Unclassified) -- tecnicamente corre, pero es circular: le
  estarias enseñando al modelo lo mismo que CSF/PMF ya hacen solos,
  sin ningun valor agregado real. Solo tiene sentido con clasificacion
  MANUAL o de un proveedor externo (como autzen_classified.laz, con
  5 clases reales).
"""

import argparse
from pathlib import Path

import numpy as np
from topocore.io.e57.reader import E57Reader
from topocore.io.las.reader import LASReader
from topocore.io.laz.reader import LAZReader
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.classification import PointClassification
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.random_forest import RandomForestClassifier


def _abrir_lector(ruta: Path):
    extension = ruta.suffix.lower()
    if extension == ".las":
        return LASReader(ruta)
    if extension == ".laz":
        return LAZReader(ruta)
    if extension == ".e57":
        return E57Reader(ruta, chunk_size=1_000_000)
    raise ValueError(f"Extension no soportada: '{extension}'. Se esperaba .las, .laz o .e57.")


def _construir_nube(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> PointCloud:
    chunk = Chunk(size=len(x), attributes=(PointAttribute.X, PointAttribute.Y, PointAttribute.Z))
    chunk._data[PointAttribute.X][:] = x
    chunk._data[PointAttribute.Y][:] = y
    chunk._data[PointAttribute.Z][:] = z
    nube = PointCloud()
    nube.add_chunk(chunk)
    nube.update_bounds()
    return nube


def entrenar(ruta_archivo: str, test_size: float, n_estimators: int) -> None:
    ruta = Path(ruta_archivo)
    print(f"\n=== {ruta.name} ===")

    reader = _abrir_lector(ruta)
    nube = reader.read()
    print(f"Puntos leidos: {nube.point_count:,}")

    x_todo, y_todo, z_todo, etiquetas_todo = [], [], [], []
    for chunk in nube:
        if PointAttribute.CLASSIFICATION not in chunk.attributes:
            print(
                f"\nEste archivo ({ruta.suffix}) no trae un campo de clasificacion real -- "
                f"no se puede usar para entrenar. Los archivos .e57, en particular, nunca lo "
                f"traen (confirmado: solo X/Y/Z). Usa un .las/.laz ya clasificado (verifica "
                f"primero con verificar_clasificacion.py)."
            )
            return
        x_todo.extend(chunk[PointAttribute.X])
        y_todo.extend(chunk[PointAttribute.Y])
        z_todo.extend(chunk[PointAttribute.Z])
        etiquetas_todo.extend(chunk[PointAttribute.CLASSIFICATION])

    x_todo = np.array(x_todo)
    y_todo = np.array(y_todo)
    z_todo = np.array(z_todo)
    etiquetas_todo = np.array(etiquetas_todo, dtype=np.int64)

    clases_reales, conteos = np.unique(etiquetas_todo, return_counts=True)

    if len(clases_reales) < 2:
        print(
            f"\nEste archivo solo trae {len(clases_reales)} clase real -- no hay nada que "
            f"aprender (se necesitan al menos 2 clases distintas para entrenar un clasificador)."
        )
        return

    print("Clases reales encontradas en el archivo:")
    for clase, conteo in zip(clases_reales, conteos):
        try:
            nombre = PointClassification(int(clase)).label
        except ValueError:
            nombre = "(codigo no estandar)"
        print(f"  {int(clase):3} ({nombre}): {conteo:,} puntos")

    rng = np.random.default_rng(seed=42)
    n = len(etiquetas_todo)
    indices = rng.permutation(n)
    corte = int(n * (1 - test_size))
    idx_train, idx_test = indices[:corte], indices[corte:]

    print(f"\nEntrenamiento: {len(idx_train):,} puntos  Prueba: {len(idx_test):,} puntos")

    nube_train = _construir_nube(x_todo[idx_train], y_todo[idx_train], z_todo[idx_train])
    nube_test = _construir_nube(x_todo[idx_test], y_todo[idx_test], z_todo[idx_test])
    etiquetas_train = etiquetas_todo[idx_train]
    etiquetas_test_real = etiquetas_todo[idx_test]

    print(f"\nEntrenando Random Forest (n_estimators={n_estimators})...")
    clasificador = RandomForestClassifier(n_estimators=n_estimators)
    clasificador.fit(nube_train, etiquetas_train)
    print("Entrenamiento completo.")

    resultado = clasificador.classify(nube_test)
    etiquetas_predichas = resultado.labels

    exactitud_global = float(np.mean(etiquetas_predichas == etiquetas_test_real))
    print("\n=== Resultado real de evaluacion ===")
    print(f"Exactitud global: {exactitud_global * 100:.2f}%")

    print("\nExactitud por clase real:")
    for clase in clases_reales:
        mascara_clase = etiquetas_test_real == clase
        if mascara_clase.sum() == 0:
            continue
        acierto_clase = float(np.mean(etiquetas_predichas[mascara_clase] == clase))
        try:
            nombre = PointClassification(int(clase)).label
        except ValueError:
            nombre = "(codigo no estandar)"
        print(
            f"  {int(clase):3} ({nombre}): {acierto_clase * 100:.2f}%  ({int(mascara_clase.sum()):,} puntos de prueba)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archivo", help="Ruta al archivo .las/.laz/.e57 YA ETIQUETADO (con 2+ clases reales)")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    entrenar(args.archivo, args.test_size, args.n_estimators)
