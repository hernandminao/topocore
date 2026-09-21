"""
Diagnostico real de escala para TopoCore -- conteo de puntos y RAM
real usada al leer y clasificar terreno (CSF), para 1 o mas
archivos LAS/LAZ.

Requiere psutil para medir RAM real del proceso (memoria total,
incluida la de extensiones C como el binding de CSF -- tracemalloc
NO la capturaria):
    pip install psutil

Uso:
    python diagnostico_escala.py [--cloth-resolution N] archivo1.las archivo2.laz ...

--cloth-resolution (por defecto 0.5, en metros): tamano de celda de
la malla de simulacion de CSF. Confirmado con datos reales: el
costo de CSF depende del AREA GEOGRAFICA cubierta por el archivo
(dividida por este valor al cuadrado), no de la cantidad de puntos
-- un archivo con poca densidad pero mucha extension (ej. Autzen)
puede necesitar una malla varios ordenes de magnitud mas grande que
un archivo denso pero de area pequena (ej. un corredor de via
angosto). Si el proceso se cuelga o tarda demasiado en "Configuring
cloth...", sube este valor (2.0, 5.0) para archivos de area grande.
"""

import argparse
from pathlib import Path

import psutil

from topocore.io.las.reader import LASReader
from topocore.io.laz.reader import LAZReader
from topocore.processing.ground.csf import CSFGroundClassifier

proceso = psutil.Process()


def ram_actual_mb() -> float:
    return proceso.memory_info().rss / (1024 * 1024)


def diagnosticar(ruta: str, cloth_resolution: float) -> None:
    ruta = Path(ruta)
    print(f"\n=== {ruta.name} ({ruta.stat().st_size / (1024 * 1024):.1f} MB en disco) ===")

    ram_antes_lectura = ram_actual_mb()

    reader = LASReader(ruta) if ruta.suffix.lower() == ".las" else LAZReader(ruta)
    nube = reader.read()

    ram_despues_lectura = ram_actual_mb()

    print(f"Puntos reales: {nube.point_count:,}")
    print(f"RAM tras leer: {ram_despues_lectura - ram_antes_lectura:.1f} MB")
    print(
        f"  (bytes por punto, solo lectura: {(ram_despues_lectura - ram_antes_lectura) * 1024 * 1024 / nube.point_count:.1f})"
    )

    ram_antes_csf = ram_actual_mb()
    try:
        from topocore.processing.exceptions import GroundError

        clasificador = CSFGroundClassifier(cloth_resolution=cloth_resolution)
        clasificador.classify(nube)
        ram_despues_csf = ram_actual_mb()
        print(
            f"RAM adicional para clasificar (CSF, cloth_resolution={cloth_resolution}): {ram_despues_csf - ram_antes_csf:.1f} MB"
        )
        print(f"RAM TOTAL del proceso en el pico: {ram_despues_csf:.1f} MB")
    except GroundError as e:
        print(f"Clasificacion CSF fallo (error real de TopoCore): {e}")
    except MemoryError:
        print("Clasificacion CSF fallo -- RAM insuficiente para este archivo (MemoryError real).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archivos", nargs="+", help="Rutas a archivos .las/.laz")
    parser.add_argument(
        "--cloth-resolution",
        type=float,
        default=0.5,
        help="Tamano de celda de la malla CSF, en metros (default 0.5). Sube este valor para archivos de area grande.",
    )
    args = parser.parse_args()

    for ruta in args.archivos:
        diagnosticar(ruta, args.cloth_resolution)
