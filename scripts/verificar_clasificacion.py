"""
Verifica si un archivo LAS/LAZ ya trae clasificacion ASPRS real
embebida (no solo ceros/sin clasificar), y muestra cuantos puntos
hay de cada clase real.

Uso:
    python verificar_clasificacion.py archivo.laz
"""

import sys
from collections import Counter
from pathlib import Path

from topocore.io.las.reader import LASReader
from topocore.io.laz.reader import LAZReader
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.classification import PointClassification


def verificar(ruta_archivo: str) -> None:
    ruta = Path(ruta_archivo)
    print(f"\n=== {ruta.name} ===")

    reader = LASReader(ruta) if ruta.suffix.lower() == ".las" else LAZReader(ruta)
    nube = reader.read()

    conteo: Counter[int] = Counter()
    for chunk in nube:
        for codigo in chunk[PointAttribute.CLASSIFICATION]:
            conteo[int(codigo)] += 1

    if len(conteo) == 1 and 0 in conteo:
        print("Este archivo NO viene clasificado -- todos los puntos son codigo 0 (sin clasificar).")
        return

    print("Este archivo SI trae clasificacion real. Distribucion encontrada:")
    for codigo, cantidad in sorted(conteo.items()):
        try:
            nombre = PointClassification(codigo).label
        except ValueError:
            nombre = "(codigo no estandar / especifico del proveedor)"
        print(f"  codigo {codigo:2} ({nombre}): {cantidad:,} puntos")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python verificar_clasificacion.py archivo.laz")
        sys.exit(1)
    verificar(sys.argv[1])
