"""
Clasifica terreno con PMF y guarda un archivo .laz NUEVO con la
clasificacion real escrita en el campo CLASSIFICATION.

Usa PMF, no CSF -- confirmado con el benchmark real ISPRS
(VALIDATION_ISPRS.md) que PMF supera consistentemente a CSF en los
3 sitios reales probados, y es ademas nativo de TopoCore (sin
dependencia externa opcional), corriendo mas rapido en general.

Confirmado con ejecucion real: Chunk no tiene un metodo publico para
modificar un atributo existente -- se clona el chunk original (mismo
patron que usa Chunk.clone() internamente) y se reemplaza su
CLASSIFICATION directamente.

Uso:
    python clasificar_y_guardar.py archivo.laz [--cell-size N] [--output archivo_clasificado.laz]
"""
import argparse
from pathlib import Path

import numpy as np
from topocore.io.las.reader import LASReader
from topocore.io.laz.reader import LAZReader
from topocore.io.laz.writer import LAZWriter
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.classification import PointClassification
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import GroundError
from topocore.processing.ground.pmf import PMFGroundClassifier


def clasificar_y_guardar(ruta_archivo: str, cell_size: float, max_grid_cells: int, ruta_salida: Path) -> None:
    ruta = Path(ruta_archivo)
    print(f"\n=== {ruta.name} ===")

    reader = LASReader(ruta) if ruta.suffix.lower() == ".las" else LAZReader(ruta)
    nube = reader.read()
    print(f"Puntos leidos: {nube.point_count:,}")

    clasificador = PMFGroundClassifier(cell_size=cell_size, max_grid_cells=max_grid_cells)
    try:
        es_terreno = clasificador.classify(nube)
    except GroundError as exc:
        print(f"\nNo se pudo clasificar: {exc}")
        print("Ajusta --cell-size al valor sugerido arriba, o sube --max-grid-cells si prefieres mantenerlo.")
        return
    print(f"Terreno detectado (PMF, cell_size={cell_size}): {es_terreno.sum():,} de {nube.point_count:,} puntos")

    # Construir una nube nueva, clonando cada chunk original y
    # reemplazando su CLASSIFICATION con el resultado real de PMF --
    # confirmado, GROUND(2) donde PMF detecto terreno,
    # UNCLASSIFIED(1) donde no. Reemplaza cualquier clasificacion
    # previa del archivo original por completo.
    nube_clasificada = PointCloud()
    offset = 0
    for chunk_original in nube:
        n = chunk_original.size
        chunk_nuevo = chunk_original.clone()
        codigos = np.where(
            es_terreno[offset : offset + n],
            PointClassification.GROUND.value,
            PointClassification.UNCLASSIFIED.value,
        ).astype(np.uint8)
        chunk_nuevo._data[PointAttribute.CLASSIFICATION][:] = codigos
        nube_clasificada.add_chunk(chunk_nuevo)
        offset += n

    nube_clasificada.update_bounds()

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    LAZWriter(ruta_salida).write(nube_clasificada)
    print(f"Archivo clasificado guardado: {ruta_salida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archivo", help="Ruta al archivo .las/.laz de entrada")
    parser.add_argument("--cell-size", type=float, default=0.5)
    parser.add_argument(
        "--max-grid-cells", type=int, default=8_000_000,
        help="Limite de seguridad de PMF -- sube este valor si prefieres mantener un cell_size "
             "fino en vez de aumentarlo (el area cubierta define cuantas celdas se necesitan).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Ruta de salida (default: <nombre>_pmf.laz)")
    args = parser.parse_args()

    ruta_entrada = Path(args.archivo)
    salida = args.output or ruta_entrada.with_name(f"{ruta_entrada.stem}_pmf.laz")

    clasificar_y_guardar(args.archivo, args.cell_size, args.max_grid_cells, salida)
