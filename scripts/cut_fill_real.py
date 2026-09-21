"""
Corte/relleno REAL -- no sintetico: construye el TIN a partir de
terreno real (clasificado con PMF desde una nube de puntos real), y
calcula el volumen de corte/relleno necesario para nivelar ese sitio
real a una cota de diseño real -- el escenario mas comun en la
practica de movimiento de tierras (cuanto material hay que cortar o
rellenar para dejar un sitio a una elevacion objetivo).

Usa TINVolume (volumen bajo/sobre un TIN respecto a un datum), que
opera directamente sobre el TIN real (sin pasar por una rejilla
DTM), evitando cualquier duda sobre discretizacion de grilla.

Uso:
    python cut_fill_real.py archivo.las [--datum N] [--cell-size N]

--datum: cota de diseño real, en metros (si no se da, se usa la
         elevacion PROMEDIO real del terreno clasificado, un valor
         neutro razonable para una primera prueba).
"""
import argparse
from pathlib import Path

from topocore.analysis.volume.tin_volume import TINVolume
from topocore.geometry.point3d import Point3D
from topocore.io.e57.reader import E57Reader
from topocore.io.las.reader import LASReader
from topocore.io.laz.reader import LAZReader
from topocore.pointcloud.attributes import PointAttribute
from topocore.processing.ground.pmf import PMFGroundClassifier
from topocore.terrain.tin import TIN


def _abrir_lector(ruta: Path):
    extension = ruta.suffix.lower()
    if extension == ".las":
        return LASReader(ruta)
    if extension == ".laz":
        return LAZReader(ruta)
    if extension == ".e57":
        return E57Reader(ruta, chunk_size=1_000_000)
    raise ValueError(f"Extension no soportada: '{extension}'.")


def calcular(ruta_archivo: str, cell_size: float, datum: float | None) -> None:
    ruta = Path(ruta_archivo)
    print(f"\n=== {ruta.name} ===")

    reader = _abrir_lector(ruta)
    nube = reader.read()
    print(f"Puntos leidos: {nube.point_count:,}")

    clasificador = PMFGroundClassifier(cell_size=cell_size)
    terreno_mask = clasificador.classify(nube)
    print(f"Puntos de terreno real (PMF): {int(terreno_mask.sum()):,}")

    puntos = []
    for chunk in nube:
        x = chunk[PointAttribute.X]
        y = chunk[PointAttribute.Y]
        z = chunk[PointAttribute.Z]
        puntos.extend(zip(x, y, z))

    puntos_terreno = [Point3D(float(x), float(y), float(z)) for (x, y, z), m in zip(puntos, terreno_mask) if m]

    # Deduplicar por XY (mismo criterio ya verificado antes) -- TIN
    # no acepta coordenadas XY duplicadas.
    xy_a_z: dict[tuple[float, float], float] = {}
    for p in puntos_terreno:
        clave = (p.x, p.y)
        actual = xy_a_z.get(clave)
        if actual is None or p.z < actual:
            xy_a_z[clave] = p.z
    puntos_terreno_dedup = tuple(Point3D(x, y, z) for (x, y), z in xy_a_z.items())

    tin = TIN.from_points(puntos_terreno_dedup)
    print(f"TIN real: {tin.vertex_count:,} vertices, {tin.triangle_count:,} triangulos")

    elevaciones_reales = [p.z for p in puntos_terreno_dedup]
    elevacion_promedio_real = sum(elevaciones_reales) / len(elevaciones_reales)
    cota_diseno = datum if datum is not None else elevacion_promedio_real

    print(
        f"Elevacion real: min={min(elevaciones_reales):.2f}  max={max(elevaciones_reales):.2f}  "
        f"promedio={elevacion_promedio_real:.2f}"
    )
    print(
        f"Cota de diseno usada: {cota_diseno:.2f} m "
        f"{'(promedio real del sitio, no especificada)' if datum is None else '(especificada)'}"
    )

    resultado = TINVolume(datum=cota_diseno).compute(tin)

    print("\n=== Resultado real de corte/relleno ===")
    print(f"Volumen de CORTE (terreno por encima de la cota):    {resultado.cut_volume:,.2f} m3")
    print(f"Volumen de RELLENO (terreno por debajo de la cota):     {resultado.fill_volume:,.2f} m3")
    print(f"Volumen neto:                                              {resultado.net_volume:,.2f} m3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archivo", help="Ruta al archivo .las/.laz/.e57")
    parser.add_argument("--cell-size", type=float, default=1.0, help="cell_size de PMF para clasificar terreno.")
    parser.add_argument("--datum", type=float, default=None, help="Cota de diseno real, en metros.")
    args = parser.parse_args()

    calcular(args.archivo, args.cell_size, args.datum)
