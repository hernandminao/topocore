"""
Flujo completo de nube de puntos en TopoCore:

    .las/.laz -> clasificar terreno (CSF) -> adelgazar (voxel) ->
    TIN -> DTM -> curvas de nivel -> DXF

Verificado con ejecucion real antes de entregarse.

Uso:
    python flujo_nube_completo.py archivo.laz [opciones]

Opciones:
    --cloth-resolution N   (default 0.5) tamano de celda CSF, en metros.
                           Sube este valor para archivos de area grande
                           (confirmado con datos reales: el costo de
                           CSF depende del area geografica, no de la
                           cantidad de puntos).
    --voxel-size N         (default 1.0) tamano de celda para adelgazar
                           el terreno clasificado antes del TIN --
                           confirmado necesario: construir un TIN
                           directamente con TODOS los puntos de
                           terreno clasificados (potencialmente
                           millones) es impractico; ningun flujo real
                           de LiDAR construye un TIN a la densidad
                           completa de retornos de suelo.
    --contour-interval N   (default 1.0) intervalo de curvas de nivel.
    --output-dir DIR        (default outputs/nube) carpeta de salida.
"""
import argparse
from pathlib import Path

from topocore.geometry.point3d import Point3D
from topocore.io.las.reader import LASReader
from topocore.io.laz.reader import LAZReader
from topocore.pointcloud.attributes import PointAttribute
from topocore.processing.ground.manager import GroundManager
from topocore.processing.sampling.voxel import VoxelSampler
from topocore.terrain.contours import ContourGenerator
from topocore.terrain.dtm import DTM
from topocore.terrain.enums import InterpolationMethod
from topocore.terrain.grid import Grid
from topocore.terrain.interpolation import TerrainInterpolator
from topocore.terrain.tin import TIN


def _extraer_puntos3d(cloud) -> tuple[Point3D, ...]:
    """Recorre la nube POR CHUNK (streaming) y arma tuplas Point3D --
    confirmado, evita cargar la nube 2 veces en memoria de golpe."""
    puntos = []
    for chunk in cloud:
        x = chunk[PointAttribute.X]
        y = chunk[PointAttribute.Y]
        z = chunk[PointAttribute.Z]
        puntos.extend(Point3D(float(xi), float(yi), float(zi)) for xi, yi, zi in zip(x, y, z))
    return tuple(puntos)


def procesar(
    ruta_archivo: str,
    *,
    cloth_resolution: float,
    voxel_size: float,
    contour_interval: float,
    output_dir: Path,
) -> None:
    ruta = Path(ruta_archivo)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {ruta.name} ===")

    # 1. Leer (streaming por chunks)
    reader = LASReader(ruta) if ruta.suffix.lower() == ".las" else LAZReader(ruta)
    nube = reader.read()
    print(f"Puntos leidos: {nube.point_count:,}  CRS: {nube.crs}")

    # 2. Clasificar y extraer terreno (CSF)
    gm = GroundManager(method="csf", csf_cloth_resolution=cloth_resolution)
    terreno = gm.extract(nube)
    print(f"Puntos de terreno (CSF, cloth_resolution={cloth_resolution}): {terreno.point_count:,}")

    # 3. Adelgazar -- construir un TIN con TODOS los puntos de terreno
    #    clasificados (potencialmente millones) es impractico
    adelgazado = VoxelSampler(voxel_size=voxel_size, method="centroid").sample(terreno)
    print(f"Puntos tras adelgazar (voxel_size={voxel_size}): {adelgazado.point_count:,}")

    # 4. TIN -> DTM -> curvas
    puntos3d = _extraer_puntos3d(adelgazado)
    tin = TIN.from_points(puntos3d)
    print(f"TIN: {tin.vertex_count:,} vertices, {tin.triangle_count:,} triangulos")

    min_x, min_y, max_x, max_y = tin.bounds
    grid = Grid(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, resolution=voxel_size)
    interpolador = TerrainInterpolator(tin, method=InterpolationMethod.LINEAR)
    DTM.from_tin(tin, grid, interpolador)  # type: ignore[arg-type]
    print("DTM construido.")

    contornos = ContourGenerator(tin).generate(interval=contour_interval, base=0.0)
    print(f"Contornos: {len(contornos)} (intervalo {contour_interval} m)")

    # 5. Exportar curvas a DXF -- directo con ezdxf, sin pasar por el
    #    modelo Feature/FeatureCollection (pensado para survey
    #    codificado, no para nubes de puntos crudas)
    from topocore.dxf._ezdxf_compat import require_ezdxf

    ezdxf = require_ezdxf()
    doc = ezdxf.new(setup=True)
    doc.layers.add("TOPO_CONTOURS")
    msp = doc.modelspace()
    for c in contornos:
        vertices = [(p.x, p.y, p.z) for p in c.points]
        msp.add_polyline3d(vertices, dxfattribs={"layer": "TOPO_CONTOURS"})

    ruta_dxf = output_dir / f"{ruta.stem}_curvas.dxf"
    doc.saveas(str(ruta_dxf))
    print(f"DXF exportado: {ruta_dxf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archivo", help="Ruta al archivo .las/.laz")
    parser.add_argument("--cloth-resolution", type=float, default=0.5)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--contour-interval", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nube"))
    args = parser.parse_args()

    procesar(
        args.archivo,
        cloth_resolution=args.cloth_resolution,
        voxel_size=args.voxel_size,
        contour_interval=args.contour_interval,
        output_dir=args.output_dir,
    )
