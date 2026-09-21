"""
Flujo completo de nube de puntos en TopoCore, con exportacion DXF
RICA (etiquetas de elevacion reales en cada curva, capas correctas)
-- reutiliza el mismo DXFExporter ya validado para el flujo de
survey, en vez de un DXF minimo hecho directo con ezdxf.

    .las/.laz -> clasificar terreno (CSF) -> adelgazar (voxel) ->
    TIN -> DTM (+ GeoTIFF si hay CRS) -> curvas de nivel ->
    DXF con etiquetas de elevacion

Uso:
    python flujo_nube_completo_dxf_rico.py archivo.laz [opciones]

Opciones: iguales a flujo_nube_completo.py (--cloth-resolution,
--voxel-size, --contour-interval, --output-dir), mas:
    --point-labels   si tambien etiqueta puntos individuales (no
                     aplica aqui, ninguna feature puntual se genera
                     desde una nube de puntos cruda -- se deja el
                     flag por consistencia con SurveyPipeline).
"""

import argparse
from pathlib import Path

import numpy as np
from topocore.dxf import DXFExporter
from topocore.dxf.models import DXFExportOptions, ExportContext
from topocore.features.models import (
    Feature,
    FeatureCategory,
    FeatureCollection,
    FeatureGeometry,
    FeatureType,
    GeometryType,
)
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
    """
    Recorre la nube POR CHUNK (streaming) y arma tuplas Point3D,
    deduplicando por XY (conserva el Z mas bajo -- convencion de
    superficie de terreno) antes de devolver.

    Confirmado necesario con datos reales: VoxelSampler agrupa en 3D
    completo (X, Y, Z), no en una rejilla 2D -- para una superficie
    de terreno con variacion vertical real (pendiente, micro-relieve),
    2 voxels de distinta altura dentro de la misma columna horizontal
    pueden promediar a coordenadas XY identicas. TIN.from_points()
    rechaza XY duplicada (TriangulationError), correctamente -- un
    TIN no puede tener 2 alturas para el mismo punto en planta.
    """
    xy_a_z: dict[tuple[float, float], float] = {}
    for chunk in cloud:
        x = chunk[PointAttribute.X]
        y = chunk[PointAttribute.Y]
        z = chunk[PointAttribute.Z]
        for xi, yi, zi in zip(x, y, z):
            clave = (float(xi), float(yi))
            actual = xy_a_z.get(clave)
            if actual is None or zi < actual:
                xy_a_z[clave] = float(zi)

    return tuple(Point3D(x, y, z) for (x, y), z in xy_a_z.items())


def procesar(
    ruta_archivo: str,
    *,
    cloth_resolution: float,
    voxel_size: float,
    contour_interval: float,
    output_dir: Path,
    epsg_forzado: int | None = None,
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

    # 3. Adelgazar antes del TIN
    adelgazado = VoxelSampler(voxel_size=voxel_size, method="centroid").sample(terreno)
    print(f"Puntos tras adelgazar (voxel_size={voxel_size}): {adelgazado.point_count:,}")

    # 4. TIN -> DTM -> curvas
    puntos3d = _extraer_puntos3d(adelgazado)
    if len(puntos3d) < adelgazado.point_count:
        print(
            f"Aviso: {adelgazado.point_count - len(puntos3d)} punto(s) con XY duplicada "
            f"deduplicados (se conservo el Z mas bajo de cada par)."
        )
    tin = TIN.from_points(puntos3d)
    print(f"TIN: {tin.vertex_count:,} vertices, {tin.triangle_count:,} triangulos")

    min_x, min_y, max_x, max_y = tin.bounds
    grid = Grid(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, resolution=voxel_size)
    interpolador = TerrainInterpolator(tin, method=InterpolationMethod.LINEAR)
    dtm = DTM.from_tin(tin, grid, interpolador)  # type: ignore[arg-type]
    print("DTM construido.")

    if nube.crs is not None and nube.crs.startswith("EPSG:"):
        from topocore.geodesy.crs import CRS
        from topocore.io.raster.geotiff import GeoTIFFWriteError, write_dtm_geotiff

        epsg = int(nube.crs.split(":")[1])
        ruta_geotiff = output_dir / f"{ruta.stem}_dtm.tif"
        try:
            write_dtm_geotiff(dtm, ruta_geotiff, crs_wkt=CRS.from_epsg(epsg).to_wkt())
            print(f"GeoTIFF exportado: {ruta_geotiff}")
        except GeoTIFFWriteError as exc:
            print(f"Aviso: GeoTIFF omitido -- {exc}")
    elif epsg_forzado is not None:
        from topocore.geodesy.crs import CRS
        from topocore.io.raster.geotiff import GeoTIFFWriteError, write_dtm_geotiff

        ruta_geotiff = output_dir / f"{ruta.stem}_dtm.tif"
        try:
            write_dtm_geotiff(dtm, ruta_geotiff, crs_wkt=CRS.from_epsg(epsg_forzado).to_wkt())
            print(f"GeoTIFF exportado (EPSG:{epsg_forzado} forzado manualmente): {ruta_geotiff}")
        except GeoTIFFWriteError as exc:
            print(f"Aviso: GeoTIFF omitido -- {exc}")
    else:
        print(
            f"GeoTIFF omitido -- la nube trae un CRS sin EPSG detectado automaticamente "
            f"(crs={nube.crs!r}). Si conoces el EPSG real, pasalo con --epsg."
        )

    # 4c. Vista rapida del DTM como PNG -- no necesita GDAL ni QGIS,
    #     solo abrir el archivo. Confirmado util para "ver" el DTM
    #     cuando GDAL no esta instalado (el caso real mas comun).
    import matplotlib.pyplot as plt

    ruta_png = output_dir / f"{ruta.stem}_dtm_vista.png"
    fig, ax = plt.subplots(figsize=(8, 8))
    imagen = ax.imshow(dtm.array(), cmap="Spectral", origin="lower")
    fig.colorbar(imagen, ax=ax, label="Elevacion (m)")
    ax.set_title(f"DTM -- {ruta.name}")
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Vista rapida del DTM (PNG, sin GDAL): {ruta_png}")

    contornos = ContourGenerator(tin).generate(interval=contour_interval, base=0.0)
    print(f"Contornos: {len(contornos)} (intervalo {contour_interval} m)")

    # 5. Exportar a DXF -- exportador RICO (el mismo de SurveyPipeline):
    #    cada curva se convierte en un Feature real con su elevacion
    #    como atributo, que DXFExporter dibuja como etiqueta de texto
    #    visible junto a la curva (confirmado, no solo geometria).
    coleccion = FeatureCollection()
    for i, c in enumerate(contornos):
        vertices = np.array([[p.x, p.y, p.z] for p in c.points])
        geom = FeatureGeometry(geometry_type=GeometryType.POLYLINE, vertices=vertices, closed=c.closed)
        curva = Feature(
            feature_id=100000 + i,
            category=FeatureCategory.TERRAIN,
            feature_type=FeatureType.CONTOUR,
            geometry=geom,
            attributes={"elevation": c.elevation},
        )
        coleccion.add(curva)

    ruta_dxf = output_dir / f"{ruta.stem}_curvas.dxf"
    DXFExporter(ExportContext(options=DXFExportOptions(contour_labels=True))).export(coleccion, ruta_dxf)
    print(f"DXF exportado (con etiquetas de elevacion): {ruta_dxf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archivo", help="Ruta al archivo .las/.laz")
    parser.add_argument("--cloth-resolution", type=float, default=0.5)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--contour-interval", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nube"))
    parser.add_argument(
        "--epsg",
        type=int,
        default=None,
        help="Fuerza el EPSG del GeoTIFF cuando la nube trae un CRS que no se detecta "
        "automaticamente como 'EPSG:XXXX' (ej. un CRS compuesto horizontal+vertical).",
    )
    args = parser.parse_args()

    procesar(
        args.archivo,
        cloth_resolution=args.cloth_resolution,
        voxel_size=args.voxel_size,
        contour_interval=args.contour_interval,
        output_dir=args.output_dir,
        epsg_forzado=args.epsg,
    )
