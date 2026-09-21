"""
Flujo completo de nube de puntos en TopoCore, con soporte para
.las, .laz Y .e57 -- exportacion DXF rica (etiquetas de elevacion
reales), GeoTIFF, vista PNG.

    .las/.laz/.e57 -> clasificar terreno (CSF) -> adelgazar (voxel)
    -> TIN -> DTM (+ GeoTIFF si hay CRS) -> curvas de nivel ->
    DXF con etiquetas de elevacion

Uso:
    python flujo_nube_completo_e57.py archivo.e57 [opciones]

Opciones: iguales a flujo_nube_completo_dxf_rico.py (--cloth-resolution,
--voxel-size, --contour-interval, --output-dir, --epsg).
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
from topocore.io.e57.reader import E57Reader
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


def _abrir_lector(ruta: Path):
    """Elige el lector real segun la extension -- .las, .laz, o .e57."""
    extension = ruta.suffix.lower()
    if extension == ".las":
        return LASReader(ruta)
    if extension == ".laz":
        return LAZReader(ruta)
    if extension == ".e57":
        return E57Reader(ruta, chunk_size=1_000_000)
    raise ValueError(f"Extension no soportada: '{extension}'. Se esperaba .las, .laz o .e57.")


def _extraer_puntos3d(cloud) -> tuple[Point3D, ...]:
    """
    Recorre la nube POR CHUNK (streaming) y arma tuplas Point3D,
    deduplicando por XY (conserva el Z mas bajo -- convencion de
    superficie de terreno) antes de devolver. Confirmado necesario:
    VoxelSampler agrupa en 3D completo, no en una rejilla 2D.
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

    reader = _abrir_lector(ruta)
    nube = reader.read()
    print(f"Puntos leidos: {nube.point_count:,}  CRS: {nube.crs}")

    gm = GroundManager(method="csf", csf_cloth_resolution=cloth_resolution)
    terreno = gm.extract(nube)
    print(f"Puntos de terreno (CSF, cloth_resolution={cloth_resolution}): {terreno.point_count:,}")

    adelgazado = VoxelSampler(voxel_size=voxel_size, method="centroid").sample(terreno)
    print(f"Puntos tras adelgazar (voxel_size={voxel_size}): {adelgazado.point_count:,}")

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
            f"(crs={nube.crs!r}). Si conoces el EPSG real, pasalo con --epsg. "
            f"Los archivos .e57 tipicamente NO traen CRS embebido (escaneres terrestres "
            f"suelen usar un sistema de coordenadas local propio del instrumento)."
        )

    import matplotlib.pyplot as plt

    ruta_png = output_dir / f"{ruta.stem}_dtm_vista.png"
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        imagen = ax.imshow(dtm.array(), cmap="Spectral", origin="lower")
        fig.colorbar(imagen, ax=ax, label="Elevacion (m)")
        ax.set_title(f"DTM -- {ruta.name}")
        fig.savefig(ruta_png, dpi=150)
        plt.close(fig)
        print(f"Vista rapida del DTM (PNG, sin GDAL): {ruta_png}")
    except OSError as exc:
        # Confirmado necesario: un fallo real al escribir el PNG (ej.
        # Errno 22 en Windows, visto con datos reales) no debe tumbar
        # el resto del flujo -- las curvas y el DXF siguen siendo
        # utiles aunque la vista rapida no se haya podido guardar.
        print(f"Aviso: vista PNG del DTM omitida -- {exc}")

    contornos = ContourGenerator(tin).generate(interval=contour_interval, base=0.0)
    print(f"Contornos: {len(contornos)} (intervalo {contour_interval} m)")

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
    parser.add_argument("archivo", help="Ruta al archivo .las/.laz/.e57")
    parser.add_argument("--cloth-resolution", type=float, default=0.5)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--contour-interval", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nube"))
    parser.add_argument("--epsg", type=int, default=None)
    args = parser.parse_args()

    procesar(
        args.archivo,
        cloth_resolution=args.cloth_resolution,
        voxel_size=args.voxel_size,
        contour_interval=args.contour_interval,
        output_dir=args.output_dir,
        epsg_forzado=args.epsg,
    )
