"""
topocore.io.raster.geotiff -- PROPUESTA, no auditada todavia con la
disciplina completa de PR22.

Escribe un ``topocore.terrain.DTM`` como un GeoTIFF real. TopoCore
no tenia, hasta este modulo, ningun escritor de GeoTIFF propio --
confirmado con busqueda exhaustiva en todo el repositorio: solo
existian LECTORES de GeoTIFF (para geoides, en
``topocore.geodesy.vertical.geoid_grid``), nunca un escritor.

GDAL se importa de forma perezosa, exactamente con el mismo patron
ya establecido en ``geoid_grid.py`` -- confirmado, misma dependencia
opcional (extra ``geoid``), mismo mensaje de instalacion, mismo
manejo de ``ImportError``.

Author
------
Hernán Mina

License
-------
MIT
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from topocore.terrain.dtm import DTM


class GeoTIFFWriteError(Exception):
    """La escritura del GeoTIFF fallo (GDAL ausente, ruta invalida, etc.)."""


def write_dtm_geotiff(
    dtm: DTM,
    output_path: str | Path,
    *,
    crs_wkt: str,
    nodata: float = -9999.0,
) -> Path:
    """
    Escribe ``dtm`` como un GeoTIFF de una sola banda (float64).

    Parameters
    ----------
    dtm
        El DTM ya construido (ej. via ``DTM.from_tin(...)``).
    output_path
        Ruta del archivo ``.tif`` a escribir.
    crs_wkt
        El CRS real del DTM, como WKT -- confirmado, se obtiene con
        ``CRS.from_epsg(...).to_wkt()`` (``topocore.geodesy.crs``).
        Este modulo no asume ningun EPSG por defecto -- un DTM sin
        CRS conocido no deberia escribirse con una proyeccion
        inventada.
    nodata
        Valor de nodata del raster. Por defecto ``-9999.0`` --
        confirmado, DTM.array() no expone actualmente que celdas son
        nodata reales vs. interpoladas, asi que este valor solo se
        usa como metadato del raster (SetNoDataValue), no para
        marcar celdas reales como nodata en los datos escritos.

    Raises
    ------
    GeoTIFFWriteError
        Si GDAL no esta instalado, o si la escritura falla (ruta
        invalida, permisos, disco lleno, etc.).
    """
    try:
        from osgeo import gdal, osr  # type: ignore
    except ImportError as exc:
        raise GeoTIFFWriteError(
            "Writing a GeoTIFF requires GDAL, which is not installed. "
            "Install it with `pip install topocore[geoid]` "
            "(matching your system's libgdal version)."
        ) from exc

    output_path = Path(output_path)
    min_x, _min_y, _max_x, max_y = dtm.bounds

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(output_path), dtm.columns, dtm.rows, 1, gdal.GDT_Float64)
    if dataset is None:
        raise GeoTIFFWriteError(f"GDAL could not create '{output_path}' -- check the path and permissions.")

    try:
        # GDAL espera el origen en la esquina SUPERIOR-izquierda con
        # un pixel_height NEGATIVO -- confirmado, DTM.array() entrega
        # sus filas de abajo hacia arriba (origen inferior-izquierdo,
        # convencion matematica/geografica estandar), asi que el
        # array se invierte verticalmente antes de escribirlo.
        dataset.SetGeoTransform((min_x, dtm.resolution, 0.0, max_y, 0.0, -dtm.resolution))

        srs = osr.SpatialReference()
        srs.ImportFromWkt(crs_wkt)
        dataset.SetProjection(srs.ExportToWkt())

        band = dataset.GetRasterBand(1)
        band.WriteArray(np.flipud(dtm.array()))
        band.SetNoDataValue(nodata)
        dataset.FlushCache()
    except RuntimeError as exc:
        raise GeoTIFFWriteError(f"GDAL failed while writing '{output_path}': {exc}") from exc
    finally:
        dataset = None  # cierra/flushea el dataset GDAL, patron estandar de gdal-python

    return output_path


__all__ = ["GeoTIFFWriteError", "write_dtm_geotiff"]
