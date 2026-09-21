"""
topocore.analysis.profile.writer -- PROPUESTA, no auditada todavia
con la disciplina completa de PR22.

Escribe una lista de ``ProfileResult`` como un CSV tabulado:
estacion, offset, coordenadas, y elevacion. No se introduce ningun
modelo nuevo -- ``ProfileResult``/``ProfilePoint`` (ya existentes en
``topocore.analysis.types``) representan exactamente este concepto.

Vive junto al propio analisis de perfiles, no bajo ``topocore.io``,
por la misma razon que ``topocore.dxf.annotation`` vive junto al
exporter de DXF que anota: es un pequeno ayudante de exportacion
especifico de este analisis, no un formato de archivo generico que
otros modulos tambien necesiten leer/escribir.

Producto de analisis, no una Feature -- no participa del modelo
Feature/FeatureCollection ni de la exportacion DXF/GPKG a proposito:
una seccion transversal es una tabla de ingenieria, no una geometria
cartografica.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import csv
from pathlib import Path

from topocore.analysis.types import ProfileResult


class ProfileWriteError(Exception):
    """La escritura del CSV de perfiles fallo (ruta invalida, permisos, etc.)."""


def write_profile_csv(profiles: list[ProfileResult], output_path: str | Path) -> Path:
    """
    Escribe ``profiles`` (tipicamente el resultado de
    ``ProfileAnalysis().cross_section(...)``) como un CSV con
    columnas ``station, offset, x, y, elevation``.

    ``ProfilePoint.z`` se mapea a la columna ``elevation`` del CSV
    -- ``elevation`` es mas claro para quien consume el archivo,
    pero el modelo ``ProfilePoint`` en si conserva su propio nombre
    de campo (``z``) sin cambios.

    Cada llamada escribe un archivo con todos los puntos de todos
    los perfiles dados, concatenados en el orden recibido -- no
    agrega ninguna columna que distinga a que perfil pertenece cada
    fila mas alla de ese orden.

    Raises
    ------
    ProfileWriteError
        Si el archivo no puede escribirse (ruta invalida, permisos,
        disco lleno, etc.).
    """
    output_path = Path(output_path)

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(("station", "offset", "x", "y", "elevation"))
            for profile in profiles:
                for point in profile.points:
                    writer.writerow((point.station, point.offset, point.x, point.y, point.z))
    except OSError as exc:
        raise ProfileWriteError(f"Could not write profile CSV to '{output_path}': {exc}") from exc

    return output_path


__all__ = ["ProfileWriteError", "write_profile_csv"]
