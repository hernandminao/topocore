"""
topocore.analysis.profile.dxf_export -- PROPUESTA, no auditada
todavia con la disciplina completa de PR22.

Dibuja cada ``ProfileResult`` como una linea de terreno real, en un
sistema de coordenadas local por seccion (offset -> X, elevacion ->
Y) -- la convencion estandar de una lamina de seccion transversal.

Alcance V1, deliberado y explicito: SOLO el terreno real (lo que
``ProfileAnalysis().cross_section()`` ya calcula, confirmado con
ejecucion real). NO dibuja:

- superficie de diseno (TopoCore no tiene, hoy, ningun modelo de
  plantilla tipica de via -- ancho de calzada, taludes, bombeo);
- area de corte/relleno (``AverageEndAreaVolume`` ya existe, pero
  recibe el area YA calculada -- no hay, en todo el repositorio,
  ningun calculo real de "area entre 2 perfiles");
- bermas, cunetas, bordillos, ni ninguna otra plantilla de via.

Cada una de esas 3 cosas requiere su propia decision de alcance
antes de construirse -- no se agregan aqui silenciosamente.

Author
------
Hernán Mina

License
-------
MIT
"""
from __future__ import annotations

from pathlib import Path

from topocore.analysis.types import ProfileResult


class CrossSectionDXFExportError(Exception):
    """La exportacion DXF de secciones transversales fallo."""


def export_cross_sections_dxf(
    profiles: list[ProfileResult],
    output_path: str | Path,
    *,
    horizontal_spacing: float = 40.0,
    text_height: float = 1.0,
    layer: str = "TOPO_CROSS_SECTIONS",
    show_endpoint_elevations: bool = True,
    show_lowest_point_elevation: bool = True,
    show_elevation_axis: bool = True,
    elevation_axis_interval: float = 1.0,
) -> Path:
    """
    Dibuja cada seccion en su propio sistema de coordenadas LOCAL
    (offset -> X, elevacion -> Y) -- no en las coordenadas reales del
    terreno, que es la convencion estandar de una lamina de seccion
    transversal (una "vista de perfil", no un plano en planta).

    Las secciones se disponen una junto a otra, separadas por
    ``horizontal_spacing`` (en las mismas unidades que offset/elevacion
    -- metros, por convencion de todo el proyecto), en el orden dado
    en ``profiles``.

    Parameters
    ----------
    profiles
        Tipicamente el resultado de
        ``ProfileAnalysis().cross_section(...)``.
    output_path
        Ruta del archivo ``.dxf`` a escribir.
    horizontal_spacing
        Separacion horizontal entre el borde derecho de una seccion y
        el borde izquierdo de la siguiente.
    text_height
        Altura del texto de la etiqueta de estacion.
    layer
        Capa DXF para toda la geometria y las etiquetas.
    show_endpoint_elevations
        Si escribe el valor numerico de elevacion junto a cada
        extremo (izquierdo y derecho) de la seccion.
    show_lowest_point_elevation
        Si marca y etiqueta el punto mas bajo de cada seccion, con su
        valor real de elevacion -- el dato mas relevante para
        analisis de drenaje.
    show_elevation_axis
        Si dibuja una regla vertical de referencia a la izquierda de
        cada seccion, con marcas y valores de elevacion cada
        ``elevation_axis_interval`` metros.
    elevation_axis_interval
        Intervalo de las marcas de la regla vertical, en metros de
        elevacion. Solo aplica si ``show_elevation_axis`` es True.

    Raises
    ------
    CrossSectionDXFExportError
        Si ``ezdxf`` no esta instalado, si una seccion tiene menos de
        2 puntos (no forma una linea valida), o si la escritura falla.
    """
    import math

    from topocore.dxf._ezdxf_compat import require_ezdxf
    from topocore.dxf.exceptions import DXFExportError

    try:
        ezdxf = require_ezdxf()
    except DXFExportError as exc:
        raise CrossSectionDXFExportError(str(exc)) from exc

    if horizontal_spacing <= 0:
        raise CrossSectionDXFExportError("horizontal_spacing must be positive.")
    if text_height <= 0:
        raise CrossSectionDXFExportError("text_height must be positive.")
    if elevation_axis_interval <= 0:
        raise CrossSectionDXFExportError("elevation_axis_interval must be positive.")

    doc = ezdxf.new(setup=True)
    if layer not in doc.layers:
        doc.layers.add(layer)
    msp = doc.modelspace()

    cursor_x = 0.0
    for profile in profiles:
        if len(profile.points) < 2:
            raise CrossSectionDXFExportError(
                f"A cross-section with fewer than 2 points cannot be drawn as a line "
                f"(station={profile.points[0].station if profile.points else 'unknown'})."
            )

        offsets = [p.offset for p in profile.points]
        min_offset = min(offsets)
        section_width = max(offsets) - min_offset

        vertices = [(cursor_x + (p.offset - min_offset), p.z) for p in profile.points]
        msp.add_lwpolyline(vertices, dxfattribs={"layer": layer})

        elevaciones = [p.z for p in profile.points]
        elevacion_min = min(elevaciones)
        elevacion_max = max(elevaciones)

        if show_endpoint_elevations:
            for punto, (x_local, z_local) in ((profile.points[0], vertices[0]), (profile.points[-1], vertices[-1])):
                texto_elev = msp.add_text(f"{z_local:.2f}", dxfattribs={"layer": layer, "height": text_height * 0.7})
                texto_elev.set_placement((x_local, z_local + text_height * 0.3))

        if show_lowest_point_elevation:
            indice_mas_bajo = min(range(len(profile.points)), key=lambda i: profile.points[i].z)
            x_mas_bajo, z_mas_bajo = vertices[indice_mas_bajo]
            msp.add_circle((x_mas_bajo, z_mas_bajo), radius=text_height * 0.15, dxfattribs={"layer": layer})
            texto_bajo = msp.add_text(f"{z_mas_bajo:.2f}", dxfattribs={"layer": layer, "height": text_height * 0.8})
            texto_bajo.set_placement((x_mas_bajo, z_mas_bajo - text_height * 1.8))

        if show_elevation_axis:
            eje_x = cursor_x - text_height * 1.5
            msp.add_line((eje_x, elevacion_min), (eje_x, elevacion_max), dxfattribs={"layer": layer})

            cota_actual = math.ceil(elevacion_min / elevation_axis_interval) * elevation_axis_interval
            while cota_actual <= elevacion_max:
                msp.add_line(
                    (eje_x - text_height * 0.3, cota_actual), (eje_x + text_height * 0.3, cota_actual),
                    dxfattribs={"layer": layer},
                )
                texto_cota = msp.add_text(f"{cota_actual:.2f}", dxfattribs={"layer": layer, "height": text_height * 0.6})
                texto_cota.set_placement((eje_x - text_height * 3.5, cota_actual - text_height * 0.3))
                cota_actual += elevation_axis_interval

        station_label = f"K{profile.points[0].station:.2f}"
        text = msp.add_text(station_label, dxfattribs={"layer": layer, "height": text_height})
        text.set_placement((cursor_x, elevacion_min - text_height * 2))

        cursor_x += section_width + horizontal_spacing

    try:
        doc.saveas(str(output_path))
    except OSError as exc:
        raise CrossSectionDXFExportError(f"Could not write DXF to '{output_path}': {exc}") from exc

    return Path(output_path)


__all__ = ["CrossSectionDXFExportError", "export_cross_sections_dxf"]
