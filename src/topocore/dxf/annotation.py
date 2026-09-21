"""
topocore.dxf.annotation -- PROPUESTA, no auditada todavia con la
disciplina completa de PR22.

Genera entidades TEXT visibles en AutoCAD/Civil 3D para atributos
con valor cartografico directo -- confirmado el caso principal:
la elevacion de una curva de nivel. Existe por separado de
``entities.py``/``xdata.py`` a proposito:

- ``entities.py`` escribe la GEOMETRIA real de un Feature.
- ``xdata.py`` adjunta TODOS los atributos como datos extendidos,
  invisibles en el dibujo pero recuperables programaticamente.
- Este modulo (``annotation.py``) agrega una representacion VISIBLE
  de un atributo especifico -- un TEXT que cualquiera que abra el
  archivo en AutoCAD puede leer directamente, sin comandos ni scripts.

Ninguna de las 3 responsabilidades reemplaza a las otras -- una
curva de nivel, con las 3 activas, tiene su geometria (LWPOLYLINE),
su XDATA (elevation=205.0, recuperable programaticamente), y su
etiqueta visible ("205.00", legible a simple vista) al mismo tiempo.

Deliberadamente NO todas las features llevan etiqueta por defecto --
confirmado, un dataset real y grande con una etiqueta en cada
elemento se vuelve ilegible. Solo las curvas de nivel llevan
etiqueta por defecto (``contour_labels=True``); las features
puntuales (arbol, poste, punto de control) son opt-in
(``point_labels=False`` por defecto).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Any

from topocore.features.models import Feature, FeatureType


def add_contour_label(msp: Any, feature: Feature, layer: str, text_height: float) -> Any | None:
    """
    Agrega un TEXT con la elevacion de una curva de nivel, en el
    vertice medio de su polilinea (un punto representativo a lo
    largo de la curva, no necesariamente su primer o ultimo punto).

    Devuelve la entidad TEXT creada, o ``None`` si el feature no
    tiene un atributo ``elevation`` valido para etiquetar (no es un
    error -- una curva sin ese atributo simplemente no lleva etiqueta).
    """
    elevation = feature.attributes.get("elevation")
    if elevation is None or isinstance(elevation, bool) or not isinstance(elevation, (int, float)):
        return None

    vertices = feature.geometry.vertices
    if len(vertices) == 0:
        return None

    midpoint = vertices[len(vertices) // 2]
    text = msp.add_text(
        f"{float(elevation):.2f}",
        dxfattribs={"layer": layer, "height": text_height},
    )
    text.set_placement((float(midpoint[0]), float(midpoint[1])))
    return text


def add_point_label(msp: Any, feature: Feature, layer: str, text_height: float) -> Any | None:
    """
    Agrega un TEXT junto a una feature puntual, con su identificador
    de campo real si esta disponible, o el propio ``feature_id`` como
    respaldo.

    El identificador de campo real, cuando la feature viene de
    ``FeatureBuilder`` (un survey real), esta en
    ``attributes["survey_point_ids"]`` -- una TUPLA de ids (incluso
    para una feature puntual, un tuple de 1 elemento, para mantener
    la misma forma que las features de linea, que traen varios).
    Confirmado con ejecucion real contra FeatureBuilder -- NO existe
    ninguna clave ``survey_id`` de por si; asumirla sin verificar
    habria dejado el respaldo (feature_id) como el unico resultado
    real para cualquier pipeline basado en FeatureBuilder.

    Devuelve la entidad TEXT creada, o ``None`` si la geometria no
    tiene ningun vertice (no deberia ocurrir para una feature puntual
    valida, pero se evita el error en vez de asumirlo).
    """
    vertices = feature.geometry.vertices
    if len(vertices) == 0:
        return None

    survey_point_ids = feature.attributes.get("survey_point_ids")
    if survey_point_ids:
        label_text = str(survey_point_ids[0])
    else:
        label_text = str(feature.feature_id)

    position = vertices[0]
    text = msp.add_text(
        label_text,
        dxfattribs={"layer": layer, "height": text_height},
    )
    text.set_placement((float(position[0]), float(position[1])))
    return text


def should_label(feature: Feature, *, contour_labels: bool, point_labels: bool, is_point: bool) -> bool:
    """Decide si ``feature`` recibe una etiqueta, segun su tipo y las opciones dadas."""
    if feature.feature_type == FeatureType.CONTOUR:
        return contour_labels
    if is_point:
        return point_labels
    return False


__all__ = ["add_contour_label", "add_point_label", "should_label"]
