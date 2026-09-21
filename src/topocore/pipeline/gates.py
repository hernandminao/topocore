"""
topocore.pipeline.gates -- PROPUESTA, no auditada todavia con la
disciplina completa de PR22.

Puertas de validacion CONCRETAS para un limite cerrado
(survey_type=PROPERTY) -- cada una es un algoritmo real, no una
heuristica aspiracional. Deliberadamente NO incluye "coherencia de
perimetro" ni "evidencia suficiente" como puertas propias: ninguna
de las 2 tiene, hoy, una definicion algoritmica concreta -- se
omiten en vez de implementar algo que solo aparenta validar.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.pipeline.boundary import has_self_intersections, shoelace_area
from topocore.pipeline.exceptions import PropertyBoundaryError

_MIN_BOUNDARY_POINTS = 3


def validate_property_boundary(vertices: list[tuple[float, float]]) -> float:
    """
    Aplica las puertas de validacion concretas a un limite cerrado, en
    orden, y devuelve el area (Shoelace) si todas pasan.

    Puertas, en orden:
    1. Puntos suficientes (>= 3 -- un poligono real minimo).
    2. Sin autointersecciones (algoritmo real de interseccion de
       segmentos -- no una heuristica de longitud de lado).
    3. Area > 0 (un poligono degenerado, ej. todos los puntos
       colineales, da area 0).

    Raises
    ------
    PropertyBoundaryError
        Si alguna puerta falla. El pipeline se detiene aqui --
        nunca genera salidas sobre un limite invalido.
    """
    if len(vertices) < _MIN_BOUNDARY_POINTS:
        raise PropertyBoundaryError(
            f"Property boundary has only {len(vertices)} point(s); "
            f"at least {_MIN_BOUNDARY_POINTS} are needed to form a closed polygon."
        )

    if has_self_intersections(vertices):
        raise PropertyBoundaryError(
            "Property boundary self-intersects -- this is not a valid simple polygon. "
            "Pipeline stopped to prevent generating outputs from an invalid boundary."
        )

    area = shoelace_area(vertices)
    if area <= 0:
        raise PropertyBoundaryError(
            f"Property boundary has zero or negative area ({area}) -- "
            "the points are likely collinear or otherwise degenerate."
        )

    return area


__all__ = ["validate_property_boundary"]
