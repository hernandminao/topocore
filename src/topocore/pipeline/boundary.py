"""
topocore.pipeline.boundary -- PROPUESTA, no auditada todavia con la
disciplina completa de PR22.

Valida un limite cerrado (perimetro de predio, contorno de una
estructura) antes de aceptarlo como entrada valida para el resto del
pipeline -- confirmado necesario: ninguna heuristica de "longitud de
segmento" demuestra que un poligono sea invalido (un lote real puede
tener legitimamente un lado mucho mas largo que los demas); la
prueba correcta es geometrica: ¿el poligono se autointerseca?

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations


def shoelace_area(vertices: list[tuple[float, float]]) -> float:
    """
    Area encerrada por un poligono cerrado simple (formula de
    Shoelace / Gauss). ``vertices`` no necesita repetir el primer
    punto al final -- el cierre es implicito.

    Devuelve siempre un valor >= 0 (area absoluta, sin importar el
    sentido de recorrido del poligono).
    """
    n = len(vertices)
    if n < 3:
        return 0.0

    total = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        total += x1 * y2 - x2 * y1

    return abs(total) / 2.0


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """
    True si el segmento (p1,p2) cruza realmente el segmento (p3,p4)
    -- test de orientacion estandar (producto cruzado), no una
    aproximacion por distancia. Segmentos que solo se TOCAN en un
    extremo compartido (el caso normal entre 2 lados consecutivos de
    un poligono) NO cuentan como cruce -- ver has_self_intersections()
    para como se excluyen esos pares antes de llamar aqui.
    """

    def orientacion(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = orientacion(p3, p4, p1)
    d2 = orientacion(p3, p4, p2)
    d3 = orientacion(p1, p2, p3)
    d4 = orientacion(p1, p2, p4)

    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def has_self_intersections(vertices: list[tuple[float, float]]) -> bool:
    """
    True si el poligono cerrado formado por ``vertices`` (en orden,
    cierre implicito del ultimo al primero) tiene algun cruce real
    entre 2 de sus lados -- la prueba correcta de que un perimetro es
    geometricamente invalido, en vez de una heuristica de longitud de
    segmento (un lado real puede ser legitimamente mucho mas largo
    que los demas sin que el poligono sea invalido).

    Lados ADYACENTES (que comparten un vertice) nunca se evaluan
    entre si -- comparten un punto por construccion, eso no es una
    autointerseccion real.
    """
    n = len(vertices)
    if n < 4:
        return False

    lados = [(vertices[i], vertices[(i + 1) % n]) for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # lados adyacentes -- comparten un vertice, no es un cruce real
            a1, a2 = lados[i]
            b1, b2 = lados[j]
            if _segments_intersect(a1, a2, b1, b2):
                return True

    return False


__all__ = ["has_self_intersections", "shoelace_area"]
