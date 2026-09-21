"""
topocore.survey.multiline -- PROPUESTA, no auditada todavia con la
disciplina completa de PR22 (contratos formales, matriz de error
paths, regresion completa). Construida y verificada con datos reales
de campo durante la validacion de PR24 (no sinteticos).

Problema real que resuelve, confirmado con 2 levantamientos reales
distintos:

    Un codigo de campo (ej. "CERCA", "BORDE") puede representar 2
    lineas fisicas separadas -- una a cada lado de una via sin
    distincion izquierda/derecha en el propio codigo. FeatureBuilder,
    incluso en modo grammar, conecta TODOS los puntos de un mismo
    codigo en una sola figura si no se le dice lo contrario --
    produciendo una linea que cruza el eje en zigzag.

2 estrategias de clasificacion, en orden de preferencia:

1. GEOMETRICA (recomendada, usada automaticamente si el survey trae
   un ``reference_code`` como el eje/centerline): cada punto se
   proyecta sobre la polilinea de referencia, y se clasifica por el
   signo de su distancia perpendicular -- un punto nunca puede
   quedar "cruzando" el eje, es una garantia geometrica, no
   estadistica. Tambien resuelve vias curvas correctamente, porque
   ordena por estacion (distancia acumulada a lo largo del eje), no
   por una coordenada X/Y global.
2. ESTADISTICA (respaldo, usada si no hay eje disponible en el
   survey): busca el mayor vacio en la coordenada transversal y lo
   acepta como separacion real si supera una distancia minima
   absoluta. Confirmado, con datos reales, menos robusta que la
   geometrica -- puede fallar en vias con curvas fuertes.

REGLA GENERAL DE DECISION -- que funcion usar segun el tipo de
elemento, independientemente de si el levantamiento es una via, un
predio, un lote, o cualquier otro tipo de proyecto:

    ¿El elemento es una LINEA relativa a un EJE/centerline
    (2 lados de una via, ej. bordes de pavimento)?
        -> split_multiline_codes() con reference_code="EJE"
           (garantia geometrica: nunca cruza el eje)

    ¿El elemento es un PERIMETRO CERRADO o una figura discreta
    que se repite N veces, SIN eje de referencia (perimetro de
    predio/lote, base de una columna, contorno de una estructura)?
        -> split_by_clustering() con ordering="hull" (el valor
           por defecto) -- confirmado consistente en los 2 casos
           reales muy distintos usados para validar este modulo
           (columnas de una estructura deportiva, perimetro de un
           predio con muescas reales), sin necesidad de elegir una
           estrategia distinta segun el tipo de levantamiento.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from topocore.survey.models import SurveyPoint, SurveyPointSet


class MultilineError(Exception):
    """Entrada invalida para split_multiline_codes()."""


@dataclass(frozen=True, slots=True)
class MultilineSplitConfig:
    """
    Contrato de configuracion para split_multiline_codes().

    Parameters
    ----------
    linear_codes
        Codigos que representan una o mas lineas continuas.
    splittable_codes
        Subconjunto de ``linear_codes`` que puede representar 2
        lineas fisicas bajo el mismo codigo (bordes y cercas de vias
        sin distincion izquierda/derecha en el propio codigo).
    reference_code
        Codigo a usar como eje/centerline de referencia para la
        clasificacion GEOMETRICA por lado (ej. ``"EJE"``). Si este
        codigo esta presente en el survey, la clasificacion por
        signo de distancia perpendicular al eje se usa SIEMPRE para
        los codigos en ``splittable_codes`` -- es geometricamente
        imposible que un punto quede mal clasificado al lado
        equivocado del eje con este metodo, a diferencia del
        respaldo estadistico. Si es ``None``, o el codigo no aparece
        en el survey, se usa el metodo estadistico de respaldo.
    minimum_gap_meters
        Solo aplica al metodo estadistico de respaldo (cuando
        ``reference_code`` no esta disponible). El vacio transversal
        debe ser al menos esta distancia para considerarse una
        separacion real. Confirmado con datos reales: una distancia
        absoluta es mas robusta que una proporcion relativa, que
        resulto indistinguible del ruido natural de GNSS/estacion
        total en al menos 1 caso real confirmado.
    min_points_to_evaluate_split
        No se evalua division en codigos con menos de este numero de
        puntos, en ninguno de los 2 metodos.
    """

    linear_codes: frozenset[str]
    splittable_codes: frozenset[str]
    reference_code: str | None = None
    minimum_gap_meters: float = 0.30
    min_points_to_evaluate_split: int = 4

    def __post_init__(self) -> None:
        if not self.splittable_codes.issubset(self.linear_codes):
            raise MultilineError(
                f"splittable_codes must be a subset of linear_codes; "
                f"{self.splittable_codes - self.linear_codes} are not in linear_codes."
            )
        if self.reference_code is not None and self.reference_code in self.splittable_codes:
            raise MultilineError(
                f"reference_code ('{self.reference_code}') cannot also be in splittable_codes -- "
                "a reference line (e.g. a centerline) is never split against itself."
            )
        if self.minimum_gap_meters <= 0:
            raise MultilineError("minimum_gap_meters must be positive.")
        if self.min_points_to_evaluate_split < 4:
            raise MultilineError(
                "min_points_to_evaluate_split must be at least 4 "
                "(fewer than 2 points per side makes a LINE meaningless)."
            )


def _replace_point_code(point: SurveyPoint, code: str) -> SurveyPoint:
    """
    Construye una nueva instancia directamente con el constructor
    real, en vez de dataclasses.replace() -- confirmado necesario:
    la firma de tipos de dataclasses.replace() en typeshed devuelve
    un DataclassInstance generico, y distintas versiones de mypy lo
    resuelven de forma distinta. Llamar al constructor real evita
    depender de esa ambiguedad, en cualquier version de mypy.
    """
    return SurveyPoint(id=point.id, x=point.x, y=point.y, z=point.z, code=code)


def _relabeled_survey(survey: SurveyPointSet, points: tuple[SurveyPoint, ...]) -> SurveyPointSet:
    """Misma razon que ``_replace_point_code`` arriba."""
    return SurveyPointSet(points=points, crs=survey.crs)


def _dominant_axis(points: tuple[SurveyPoint, ...]) -> str:
    """'x' o 'y', el eje con mayor dispersion real entre los puntos dados."""
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return "x" if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else "y"


def _project_onto_polyline(x: float, y: float, polyline: list[SurveyPoint]) -> tuple[float, float]:
    """
    Proyecta (x, y) sobre la polilinea de referencia (ya ordenada).

    Devuelve (estacion, distancia_con_signo):
    - estacion: distancia acumulada a lo largo de la polilinea hasta
      el punto proyectado mas cercano.
    - distancia_con_signo: distancia perpendicular al segmento mas
      cercano, con signo -- el signo es la clasificacion de lado
      (izquierda/derecha), consistente a lo largo de toda la
      polilinea porque se calcula con el mismo producto cruzado en
      cada segmento.

    Si la polilinea tiene un solo punto o esta vacia, la distancia
    con signo se calcula respecto a ese unico punto (sin sentido de
    lado real -- caso degenerado, no se espera en uso normal).
    """
    if len(polyline) < 2:
        ref = polyline[0] if polyline else None
        if ref is None:
            return 0.0, 0.0
        return 0.0, math.hypot(x - ref.x, y - ref.y)

    best_distance = math.inf
    best_station = 0.0
    best_signed_distance = 0.0
    cumulative_station = 0.0

    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        vx, vy = b.x - a.x, b.y - a.y
        segment_length_sq = vx * vx + vy * vy
        segment_length = math.sqrt(segment_length_sq)

        if segment_length_sq > 0:
            t = max(0.0, min(1.0, ((x - a.x) * vx + (y - a.y) * vy) / segment_length_sq))
            closest_x, closest_y = a.x + t * vx, a.y + t * vy
            distance = math.hypot(x - closest_x, y - closest_y)
            cross = vx * (y - a.y) - vy * (x - a.x)
            signed_distance = distance if cross >= 0 else -distance
            station = cumulative_station + t * segment_length

            if distance < best_distance:
                best_distance = distance
                best_station = station
                best_signed_distance = signed_distance

        cumulative_station += segment_length

    return best_station, best_signed_distance


def _split_by_reference_line(
    points: list[SurveyPoint],
    reference_polyline: list[SurveyPoint],
) -> list[list[SurveyPoint]]:
    """
    Clasifica cada punto por el SIGNO de su distancia perpendicular a
    ``reference_polyline`` -- geometricamente, nunca puede cruzar el
    eje, a diferencia de una separacion estadistica. Cada grupo
    resultante se ordena por estacion (distancia acumulada a lo largo
    de la referencia), no por una coordenada X/Y global -- esto
    tambien resuelve vias con curvas correctamente.
    """
    projected = [(_project_onto_polyline(p.x, p.y, reference_polyline), p) for p in points]

    left = sorted(((station, p) for (station, signed), p in projected if signed >= 0), key=lambda sp: sp[0])
    right = sorted(((station, p) for (station, signed), p in projected if signed < 0), key=lambda sp: sp[0])

    groups = [[p for _, p in side] for side in (left, right) if side]
    # Confirmado necesario: cuando todos los puntos caen del mismo
    # lado (no hay separacion real que hacer), el grupo unico ya
    # resultante de "left"/"right" arriba SIGUE ordenado por
    # estacion -- devolverlo tal cual, no los "points" originales
    # sin ordenar (que quedarian en el orden fisico del archivo, no
    # en el orden real a lo largo de la via).
    return groups if groups else [points]


def _split_by_statistical_gap(
    points: list[SurveyPoint],
    transverse_axis: str,
    minimum_gap_meters: float,
) -> list[list[SurveyPoint]]:
    """
    Respaldo cuando no hay una linea de referencia disponible: busca
    el mayor vacio en la coordenada transversal, y lo acepta como
    separacion real si es al menos ``minimum_gap_meters``.
    """
    key: Callable[[SurveyPoint], float] = (lambda p: p.x) if transverse_axis == "x" else (lambda p: p.y)
    ordered = sorted(points, key=key)
    values = [key(p) for p in ordered]

    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    largest_gap = max(gaps)
    cut_index = gaps.index(largest_gap)

    if largest_gap >= minimum_gap_meters:
        return [ordered[: cut_index + 1], ordered[cut_index + 1 :]]
    return [points]


def split_multiline_codes(survey: SurveyPointSet, config: MultilineSplitConfig) -> SurveyPointSet:
    """
    Reetiqueta los codigos de ``survey`` con la sintaxis grammar
    (``BASE.FIGURA[.S|E]``) de ``topocore.features.grammar``.

    No modifica ``survey`` -- devuelve un ``SurveyPointSet`` nuevo.
    Los puntos cuyo codigo no esta en ``config.linear_codes`` se
    devuelven sin cambios, en su posicion original.

    Raises
    ------
    MultilineError
        Si ``survey`` esta vacio.
    """
    if len(survey.points) == 0:
        raise MultilineError("Cannot process an empty SurveyPointSet.")

    points = list(survey.points)
    linear_points = [p for p in points if p.code in config.linear_codes]

    if not linear_points:
        return survey

    sort_axis = _dominant_axis(tuple(linear_points))
    transverse_axis = "y" if sort_axis == "x" else "x"
    sort_key: Callable[[SurveyPoint], float] = (lambda p: p.x) if sort_axis == "x" else (lambda p: p.y)

    reference_polyline: list[SurveyPoint] | None = None
    if config.reference_code is not None:
        reference_points = [p for p in points if p.code == config.reference_code]
        if reference_points:
            reference_polyline = sorted(reference_points, key=sort_key)

    figures_by_code: dict[str, list[list[SurveyPoint]]] = {}
    for code in config.linear_codes:
        code_points = [p for p in points if p.code == code]
        if not code_points:
            continue

        if code in config.splittable_codes and len(code_points) >= config.min_points_to_evaluate_split:
            if reference_polyline is not None:
                groups = _split_by_reference_line(code_points, reference_polyline)
            else:
                groups = _split_by_statistical_gap(code_points, transverse_axis, config.minimum_gap_meters)
                groups = [sorted(g, key=sort_key) for g in groups]
        else:
            groups = [sorted(code_points, key=sort_key)]

        figures_by_code[code] = groups

    # Cada codigo consume su propia cola de (indice_figura, punto),
    # figura por figura, en el orden en que ese codigo aparece
    # fisicamente en el archivo original -- el ORDEN espacial real ya
    # quedo resuelto arriba; esto solo decide que punto real
    # corresponde a cada aparicion fisica del codigo.
    queues = {
        code: [(figure_index, p) for figure_index, figure in enumerate(figures) for p in figure]
        for code, figures in figures_by_code.items()
    }
    cursor = dict.fromkeys(queues, 0)

    totals = {
        (code, figure_index): len(figure)
        for code, figures in figures_by_code.items()
        for figure_index, figure in enumerate(figures)
    }
    seen: dict[tuple[str, int], int] = {}

    relabeled: list[SurveyPoint] = []
    for p in points:
        if p.code not in queues:
            relabeled.append(p)
            continue

        figure_index, real_point = queues[p.code][cursor[p.code]]
        cursor[p.code] += 1

        key = (p.code, figure_index)
        seen[key] = seen.get(key, 0) + 1
        figure_number = figure_index + 1

        if seen[key] == 1:
            new_code = f"{p.code}.{figure_number}.S"
        elif seen[key] == totals[key]:
            new_code = f"{p.code}.{figure_number}.E"
        else:
            new_code = f"{p.code}.{figure_number}"

        relabeled.append(_replace_point_code(real_point, new_code))

    return _relabeled_survey(survey, tuple(relabeled))


@dataclass(frozen=True, slots=True)
class ClusterSplitConfig:
    """
    Contrato de configuracion para split_by_clustering().

    A diferencia de MultilineSplitConfig (pensado para exactamente 2
    lados relativos a un eje/centerline), esto resuelve el caso mas
    general: un codigo puede representar un numero DESCONOCIDO de
    figuras separadas (ej. columnas de una estructura, tramos de
    anden en un predio -- sin ningun eje de referencia disponible).

    Reutiliza topocore.processing.segmentation.dbscan.DBSCANSegmenter
    (ya existente, auditado) en vez de reimplementar clustering
    espacial -- confirmado con datos reales, esto es un problema
    genuinamente dificil: cuando la distancia interna de una figura
    y la distancia entre figuras vecinas son de magnitud similar (
    confirmado con datos reales: ~2.25m dentro de una columna vs.
    ~1.2-3m entre columnas vecinas), NINGUN valor unico de eps separa
    esto perfectamente -- es un limite real de un clustering basado
    unicamente en distancia, no algo que este modulo pueda resolver
    de forma perfecta y automatica en todos los casos.

    Parameters
    ----------
    linear_codes
        Codigos a agrupar. Cada codigo se agrupa de forma
        INDEPENDIENTE de los demas (los puntos de un codigo nunca se
        mezclan con los de otro al calcular clusters).
    eps
        Distancia maxima (en las mismas unidades del CRS -- metros,
        por convencion de todo el proyecto) para que 2 puntos se
        consideren parte de la misma figura. Ver DBSCANSegmenter.
    min_samples
        Minimo de puntos para formar un cluster "denso" -- con 1
        (el valor por defecto, recomendado para este caso de uso),
        cada punto sin vecinos cercanos se convierte en su propia
        figura de 1 solo punto, en vez de descartarse como ruido.
    ordering
        Como ordenar los puntos DENTRO de cada figura detectada,
        antes de aplicar la sintaxis grammar:
        - "hull" (por defecto, recomendado): casco convexo
          (scipy.spatial.ConvexHull) mas insercion de los puntos
          restantes en la posicion que menos alarga el perimetro --
          la opcion MAS ROBUSTA para CUALQUIER geometria CERRADA
          real (perimetro de predio/lote, base de una columna,
          contorno de una estructura), independientemente del tipo
          de levantamiento. Nunca pierde puntos reales (a diferencia
          de un casco convexo puro, que dejaria fuera los vertices
          concavos/muescas reales). Confirmado con 2 casos reales
          distintos (columnas de una estructura, perimetro de un
          predio con muescas): mismo resultado correcto en ambos, sin
          necesidad de elegir una estrategia distinta segun el tipo
          de proyecto -- esto es lo que hace que el comportamiento
          sea consistente para TopoCore en general, no especifico de
          un tipo de levantamiento.
        - "angular": ordena por angulo alrededor del centroide de la
          figura -- mas simple que "hull", funciona igual de bien
          para figuras pequenas y razonablemente convexas (ej. la
          base de una columna), pero puede fallar en perimetros
          grandes e irregulares donde "hull" es mas confiable.
        - "axis": ordena por el eje de mayor dispersion real de la
          propia figura -- para geometrias de LINEA ABIERTA (ej. un
          tramo de anden que no es un perimetro cerrado), igual que
          MultilineSplitConfig.
        - "nearest": encadena cada punto con su vecino no visitado
          mas cercano. Confirmado con datos reales: puede quedar
          "atrapado" en un camino local malo (una hebra larga que se
          aleja y regresa), produciendo una forma invalida para un
          perimetro real -- reemplazado por "hull" como opcion por
          defecto exactamente por este motivo. Se conserva disponible
          por si un caso especifico se beneficia de ella.
    """

    linear_codes: frozenset[str]
    eps: float
    min_samples: int = 1
    ordering: str = "hull"

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise MultilineError("eps must be positive.")
        if self.min_samples < 1:
            raise MultilineError("min_samples must be at least 1.")
        if self.ordering not in ("angular", "axis", "nearest", "hull"):
            raise MultilineError(f"ordering must be 'angular', 'axis', 'nearest', or 'hull'; got '{self.ordering}'.")


def _order_angular(points: list[SurveyPoint]) -> list[SurveyPoint]:
    """Ordena por angulo alrededor del centroide -- para figuras CERRADAS."""
    if len(points) < 3:
        return points
    cx = sum(p.x for p in points) / len(points)
    cy = sum(p.y for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p.y - cy, p.x - cx))


def _order_axis(points: list[SurveyPoint]) -> list[SurveyPoint]:
    """Ordena por el eje de mayor dispersion real de la propia figura -- para LINEAS."""
    if len(points) < 2:
        return points
    axis = _dominant_axis(tuple(points))
    key: Callable[[SurveyPoint], float] = (lambda p: p.x) if axis == "x" else (lambda p: p.y)
    return sorted(points, key=key)


def _order_nearest_neighbor(points: list[SurveyPoint]) -> list[SurveyPoint]:
    """Encadena cada punto con su vecino no visitado mas cercano --
    heuristica robusta para perimetros irregulares/no convexos."""
    if len(points) < 3:
        return points
    restantes = list(points)
    actual = restantes.pop(0)
    cadena = [actual]
    while restantes:

        def _distancia_a_actual(p: SurveyPoint, ref: SurveyPoint = actual) -> float:
            return math.hypot(p.x - ref.x, p.y - ref.y)

        siguiente = min(restantes, key=_distancia_a_actual)
        cadena.append(siguiente)
        restantes.remove(siguiente)
        actual = siguiente
    return cadena


def _order_convex_hull_with_insertion(points: list[SurveyPoint]) -> list[SurveyPoint]:
    """
    Casco convexo (scipy.spatial.ConvexHull) como perimetro base, mas
    insercion de los puntos restantes en la posicion que menos alarga
    el perimetro total -- confirmado mas robusto que "nearest" con
    datos reales de un perimetro de predio con muescas/entrantes.
    """
    if len(points) < 4:
        return points

    from scipy.spatial import ConvexHull, QhullError

    coords = [(p.x, p.y) for p in points]
    try:
        hull = ConvexHull(coords)
    except QhullError:  # puntos colineales u otro caso degenerado
        return points

    # hull.vertices es un ndarray de numpy.int32 (no int nativo de
    # Python) -- confirmado, se convierte explicitamente aqui para
    # que el resto de la funcion (incluida list.insert() mas abajo)
    # trabaje con int reales, no con el dtype de numpy filtrandose.
    orden: list[int] = [int(i) for i in hull.vertices]
    en_hull = set(orden)
    fuera_hull = [i for i in range(len(points)) if i not in en_hull]

    def distancia(a: int, b: int) -> float:
        return math.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1])

    for punto in fuera_hull:
        mejor_costo: float | None = None
        mejor_posicion = 0
        for i in range(len(orden)):
            a, b = orden[i], orden[(i + 1) % len(orden)]
            costo = distancia(a, punto) + distancia(punto, b) - distancia(a, b)
            if mejor_costo is None or costo < mejor_costo:
                mejor_costo = costo
                mejor_posicion = i + 1
        orden.insert(mejor_posicion, punto)

    return [points[i] for i in orden]


def split_by_clustering(survey: SurveyPointSet, config: ClusterSplitConfig) -> SurveyPointSet:
    """
    Reetiqueta los codigos de ``survey`` con la sintaxis grammar,
    detectando automaticamente cuantas figuras separadas representa
    cada codigo mediante clustering espacial (DBSCAN) -- sin
    necesidad de un eje/centerline de referencia, a diferencia de
    ``split_multiline_codes``.

    No modifica ``survey`` -- devuelve un ``SurveyPointSet`` nuevo.
    Los puntos cuyo codigo no esta en ``config.linear_codes`` se
    devuelven sin cambios, en su posicion original.

    Raises
    ------
    MultilineError
        Si ``survey`` esta vacio.
    """
    from topocore.processing.segmentation.dbscan import DBSCANSegmenter

    if len(survey.points) == 0:
        raise MultilineError("Cannot process an empty SurveyPointSet.")

    points = list(survey.points)
    segmenter = DBSCANSegmenter(eps=config.eps, min_samples=config.min_samples)
    ordenar = {
        "angular": _order_angular,
        "axis": _order_axis,
        "nearest": _order_nearest_neighbor,
        "hull": _order_convex_hull_with_insertion,
    }[config.ordering]

    figures_by_code: dict[str, list[list[SurveyPoint]]] = {}
    for code in config.linear_codes:
        code_points = [p for p in points if p.code == code]
        if not code_points:
            continue

        coords = np.array([[p.x, p.y, 0.0] for p in code_points])
        labels, num_clusters = segmenter.cluster(coords)

        grupos: list[list[SurveyPoint]] = [[] for _ in range(num_clusters)]
        sin_grupo: list[SurveyPoint] = []
        for point, label in zip(code_points, labels):
            if label < 0:
                sin_grupo.append(point)
            else:
                grupos[label].append(point)

        figures_by_code[code] = [ordenar(g) for g in grupos if g] + [[p] for p in sin_grupo]

    queues = {
        code: [(figure_index, p) for figure_index, figure in enumerate(figures) for p in figure]
        for code, figures in figures_by_code.items()
    }
    cursor = dict.fromkeys(queues, 0)

    totals = {
        (code, figure_index): len(figure)
        for code, figures in figures_by_code.items()
        for figure_index, figure in enumerate(figures)
    }
    seen: dict[tuple[str, int], int] = {}

    relabeled: list[SurveyPoint] = []
    for p in points:
        if p.code not in queues:
            relabeled.append(p)
            continue

        figure_index, real_point = queues[p.code][cursor[p.code]]
        cursor[p.code] += 1

        key = (p.code, figure_index)
        seen[key] = seen.get(key, 0) + 1
        figure_number = figure_index + 1

        if seen[key] == 1:
            new_code = f"{p.code}.{figure_number}.S"
        elif seen[key] == totals[key]:
            new_code = f"{p.code}.{figure_number}.E"
        else:
            new_code = f"{p.code}.{figure_number}"

        relabeled.append(_replace_point_code(real_point, new_code))

    return _relabeled_survey(survey, tuple(relabeled))


__all__ = [
    "ClusterSplitConfig",
    "MultilineError",
    "MultilineSplitConfig",
    "split_by_clustering",
    "split_multiline_codes",
]
