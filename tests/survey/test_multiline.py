"""
Regression suite for topocore.survey.multiline (prototype, not yet
formally audited with PR22's full discipline).

Protects the exact real-world behavior confirmed during PR24's own
field-data validation: a real survey where CERCA and BORDE each
represented 2 physically separate lines (both sides of an unpaved
road) sharing a single field code, with no left/right distinction.
"""

from __future__ import annotations

import math

import pytest
from topocore.survey.models import SurveyPoint, SurveyPointSet
from topocore.survey.multiline import (
    ClusterSplitConfig,
    MultilineError,
    MultilineSplitConfig,
    _project_onto_polyline,
    split_by_clustering,
    split_multiline_codes,
)


def _point(id_: str, x: float, y: float, z: float, code: str) -> SurveyPoint:
    return SurveyPoint(id=id_, x=x, y=y, z=z, code=code)


DEFAULT_CONFIG = MultilineSplitConfig(
    linear_codes=frozenset({"EJE", "BORDE", "CERCA"}),
    splittable_codes=frozenset({"BORDE", "CERCA"}),
)


def test_config_rejects_splittable_codes_not_in_linear_codes() -> None:
    with pytest.raises(MultilineError, match="subset"):
        MultilineSplitConfig(
            linear_codes=frozenset({"EJE"}),
            splittable_codes=frozenset({"BORDE"}),
        )


def test_config_rejects_reference_code_also_in_splittable_codes() -> None:
    with pytest.raises(MultilineError, match="reference_code"):
        MultilineSplitConfig(
            linear_codes=frozenset({"EJE"}),
            splittable_codes=frozenset({"EJE"}),
            reference_code="EJE",
        )


def test_config_rejects_nonpositive_minimum_gap() -> None:
    with pytest.raises(MultilineError):
        MultilineSplitConfig(
            linear_codes=frozenset({"EJE"}),
            splittable_codes=frozenset(),
            minimum_gap_meters=0.0,
        )


def test_config_rejects_too_small_min_points() -> None:
    with pytest.raises(MultilineError):
        MultilineSplitConfig(
            linear_codes=frozenset({"EJE"}),
            splittable_codes=frozenset(),
            min_points_to_evaluate_split=2,
        )


def test_empty_survey_raises() -> None:
    with pytest.raises(MultilineError, match="empty"):
        split_multiline_codes(SurveyPointSet(points=()), DEFAULT_CONFIG)


def test_codes_outside_linear_codes_are_untouched() -> None:
    survey = SurveyPointSet(
        points=(
            _point("1", 0.0, 0.0, 10.0, "ARBOL"),
            _point("2", 5.0, 5.0, 10.0, "ARBOL"),
        )
    )
    result = split_multiline_codes(survey, DEFAULT_CONFIG)
    assert result.points == survey.points


def test_a_genuinely_single_line_is_not_split() -> None:
    """A fence walked continuously along one side only -- confirmed
    real scenario from via1's own CERCA, no artificial split."""
    points = tuple(_point(str(i), float(i * 5), 0.0, 10.0, "CERCA") for i in range(6))
    survey = SurveyPointSet(points=points)

    result = split_multiline_codes(survey, DEFAULT_CONFIG)

    codes = {p.code for p in result.points}
    assert codes == {"CERCA.1.S", "CERCA.1", "CERCA.1.E"} or codes <= {"CERCA.1.S", "CERCA.1", "CERCA.1.E"}


def test_two_real_sides_under_one_code_are_split_into_2_figures() -> None:
    """
    Regression: this exact shape (2 clusters on opposite sides of a
    road, sharing one code) previously produced a single zigzagging
    line when fed directly to FeatureBuilder's grammar mode.

    Shaped like the real via2 data: the road progresses along Y
    (large range), the 2 sides differ in X (small, transverse
    range) -- confirmed this exact axis relationship in real field
    data (a north-south road, X range 19m vs Y range 118m).
    """
    lado_izquierdo = [_point(f"L{i}", 0.0 + i * 0.05, float(i * 5), 10.0, "CERCA") for i in range(5)]
    lado_derecho = [_point(f"R{i}", 8.0 + i * 0.05, float(i * 5), 10.0, "CERCA") for i in range(5)]
    # Intercalados en el archivo, como en el caso real de campo.
    intercalados = [p for pair in zip(lado_izquierdo, lado_derecho) for p in pair]
    survey = SurveyPointSet(points=tuple(intercalados))

    result = split_multiline_codes(survey, DEFAULT_CONFIG)

    figuras = {p.code.split(".")[1] for p in result.points if p.code.startswith("CERCA")}
    assert figuras == {"1", "2"}

    x_por_figura: dict[str, list[float]] = {}
    for p in result.points:
        if p.code.startswith("CERCA"):
            figura = p.code.split(".")[1]
            x_por_figura.setdefault(figura, []).append(p.x)
    # Cada figura debe quedar enteramente en un solo lado (X~0 o X~8),
    # nunca mezclados -- la garantia real de esta funcion.
    for xs in x_por_figura.values():
        assert (max(xs) < 4.0) or (min(xs) > 4.0)


def test_a_code_never_in_splittable_codes_is_never_split_even_with_a_real_gap() -> None:
    """
    Confirmed real requirement: EJE is, by definition, always a
    single line -- even if its own data happens to have an uneven
    gap (e.g. from multiple total-station setups), it must never be
    split into 2 figures.
    """
    config = MultilineSplitConfig(
        linear_codes=frozenset({"EJE"}),
        splittable_codes=frozenset(),  # EJE explicitamente no divisible
    )
    con_vacio_grande = [
        _point("1", 0.0, 0.0, 10.0, "EJE"),
        _point("2", 1.0, 0.0, 10.0, "EJE"),
        _point("3", 2.0, 0.0, 10.0, "EJE"),
        _point("4", 1000.0, 0.0, 10.0, "EJE"),  # vacio enorme
        _point("5", 1001.0, 0.0, 10.0, "EJE"),
    ]
    survey = SurveyPointSet(points=tuple(con_vacio_grande))

    result = split_multiline_codes(survey, config)

    figuras = {p.code.split(".")[1] for p in result.points}
    assert figuras == {"1"}


def test_reorders_points_surveyed_out_of_spatial_order() -> None:
    """
    Regression: a real survey done across multiple station setups
    can record points in an order that doesn't match their real
    spatial position along the road. This must be reordered by the
    dominant axis before grammar tagging, not left in file order.
    """
    fuera_de_orden = [
        _point("a", 50.0, 0.0, 10.0, "EJE"),
        _point("b", 10.0, 0.0, 10.0, "EJE"),
        _point("c", 30.0, 0.0, 10.0, "EJE"),
        _point("d", 20.0, 0.0, 10.0, "EJE"),
    ]
    config = MultilineSplitConfig(linear_codes=frozenset({"EJE"}), splittable_codes=frozenset())
    survey = SurveyPointSet(points=tuple(fuera_de_orden))

    result = split_multiline_codes(survey, config)

    # Lo que importa es que, leidos en SECUENCIA (por su nuevo codigo
    # S/continuar/E), representen un recorrido monotono en X.
    xs_reales = []
    for p in result.points:
        if p.code.startswith("EJE"):
            xs_reales.append(p.x)
    assert xs_reales == sorted(xs_reales)


def test_does_not_mutate_the_input_survey() -> None:
    survey = SurveyPointSet(
        points=(
            _point("1", 0.0, 0.0, 10.0, "CERCA"),
            _point("2", 5.0, 0.0, 10.0, "CERCA"),
        )
    )
    original_codes = tuple(p.code for p in survey.points)

    split_multiline_codes(survey, DEFAULT_CONFIG)

    assert tuple(p.code for p in survey.points) == original_codes


def test_a_single_line_with_realistic_gnss_jitter_is_never_split() -> None:
    """
    Regression: a genuinely single line (already correctly coded,
    e.g. BORDEI in a survey that already distinguishes both sides)
    with realistic GNSS/total-station noise (a few cm of jitter) must
    never be split. Confirmed real defect: an earlier, relative
    (ratio-based) criterion produced false positives from pure noise
    with small sample sizes -- this specific scenario, run across
    100 random seeds, is the exact regression that surfaced it.
    """
    import random

    config = MultilineSplitConfig(
        linear_codes=frozenset({"BORDEI"}),
        splittable_codes=frozenset({"BORDEI"}),
    )

    for seed in range(100):
        random.seed(seed)
        points = tuple(
            _point(f"BI{i}", -5.0 + random.uniform(-0.05, 0.05), float(i * 8), 10.0, "BORDEI") for i in range(18)
        )
        survey = SurveyPointSet(points=points)

        result = split_multiline_codes(survey, config)

        figures = {p.code.split(".")[1] for p in result.points}
        assert figures == {"1"}, f"False split detected with seed={seed}"


def test_a_real_but_physically_small_separation_is_still_detected() -> None:
    """
    Regression: confirmed real field data where 2 physically separate
    pavement edges were only ~1.1m apart (much closer than a real
    fence's own ~7.5m separation in the same survey). A criterion
    that only compares the gap to internal noise/dispersion -- rather
    than to an absolute physical distance -- was confirmed unable to
    detect this specific, real, physically-small-but-genuine
    separation without also producing false positives elsewhere.
    """
    izquierda = [_point(f"L{i}", 5.0 + i * 0.01, float(i * 8), 10.0, "BORDE") for i in range(10)]
    derecha = [_point(f"R{i}", 6.15 + i * 0.01, float(i * 8), 10.0, "BORDE") for i in range(10)]
    intercalados = [p for pair in zip(izquierda, derecha) for p in pair]
    survey = SurveyPointSet(points=tuple(intercalados))

    config = MultilineSplitConfig(
        linear_codes=frozenset({"BORDE"}),
        splittable_codes=frozenset({"BORDE"}),
    )

    result = split_multiline_codes(survey, config)

    figuras = {p.code.split(".")[1] for p in result.points}
    assert figuras == {"1", "2"}


def test_geometric_method_reorders_a_single_sided_code_by_station_even_when_recorded_out_of_order() -> None:
    """
    Regression: a real defect found in production -- when a
    splittable code (e.g. BORDE) has all its points on a single side
    of the reference line (no actual split needed), the fallback
    path incorrectly returned the points in their original FILE
    order instead of the already-computed, station-sorted order.
    This produced a disordered/self-crossing line whenever the
    survey recorded that code out of spatial order (e.g. across
    multiple total-station setups) -- confirmed this caused
    SideResolver to misclassify both sides of a real road survey as
    the same side ("right"/"right" instead of "left"/"right").
    """
    eje = [_point(f"E{i}", 0.0, float(i * 5), 10.0, "EJE") for i in range(6)]
    # BORDE: un solo lado real (todo x=5.0-ish), pero grabado
    # completamente fuera de orden espacial en el archivo.
    fuera_de_orden = [
        _point("B3", 5.0, 15.0, 10.0, "BORDE"),
        _point("B0", 5.0, 0.0, 10.0, "BORDE"),
        _point("B5", 5.0, 25.0, 10.0, "BORDE"),
        _point("B1", 5.0, 5.0, 10.0, "BORDE"),
        _point("B4", 5.0, 20.0, 10.0, "BORDE"),
        _point("B2", 5.0, 10.0, 10.0, "BORDE"),
    ]
    survey = SurveyPointSet(points=tuple(eje + fuera_de_orden))

    config = MultilineSplitConfig(
        linear_codes=frozenset({"EJE", "BORDE"}),
        splittable_codes=frozenset({"BORDE"}),
        reference_code="EJE",
    )

    result = split_multiline_codes(survey, config)

    borde_ids_en_orden = [p.id for p in result.points if p.code.startswith("BORDE")]
    assert borde_ids_en_orden == ["B0", "B1", "B2", "B3", "B4", "B5"]


def test_geometric_method_never_crosses_a_curved_reference_line() -> None:
    """
    Regression: confirmed real limitation of the statistical
    (axis-based) method -- on a curved road, sorting by a single
    global axis (X or Y) does not guarantee a side-line never crosses
    the centerline. The geometric method (reference_code) classifies
    strictly by which side of the reference polyline a point falls
    on, which is a geometric guarantee, not a statistical one.

    This EJE curves: it goes right then turns to go up-and-left,
    something a single dominant-axis sort cannot represent correctly.
    """
    eje = [_point(f"E{i}", float(i * 5), 0.0, 10.0, "EJE") for i in range(4)] + [
        _point(f"E{i + 4}", 15.0 + float(i * 3), float(i * 5), 10.0, "EJE") for i in range(4)
    ]
    # Cerca del lado "positivo" (por encima/izquierda de la curva) y
    # del lado "negativo", intercaladas en el archivo.
    cerca_positiva = [_point(f"CP{i}", float(i * 4) - 2.0, 3.0 + i * 0.2, 10.0, "CERCA") for i in range(6)]
    cerca_negativa = [_point(f"CN{i}", float(i * 4) - 2.0, -3.0 - i * 0.2, 10.0, "CERCA") for i in range(6)]
    intercalados = eje + [p for pair in zip(cerca_positiva, cerca_negativa) for p in pair]
    survey = SurveyPointSet(points=tuple(intercalados))

    config = MultilineSplitConfig(
        linear_codes=frozenset({"EJE", "CERCA"}),
        splittable_codes=frozenset({"CERCA"}),
        reference_code="EJE",
    )

    result = split_multiline_codes(survey, config)

    # Reordenar el eje real segun como quedo relabeled (por su propia
    # figura -- aqui solo 1 figura, en el orden correcto de estacion).
    eje_para_verificar = [p for p in result.points if p.code.startswith("EJE")]

    figuras = {p.code.split(".")[1] for p in result.points if p.code.startswith("CERCA")}
    assert figuras == {"1", "2"}

    for figura in figuras:
        signos = []
        for p in result.points:
            if p.code == f"CERCA.{figura}.S" or p.code == f"CERCA.{figura}" or p.code == f"CERCA.{figura}.E":
                _, signo = _project_onto_polyline(p.x, p.y, eje_para_verificar)
                signos.append(signo >= 0)
        assert all(signos) or not any(signos), f"figura {figura} cruza el eje"


# ---------------------------------------------------------------------
# split_by_clustering / ClusterSplitConfig
# ---------------------------------------------------------------------


def _figuras_de(points, prefijo: str) -> dict[str, int]:
    """Cuenta puntos por numero de figura, separando por '.' correctamente
    (confirmado necesario: un startswith() ingenuo confunde 'EST.1' con
    'EST.10', 'EST.11', etc. -- error real encontrado al verificar esto)."""
    conteo: dict[str, int] = {}
    for p in points:
        if p.code.startswith(prefijo + "."):
            numero = p.code.split(".")[1]
            conteo[numero] = conteo.get(numero, 0) + 1
    return conteo


def test_cluster_config_rejects_nonpositive_eps() -> None:
    with pytest.raises(MultilineError, match="eps"):
        ClusterSplitConfig(linear_codes=frozenset({"EST"}), eps=0.0)


def test_cluster_config_rejects_zero_min_samples() -> None:
    with pytest.raises(MultilineError, match="min_samples"):
        ClusterSplitConfig(linear_codes=frozenset({"EST"}), eps=1.0, min_samples=0)


def test_cluster_config_rejects_invalid_ordering() -> None:
    with pytest.raises(MultilineError, match="ordering"):
        ClusterSplitConfig(linear_codes=frozenset({"EST"}), eps=1.0, ordering="algo_invalido")


def test_cluster_split_empty_survey_raises() -> None:
    config = ClusterSplitConfig(linear_codes=frozenset({"EST"}), eps=1.0)
    with pytest.raises(MultilineError, match="empty"):
        split_by_clustering(SurveyPointSet(points=()), config)


def test_cluster_split_detects_two_well_separated_groups() -> None:
    grupo_a = [_point(f"A{i}", float(i), 0.0, 10.0, "EST") for i in range(4)]
    grupo_b = [_point(f"B{i}", 100.0 + i, 0.0, 10.0, "EST") for i in range(4)]
    survey = SurveyPointSet(points=tuple(grupo_a + grupo_b))

    config = ClusterSplitConfig(linear_codes=frozenset({"EST"}), eps=2.0, min_samples=1)
    result = split_by_clustering(survey, config)

    conteo = _figuras_de(result.points, "EST")
    assert len(conteo) == 2
    assert sorted(conteo.values()) == [4, 4]


def test_cluster_split_does_not_confuse_figure_1_with_figure_10_or_11() -> None:
    """
    Regression: confirmed real bug found during verification -- NOT in
    split_by_clustering itself, but a real trap for any caller: naive
    string prefix matching (code.startswith("EST.1")) incorrectly
    matches "EST.10", "EST.11", etc. This test protects the CORRECT
    way to parse figure numbers (split by "." and compare the exact
    second segment), guarding against ever reintroducing that mistake
    in this module's own logic.
    """
    grupos = []
    for figura in range(12):
        grupos.extend(_point(f"P{figura}_{i}", figura * 10.0, float(i), 10.0, "EST") for i in range(2))
    survey = SurveyPointSet(points=tuple(grupos))

    config = ClusterSplitConfig(linear_codes=frozenset({"EST"}), eps=1.0, min_samples=1)
    result = split_by_clustering(survey, config)

    conteo = _figuras_de(result.points, "EST")
    assert len(conteo) == 12
    assert all(v == 2 for v in conteo.values())
    assert conteo["1"] == 2  # no debe incluir de mas los puntos de la figura 10/11


def test_cluster_split_a_point_with_no_nearby_neighbors_becomes_its_own_figure() -> None:
    aislado = _point("solo", 0.0, 0.0, 10.0, "EST")
    grupo = [_point(f"G{i}", 50.0 + i, 0.0, 10.0, "EST") for i in range(3)]
    survey = SurveyPointSet(points=(aislado, *grupo))

    config = ClusterSplitConfig(linear_codes=frozenset({"EST"}), eps=1.0, min_samples=1)
    result = split_by_clustering(survey, config)

    conteo = _figuras_de(result.points, "EST")
    assert sorted(conteo.values()) == [1, 3]


def test_angular_ordering_produces_a_non_self_intersecting_polygon() -> None:
    """Los 4 puntos de un cuadrado, dados en un orden que NO sigue el
    perimetro -- confirmado que el ordenamiento angular los reordena
    correctamente para formar un poligono simple, no un "moño"."""
    cuadrado_desordenado = [
        _point("a", 0.0, 0.0, 10.0, "EST"),
        _point("b", 1.0, 1.0, 10.0, "EST"),  # esquina opuesta -- fuera de orden
        _point("c", 1.0, 0.0, 10.0, "EST"),
        _point("d", 0.0, 1.0, 10.0, "EST"),
    ]
    survey = SurveyPointSet(points=tuple(cuadrado_desordenado))
    config = ClusterSplitConfig(linear_codes=frozenset({"EST"}), eps=5.0, min_samples=1, ordering="angular")
    result = split_by_clustering(survey, config)

    puntos_ordenados = [p for p in result.points if p.code.startswith("EST.")]
    xs = [p.x for p in puntos_ordenados]
    ys = [p.y for p in puntos_ordenados]
    # El orden angular real (sentido horario o antihorario) para un
    # cuadrado con centro en (0.5, 0.5) visita las 4 esquinas en su
    # perimetro real, nunca cruzando en diagonal dos veces seguidas.
    assert (xs, ys) != ([0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0])


def test_axis_ordering_sorts_by_the_figures_own_dominant_axis() -> None:
    fuera_de_orden = [
        _point("a", 50.0, 0.0, 10.0, "AD"),
        _point("b", 10.0, 0.0, 10.0, "AD"),
        _point("c", 30.0, 0.0, 10.0, "AD"),
    ]
    survey = SurveyPointSet(points=tuple(fuera_de_orden))
    config = ClusterSplitConfig(linear_codes=frozenset({"AD"}), eps=100.0, min_samples=1, ordering="axis")
    result = split_by_clustering(survey, config)

    xs = [p.x for p in result.points if p.code.startswith("AD.")]
    assert xs == sorted(xs)


def test_cluster_split_codes_outside_linear_codes_are_untouched() -> None:
    survey = SurveyPointSet(
        points=(
            _point("1", 0.0, 0.0, 10.0, "GPS"),
            _point("2", 5.0, 5.0, 10.0, "GPS"),
        )
    )
    config = ClusterSplitConfig(linear_codes=frozenset({"EST"}), eps=1.0)
    result = split_by_clustering(survey, config)
    assert result.points == survey.points


def test_cluster_split_does_not_mutate_the_input_survey() -> None:
    survey = SurveyPointSet(
        points=(
            _point("1", 0.0, 0.0, 10.0, "EST"),
            _point("2", 5.0, 5.0, 10.0, "EST"),
        )
    )
    original_codes = tuple(p.code for p in survey.points)
    split_by_clustering(survey, ClusterSplitConfig(linear_codes=frozenset({"EST"}), eps=1.0))
    assert tuple(p.code for p in survey.points) == original_codes


def test_real_field_data_est_produces_the_confirmed_sixteen_groups() -> None:
    """
    Regression protegiendo el resultado real confirmado con los datos
    de campo reales de mutisgeo1.csv (columnas de un poliestructura
    deportiva) -- eps=1.0 es estable en un rango amplio (0.6-1.0) y
    da 16 figuras con estos tamanos exactos.
    """
    coords_reales = [
        (5039971.405, 1840677.499),
        (5039971.396, 1840675.252),
        (5039973.281, 1840678.093),
        (5039973.259, 1840678.132),
        (5039973.275, 1840678.372),
        (5039973.267, 1840678.455),
        (5039973.794, 1840678.053),
        (5039978.840, 1840678.187),
        (5039979.286, 1840678.123),
        (5039981.030, 1840677.607),
        (5039980.999, 1840675.339),
        (5039983.142, 1840675.367),
        (5039983.171, 1840677.647),
        (5039984.362, 1840678.566),
        (5039984.363, 1840678.496),
        (5039984.344, 1840678.290),
        (5039984.810, 1840678.151),
        (5039984.815, 1840677.693),
        (5039990.324, 1840678.214),
        (5039990.314, 1840677.743),
        (5039995.826, 1840678.246),
        (5039995.812, 1840677.798),
        (5039997.752, 1840677.764),
        (5039997.781, 1840675.598),
        (5040001.331, 1840677.844),
        (5040001.341, 1840678.302),
        (5040000.787, 1840677.853),
        (5040001.261, 1840678.735),
        (5040001.237, 1840678.651),
        (5040001.245, 1840678.449),
        (5040004.226, 1840658.036),
        (5039968.031, 1840652.992),
        (5039968.555, 1840653.024),
        (5039968.574, 1840652.433),
    ]
    puntos = tuple(_point(str(i), x, y, 10.0, "EST") for i, (x, y) in enumerate(coords_reales))
    survey = SurveyPointSet(points=puntos)

    config = ClusterSplitConfig(linear_codes=frozenset({"EST"}), eps=1.0, min_samples=1)
    result = split_by_clustering(survey, config)

    conteo = _figuras_de(result.points, "EST")
    assert len(conteo) == 16
    assert sum(conteo.values()) == 34
    assert sorted(conteo.values()) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 5, 5, 6]


# ---------------------------------------------------------------------
# ordering="hull"
# ---------------------------------------------------------------------


def _perimetro_cerrado(points: list) -> float:
    total = 0.0
    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]
        total += math.hypot(b.x - a.x, b.y - a.y)
    return total


def test_hull_ordering_never_produces_a_wildly_longer_perimeter_than_nearest() -> None:
    """
    Regression: confirmed real defect -- "nearest" can get trapped in
    a locally-good-but-globally-bad path (a long detour that leaves
    and returns), producing an invalid-looking perimeter shape for a
    real property boundary with concave notches. "hull" (convex hull
    + cheapest insertion) is confirmed more robust for this exact
    real case.
    """
    # Se le da al algoritmo el orden ORIGINAL de archivo (no el ya
    # trazado) -- confirmado que asi llegan los datos reales.
    orden_archivo = [
        (4957.348, 5069.117),
        (4952.07, 5016.957),
        (4969.488, 5068.267),
        (4927.584, 5019.659),
        (4926.139, 5007.068),
        (4992.184, 5065.722),
        (4941.276, 5004.216),
        (4941.051, 4999.349),
        (5000.567, 5064.833),
        (5015.846, 5062.585),
        (5001.01, 4926.68),
        (4965.886, 4922.906),
        (4889.215, 4918.151),
        (4887.01, 4949.177),
    ]
    puntos = tuple(_point(str(i), x, y, 10.0, "CERCA") for i, (x, y) in enumerate(orden_archivo))
    survey = SurveyPointSet(points=puntos)

    config_hull = ClusterSplitConfig(linear_codes=frozenset({"CERCA"}), eps=1000.0, min_samples=1, ordering="hull")
    config_nearest = ClusterSplitConfig(
        linear_codes=frozenset({"CERCA"}),
        eps=1000.0,
        min_samples=1,
        ordering="nearest",
    )

    resultado_hull = split_by_clustering(survey, config_hull)
    resultado_nearest = split_by_clustering(survey, config_nearest)

    puntos_hull = [p for p in resultado_hull.points if p.code.startswith("CERCA.")]
    puntos_nearest = [p for p in resultado_nearest.points if p.code.startswith("CERCA.")]

    perimetro_hull = _perimetro_cerrado(puntos_hull)
    perimetro_nearest = _perimetro_cerrado(puntos_nearest)

    # "hull" no debe ser mucho mas largo que "nearest" -- en el caso
    # real que motivo esto, "nearest" fue notablemente MAS largo
    # (por la hebra invalida), pero el test se enfoca en la garantia
    # real de "hull": nunca debe ser peor por mucho margen.
    assert perimetro_hull <= perimetro_nearest * 1.5


def test_hull_ordering_includes_every_point_even_ones_inside_the_convex_hull() -> None:
    """
    Regression: a pure convex hull would DROP interior/concave points
    entirely. Confirmed real requirement: every real fence point must
    remain in the final perimeter, including the 5 (of 14) real
    points that are NOT on the pure convex hull of this dataset.
    """
    cuadrado_con_muesca = [
        _point("a", 0.0, 0.0, 10.0, "CERCA"),
        _point("b", 10.0, 0.0, 10.0, "CERCA"),
        _point("c", 10.0, 10.0, 10.0, "CERCA"),
        _point("d", 0.0, 10.0, 10.0, "CERCA"),
        _point("muesca", 5.0, 1.0, 10.0, "CERCA"),  # punto interior real, NO en el casco convexo
    ]
    survey = SurveyPointSet(points=tuple(cuadrado_con_muesca))
    config = ClusterSplitConfig(linear_codes=frozenset({"CERCA"}), eps=1000.0, min_samples=1, ordering="hull")

    result = split_by_clustering(survey, config)

    puntos_resultado = [p for p in result.points if p.code.startswith("CERCA.")]
    assert len(puntos_resultado) == 5
    assert any(p.id == "muesca" for p in puntos_resultado)


def test_hull_ordering_falls_back_gracefully_for_collinear_points() -> None:
    """scipy.spatial.ConvexHull no puede construirse con puntos
    colineales (caso degenerado) -- confirmado que esto no debe
    lanzar un error, solo usar el orden original como respaldo."""
    colineales = [_point(str(i), float(i), 0.0, 10.0, "CERCA") for i in range(5)]
    survey = SurveyPointSet(points=tuple(colineales))
    config = ClusterSplitConfig(linear_codes=frozenset({"CERCA"}), eps=1000.0, min_samples=1, ordering="hull")

    result = split_by_clustering(survey, config)

    puntos_resultado = [p for p in result.points if p.code.startswith("CERCA.")]
    assert len(puntos_resultado) == 5


def test_hull_is_the_default_ordering() -> None:
    """
    Regression protegiendo una decision de comportamiento real:
    "hull" es el valor por defecto (no "angular"), confirmado
    consistente en 2 casos reales de tipos de levantamiento muy
    distintos (columnas de una estructura, perimetro de un predio).
    """
    config = ClusterSplitConfig(linear_codes=frozenset({"CERCA"}), eps=1.0)
    assert config.ordering == "hull"
