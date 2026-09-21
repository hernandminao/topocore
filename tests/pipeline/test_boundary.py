from __future__ import annotations

from topocore.pipeline.boundary import has_self_intersections, shoelace_area


def test_shoelace_area_of_a_simple_square() -> None:
    cuadrado = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert shoelace_area(cuadrado) == 100.0


def test_shoelace_area_is_independent_of_winding_direction() -> None:
    horario = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    antihorario = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert shoelace_area(horario) == shoelace_area(antihorario) == 100.0


def test_shoelace_area_of_fewer_than_three_points_is_zero() -> None:
    assert shoelace_area([]) == 0.0
    assert shoelace_area([(0.0, 0.0)]) == 0.0
    assert shoelace_area([(0.0, 0.0), (1.0, 1.0)]) == 0.0


def test_no_self_intersection_for_a_simple_square() -> None:
    cuadrado = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert has_self_intersections(cuadrado) is False


def test_self_intersection_detected_for_a_bowtie() -> None:
    bowtie = [(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)]
    assert has_self_intersections(bowtie) is True


def test_adjacent_sides_sharing_a_vertex_are_not_a_false_positive() -> None:
    """Un poligono valido con muchos lados no debe reportar cruce solo
    porque lados adyacentes comparten un vertice."""
    estrella_no_convexa = [
        (0.0, 0.0), (4.0, 1.0), (5.0, 5.0), (6.0, 1.0),
        (10.0, 0.0), (7.0, -3.0), (5.0, -1.0), (3.0, -3.0),
    ]
    assert has_self_intersections(estrella_no_convexa) is False


def test_fewer_than_four_points_never_self_intersects() -> None:
    assert has_self_intersections([(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]) is False


def test_real_field_data_hull_ordering_is_valid_and_nearest_is_not() -> None:
    """
    Regression: confirmado con datos reales (Survey.csv) -- el
    resultado de ordering="hull" es un poligono valido, mientras que
    ordering="nearest" (para este caso especifico) se autointerseca,
    exactamente el defecto real que se detecto visualmente antes de
    existir esta prueba automatica.
    """
    perimetro_hull = [
        (4918.151, 4889.215), (4949.177, 4887.01), (5015.846, 5062.585),
        (5000.567, 5064.833), (4992.184, 5065.722), (4969.488, 5068.267),
        (4957.348, 5069.117), (4952.07, 5016.957), (4941.276, 5004.216),
        (4941.051, 4999.349), (4927.584, 5019.659), (4926.139, 5007.068),
        (4926.68, 5001.01), (4922.906, 4965.886),
    ]
    perimetro_nearest_defectuoso = [
        (4957.348, 5069.117), (4969.488, 5068.267), (4992.184, 5065.722),
        (5000.567, 5064.833), (5015.846, 5062.585), (4952.07, 5016.957),
        (4941.276, 5004.216), (4941.051, 4999.349), (4926.68, 5001.01),
        (4926.139, 5007.068), (4927.584, 5019.659), (4922.906, 4965.886),
        (4918.151, 4889.215), (4949.177, 4887.01),
    ]
    assert has_self_intersections(perimetro_hull) is False
    assert has_self_intersections(perimetro_nearest_defectuoso) is True
