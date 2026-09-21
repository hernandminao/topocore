from __future__ import annotations

import pytest

from topocore.pipeline.exceptions import PropertyBoundaryError
from topocore.pipeline.gates import validate_property_boundary


def test_accepts_a_valid_simple_polygon() -> None:
    cuadrado = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert validate_property_boundary(cuadrado) == 100.0


def test_rejects_fewer_than_three_points() -> None:
    with pytest.raises(PropertyBoundaryError, match="at least 3"):
        validate_property_boundary([(0.0, 0.0), (1.0, 1.0)])


def test_rejects_a_self_intersecting_polygon() -> None:
    bowtie = [(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)]
    with pytest.raises(PropertyBoundaryError, match="self-intersects"):
        validate_property_boundary(bowtie)


def test_rejects_collinear_points_with_zero_area() -> None:
    colineales = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    with pytest.raises(PropertyBoundaryError, match="zero or negative area"):
        validate_property_boundary(colineales)


def test_real_field_data_hull_ordering_passes_all_gates() -> None:
    perimetro_hull = [
        (4918.151, 4889.215),
        (4949.177, 4887.01),
        (5015.846, 5062.585),
        (5000.567, 5064.833),
        (4992.184, 5065.722),
        (4969.488, 5068.267),
        (4957.348, 5069.117),
        (4952.07, 5016.957),
        (4941.276, 5004.216),
        (4941.051, 4999.349),
        (4927.584, 5019.659),
        (4926.139, 5007.068),
        (4926.68, 5001.01),
        (4922.906, 4965.886),
    ]
    area = validate_property_boundary(perimetro_hull)
    assert area == pytest.approx(9106.68, abs=0.1)


def test_real_field_data_nearest_ordering_is_correctly_rejected() -> None:
    """
    Regression: confirmado con datos reales -- este perimetro,
    generado con ordering="nearest" para este caso especifico, es
    geometricamente invalido (se autointerseca) y debe ser
    rechazado, no aceptado silenciosamente.
    """
    perimetro_nearest_defectuoso = [
        (4957.348, 5069.117),
        (4969.488, 5068.267),
        (4992.184, 5065.722),
        (5000.567, 5064.833),
        (5015.846, 5062.585),
        (4952.07, 5016.957),
        (4941.276, 5004.216),
        (4941.051, 4999.349),
        (4926.68, 5001.01),
        (4926.139, 5007.068),
        (4927.584, 5019.659),
        (4922.906, 4965.886),
        (4918.151, 4889.215),
        (4949.177, 4887.01),
    ]
    with pytest.raises(PropertyBoundaryError, match="self-intersects"):
        validate_property_boundary(perimetro_nearest_defectuoso)
