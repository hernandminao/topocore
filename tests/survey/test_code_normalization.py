from __future__ import annotations

from topocore.survey.code_normalization import normalize_numbered_codes
from topocore.survey.models import SurveyPoint, SurveyPointSet


def _point(id_: str, x: float, y: float, z: float, code: str | None) -> SurveyPoint:
    return SurveyPoint(id=id_, x=x, y=y, z=z, code=code)


def test_normalizes_numbered_codes_to_their_prefix() -> None:
    puntos = tuple(_point(str(i), float(i), 0.0, 10.0, f"BZ{i}") for i in range(1, 32))
    survey = SurveyPointSet(points=puntos)

    result = normalize_numbered_codes(survey, frozenset({"BZ"}))

    assert all(p.code == "BZ" for p in result.points)
    assert len(result.points) == 31


def test_preserves_the_original_point_id() -> None:
    survey = SurveyPointSet(points=(_point("902", 0.0, 0.0, 10.0, "BZ1"),))
    result = normalize_numbered_codes(survey, frozenset({"BZ"}))
    assert result.points[0].id == "902"


def test_leaves_unrelated_codes_untouched() -> None:
    survey = SurveyPointSet(points=(
        _point("1", 0.0, 0.0, 10.0, "ARBOL"),
        _point("2", 1.0, 0.0, 10.0, "PT"),
    ))
    result = normalize_numbered_codes(survey, frozenset({"BZ"}))
    assert [p.code for p in result.points] == ["ARBOL", "PT"]


def test_a_code_that_is_already_exactly_the_prefix_is_left_unchanged() -> None:
    survey = SurveyPointSet(points=(_point("1", 0.0, 0.0, 10.0, "BZ"),))
    result = normalize_numbered_codes(survey, frozenset({"BZ"}))
    assert result.points[0].code == "BZ"


def test_a_code_with_a_letter_suffix_after_the_number_is_not_matched() -> None:
    """'BZ1A' no es un codigo numerado puro (BZ + digitos) -- confirmado
    que no debe normalizarse, para no fusionar algo que podria ser
    una variante real distinta."""
    survey = SurveyPointSet(points=(_point("1", 0.0, 0.0, 10.0, "BZ1A"),))
    result = normalize_numbered_codes(survey, frozenset({"BZ"}))
    assert result.points[0].code == "BZ1A"


def test_multiple_prefixes_are_normalized_independently() -> None:
    survey = SurveyPointSet(points=(
        _point("1", 0.0, 0.0, 10.0, "BZ5"),
        _point("2", 1.0, 0.0, 10.0, "E10"),
        _point("3", 2.0, 0.0, 10.0, "N3"),
    ))
    result = normalize_numbered_codes(survey, frozenset({"BZ", "E", "N"}))
    assert [p.code for p in result.points] == ["BZ", "E", "N"]


def test_does_not_mutate_the_input_survey() -> None:
    survey = SurveyPointSet(points=(_point("1", 0.0, 0.0, 10.0, "BZ1"),))
    normalize_numbered_codes(survey, frozenset({"BZ"}))
    assert survey.points[0].code == "BZ1"


def test_a_point_with_no_code_at_all_is_left_unchanged() -> None:
    survey = SurveyPointSet(points=(_point("1", 0.0, 0.0, 10.0, None),))
    result = normalize_numbered_codes(survey, frozenset({"BZ"}))
    assert result.points[0].code is None


def test_real_field_data_from_buzones_survey() -> None:
    """
    Regression: confirmado con datos reales (BUZONES.csv) -- 31
    buzones numerados individualmente (BZ1...BZ31), cada uno con un
    id de punto real distinto (no secuencial: 902-932 en el archivo
    real), deben normalizarse a un solo codigo "BZ" sin perder
    ningun punto ni su id real.
    """
    ids_reales = [str(i) for i in range(902, 933)]
    puntos = tuple(_point(id_real, float(i), 0.0, 200.0, f"BZ{i + 1}") for i, id_real in enumerate(ids_reales))
    survey = SurveyPointSet(points=puntos)

    result = normalize_numbered_codes(survey, frozenset({"BZ"}))

    assert len(result.points) == 31
    assert all(p.code == "BZ" for p in result.points)
    assert [p.id for p in result.points] == ids_reales
