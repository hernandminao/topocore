from __future__ import annotations

import pytest
from topocore.pipeline.exceptions import MissingReferenceCodeError
from topocore.pipeline.models import BoundaryStrategy, SurveyPipelineConfig, SurveyType
from topocore.pipeline.pipeline import SurveyPipeline
from topocore.survey.formats import SurveyFormat

_ROAD_CSV = """\
1,0,0,10.0,EJE
2,0,10,10.2,EJE
3,0,20,10.5,EJE
4,2,0,10.0,BORDE
5,2,10,10.2,BORDE
6,2,20,10.5,BORDE
7,-2,0,10.0,BORDE
8,-2,10,10.2,BORDE
9,-2,20,10.5,BORDE
10,5,5,9.8,TN
11,-5,5,9.9,TN
12,5,15,10.2,TN
13,-5,15,10.3,TN
14,0,10,10.0,TN
"""

_PROPERTY_CSV = """\
1,0,0,100.0,CERCA
2,10,0,100.5,CERCA
3,10,10,101.0,CERCA
4,5,15,100.8,CERCA
5,0,10,100.2,CERCA
6,5,5,99.9,TN
7,3,3,99.8,TN
8,7,3,99.7,TN
9,3,7,100.1,TN
10,7,7,100.0,TN
"""


def test_road_pipeline_runs_end_to_end(tmp_path) -> None:
    survey_path = tmp_path / "via.csv"
    survey_path.write_text(_ROAD_CSV)

    config = SurveyPipelineConfig(
        survey_type=SurveyType.ROAD,
        survey_format=SurveyFormat.PENZD,
        reference_code="EJE",
        linear_codes=frozenset({"EJE", "BORDE"}),
        splittable_codes=frozenset({"BORDE"}),
        crs_epsg=None,
        generate_quality_report=False,
        output_dir=tmp_path / "outputs",
        output_name="via",
    )
    result = SurveyPipeline(config).run(survey_path)

    assert result.points_read == 14
    assert result.ground_count == 5
    assert result.tin.vertex_count == 5
    assert result.dxf_path.exists()


def test_road_pipeline_stops_without_the_reference_code(tmp_path) -> None:
    sin_eje = _ROAD_CSV.replace("EJE", "OTRO")
    survey_path = tmp_path / "via.csv"
    survey_path.write_text(sin_eje)

    config = SurveyPipelineConfig(
        survey_type=SurveyType.ROAD,
        survey_format=SurveyFormat.PENZD,
        reference_code="EJE",
        output_dir=tmp_path / "outputs",
    )
    with pytest.raises(MissingReferenceCodeError, match="EJE"):
        SurveyPipeline(config).run(survey_path)


def test_property_pipeline_runs_end_to_end_with_hull_boundary(tmp_path) -> None:
    survey_path = tmp_path / "predio.csv"
    survey_path.write_text(_PROPERTY_CSV)

    config = SurveyPipelineConfig(
        survey_type=SurveyType.PROPERTY,
        survey_format=SurveyFormat.PENZD,
        boundary_strategy=BoundaryStrategy.CLUSTER_HULL,
        boundary_code="CERCA",
        boundary_eps=1000.0,
        crs_epsg=None,
        generate_quality_report=False,
        output_dir=tmp_path / "outputs",
        output_name="predio",
    )
    result = SurveyPipeline(config).run(survey_path)

    assert result.points_read == 10
    assert result.ground_count == 5
    assert result.boundary_area is not None
    assert result.boundary_area > 0
    assert result.dxf_path.exists()


def test_property_pipeline_without_boundary_strategy_skips_boundary_validation(tmp_path) -> None:
    survey_path = tmp_path / "predio.csv"
    survey_path.write_text(_PROPERTY_CSV)

    config = SurveyPipelineConfig(
        survey_type=SurveyType.PROPERTY,
        survey_format=SurveyFormat.PENZD,
        boundary_strategy=BoundaryStrategy.NONE,
        crs_epsg=None,
        generate_quality_report=False,
        output_dir=tmp_path / "outputs",
        output_name="predio",
    )
    result = SurveyPipeline(config).run(survey_path)

    assert result.boundary_area is None


def test_pipeline_stops_with_insufficient_ground_points(tmp_path) -> None:
    from topocore.pipeline.exceptions import InsufficientDataError

    survey_path = tmp_path / "pocos_puntos.csv"
    survey_path.write_text("1,0,0,10.0,TN\n2,5,5,10.0,TN\n")

    config = SurveyPipelineConfig(
        survey_type=SurveyType.GENERIC,
        survey_format=SurveyFormat.PENZD,
        generate_quality_report=False,
        output_dir=tmp_path / "outputs",
    )
    with pytest.raises(InsufficientDataError, match="GROUND"):
        SurveyPipeline(config).run(survey_path)


def test_quality_report_is_generated_when_requested(tmp_path) -> None:
    survey_path = tmp_path / "predio.csv"
    survey_path.write_text(_PROPERTY_CSV)

    config = SurveyPipelineConfig(
        survey_type=SurveyType.PROPERTY,
        survey_format=SurveyFormat.PENZD,
        boundary_strategy=BoundaryStrategy.CLUSTER_HULL,
        boundary_code="CERCA",
        boundary_eps=1000.0,
        crs_epsg=None,
        generate_quality_report=True,
        output_dir=tmp_path / "outputs",
        output_name="predio",
    )
    result = SurveyPipeline(config).run(survey_path)

    assert result.quality_report is not None


def test_build_terrain_false_skips_tin_dtm_contours_without_error(tmp_path) -> None:
    """
    Regression: confirmado con datos reales (un predio catastral puro
    con estructuras/linderos, sin ningun disparo de terreno) -- exigir
    un TIN ahi es un error real, no una condicion excepcional.
    build_terrain=False permite procesar features/DXF/GPKG sin
    terreno en absoluto.
    """
    solo_estructuras = "1,0,0,10.0,ARBOL\n2,5,5,10.0,ARBOL\n3,10,10,10.0,ARBOL\n"
    survey_path = tmp_path / "sin_terreno.csv"
    survey_path.write_text(solo_estructuras)

    config = SurveyPipelineConfig(
        survey_type=SurveyType.GENERIC,
        survey_format=SurveyFormat.PENZD,
        build_terrain=False,
        crs_epsg=None,
        generate_quality_report=True,
        output_dir=tmp_path / "outputs",
        output_name="sin_terreno",
    )
    result = SurveyPipeline(config).run(survey_path)

    assert result.ground_count == 0
    assert result.tin is None
    assert result.dtm is None
    assert result.contours == ()
    assert result.geotiff_path is None
    assert result.dxf_path.exists()


def test_quality_report_is_written_to_disk_as_json(tmp_path) -> None:
    survey_path = tmp_path / "predio.csv"
    survey_path.write_text(_PROPERTY_CSV)

    config = SurveyPipelineConfig(
        survey_type=SurveyType.PROPERTY,
        survey_format=SurveyFormat.PENZD,
        boundary_strategy=BoundaryStrategy.CLUSTER_HULL,
        boundary_code="CERCA",
        boundary_eps=1000.0,
        crs_epsg=None,
        generate_quality_report=True,
        output_dir=tmp_path / "outputs",
        output_name="predio",
    )
    result = SurveyPipeline(config).run(survey_path)

    assert result.quality_report_path is not None
    assert result.quality_report_path.exists()
    assert result.quality_report_path.read_text()  # no esta vacio


def test_cross_sections_are_generated_for_a_road_when_requested(tmp_path) -> None:
    survey_path = tmp_path / "via.csv"
    survey_path.write_text(_ROAD_CSV)

    config = SurveyPipelineConfig(
        survey_type=SurveyType.ROAD,
        survey_format=SurveyFormat.PENZD,
        reference_code="EJE",
        linear_codes=frozenset({"EJE", "BORDE"}),
        splittable_codes=frozenset({"BORDE"}),
        crs_epsg=None,
        generate_quality_report=False,
        generate_cross_sections=True,
        cross_section_width=3.0,
        cross_section_interval=10.0,
        output_dir=tmp_path / "outputs",
        output_name="via",
    )
    result = SurveyPipeline(config).run(survey_path)

    assert result.cross_sections_csv_path is not None
    assert result.cross_sections_csv_path.exists()


def test_cross_sections_land_on_clean_interval_multiples_not_eje_vertices(tmp_path) -> None:
    """
    Regression: confirmado con datos reales -- las secciones deben
    generarse cada cross_section_interval metros (estaciones limpias:
    0, 10, 20...), no una por cada vertice real del EJE (que producia
    estaciones en distancias irregulares segun donde el topografo
    tomo cada punto).
    """
    survey_path = tmp_path / "via.csv"
    survey_path.write_text(_ROAD_CSV)

    config = SurveyPipelineConfig(
        survey_type=SurveyType.ROAD,
        survey_format=SurveyFormat.PENZD,
        reference_code="EJE",
        linear_codes=frozenset({"EJE", "BORDE"}),
        splittable_codes=frozenset({"BORDE"}),
        crs_epsg=None,
        generate_quality_report=False,
        generate_cross_sections=True,
        cross_section_width=3.0,
        cross_section_interval=10.0,
        output_dir=tmp_path / "outputs",
        output_name="via",
    )
    result = SurveyPipeline(config).run(survey_path)

    assert result.cross_sections_csv_path is not None
    contenido = result.cross_sections_csv_path.read_text()
    estaciones = sorted({float(linea.split(",")[0]) for linea in contenido.strip().splitlines()[1:]})
    assert estaciones == [10.0]


def test_cross_sections_are_not_generated_when_not_requested(tmp_path) -> None:
    survey_path = tmp_path / "via.csv"
    survey_path.write_text(_ROAD_CSV)

    config = SurveyPipelineConfig(
        survey_type=SurveyType.ROAD,
        survey_format=SurveyFormat.PENZD,
        reference_code="EJE",
        linear_codes=frozenset({"EJE", "BORDE"}),
        splittable_codes=frozenset({"BORDE"}),
        crs_epsg=None,
        generate_quality_report=False,
        generate_cross_sections=False,
        output_dir=tmp_path / "outputs",
        output_name="via",
    )
    result = SurveyPipeline(config).run(survey_path)

    assert result.cross_sections_csv_path is None
    assert result.cross_sections_dxf_path is None


def test_cross_sections_are_skipped_for_property_surveys_even_if_requested(tmp_path) -> None:
    """generate_cross_sections solo aplica a survey_type=ROAD -- para
    PROPERTY, no hay eje/centerline al que referirlas."""
    survey_path = tmp_path / "predio.csv"
    survey_path.write_text(_PROPERTY_CSV)

    config = SurveyPipelineConfig(
        survey_type=SurveyType.PROPERTY,
        survey_format=SurveyFormat.PENZD,
        boundary_strategy=BoundaryStrategy.CLUSTER_HULL,
        boundary_code="CERCA",
        boundary_eps=1000.0,
        crs_epsg=None,
        generate_quality_report=False,
        generate_cross_sections=True,
        output_dir=tmp_path / "outputs",
        output_name="predio",
    )
    result = SurveyPipeline(config).run(survey_path)

    assert result.cross_sections_csv_path is None
