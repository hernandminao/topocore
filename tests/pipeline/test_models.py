from __future__ import annotations

import pytest

from topocore.pipeline.exceptions import SurveyPipelineError
from topocore.pipeline.models import BoundaryStrategy, SurveyPipelineConfig, SurveyType
from topocore.survey.formats import SurveyFormat


def test_generic_config_with_no_boundary_is_valid() -> None:
    config = SurveyPipelineConfig(survey_type=SurveyType.GENERIC, survey_format=SurveyFormat.PNEZD)
    assert config.boundary_strategy == BoundaryStrategy.NONE


def test_road_without_reference_code_is_rejected() -> None:
    with pytest.raises(SurveyPipelineError, match="reference_code"):
        SurveyPipelineConfig(survey_type=SurveyType.ROAD, survey_format=SurveyFormat.PNEZD)


def test_road_with_empty_string_reference_code_is_rejected() -> None:
    with pytest.raises(SurveyPipelineError, match="reference_code"):
        SurveyPipelineConfig(survey_type=SurveyType.ROAD, survey_format=SurveyFormat.PNEZD, reference_code="")


def test_road_with_reference_code_is_valid() -> None:
    config = SurveyPipelineConfig(
        survey_type=SurveyType.ROAD, survey_format=SurveyFormat.PNEZD, reference_code="EJE",
    )
    assert config.reference_code == "EJE"


def test_boundary_strategy_without_boundary_code_is_rejected() -> None:
    with pytest.raises(SurveyPipelineError, match="boundary_code"):
        SurveyPipelineConfig(
            survey_type=SurveyType.PROPERTY,
            survey_format=SurveyFormat.PNEZD,
            boundary_strategy=BoundaryStrategy.CLUSTER_HULL,
        )


def test_boundary_strategy_with_boundary_code_is_valid() -> None:
    config = SurveyPipelineConfig(
        survey_type=SurveyType.PROPERTY,
        survey_format=SurveyFormat.PNEZD,
        boundary_strategy=BoundaryStrategy.CLUSTER_HULL,
        boundary_code="CERCA",
    )
    assert config.boundary_code == "CERCA"


def test_nonpositive_contour_interval_is_rejected() -> None:
    with pytest.raises(SurveyPipelineError, match="contour_interval"):
        SurveyPipelineConfig(survey_type=SurveyType.GENERIC, survey_format=SurveyFormat.PNEZD, contour_interval=0.0)


def test_nonpositive_dtm_resolution_is_rejected() -> None:
    with pytest.raises(SurveyPipelineError, match="dtm_resolution"):
        SurveyPipelineConfig(survey_type=SurveyType.GENERIC, survey_format=SurveyFormat.PNEZD, dtm_resolution=-1.0)


def test_nonpositive_boundary_eps_is_rejected() -> None:
    with pytest.raises(SurveyPipelineError, match="boundary_eps"):
        SurveyPipelineConfig(survey_type=SurveyType.GENERIC, survey_format=SurveyFormat.PNEZD, boundary_eps=0.0)


def test_crs_epsg_none_is_valid() -> None:
    config = SurveyPipelineConfig(survey_type=SurveyType.GENERIC, survey_format=SurveyFormat.PNEZD, crs_epsg=None)
    assert config.crs_epsg is None
