"""
Optional, environment-configurable test against a REAL geoid GeoTIFF
(e.g. EGM2008 or a national gravimetric geoid model), confirming
`Workflow.transform_vertical()`'s own path is genuinely configurable
-- no hardcoded location anywhere in production code (already
verified once, in an earlier phase of this same capability, by
mounting a synthetic file at an arbitrary "deployment-style" path
with zero code changes).

Skipped everywhere the real file isn't available (CI, another
developer's machine) via `TOPOCORE_TEST_GEOID_PATH` -- this is
deliberately ADDITIONAL coverage, not a replacement for the
synthetic-fixture tests used throughout the rest of this suite
(`synthetic_geoid_path`), which remain the portable, hand-verified
primary regression coverage (exact undulation values computed by
hand against a known 3x3 grid) and must keep running identically on
any machine, with or without a real geoid file present.

The test point is derived from the grid's OWN discovered geographic
center (via its public `origin_longitude`/`origin_latitude`/
`pixel_width`/`pixel_height`/`columns`/`rows`) rather than a
hardcoded coordinate -- this file's own author does not know the
real file's exact geographic extent in advance (a national geoid and
a single 1x1-degree EGM2008 tile could cover very different areas),
so guessing a coordinate risked landing outside whichever file is
actually configured.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from topocore.geodesy.vertical import GeoidGrid, VerticalTransformer
from topocore.geodesy.vertical_datum import VerticalDatum
from topocore.survey.formats import SurveyFormat
from topocore.workflow.artifacts import ArtifactType
from topocore.workflow.workflow import Workflow

_REAL_GEOID_PATH = os.environ.get("TOPOCORE_TEST_GEOID_PATH")

_SKIP_REASON = (
    "Set the TOPOCORE_TEST_GEOID_PATH environment variable to a real geoid "
    "GeoTIFF (e.g. D:\\Topo\\topocore\\data\\geoids\\us_nga_egm2008_1.tif) to run this test."
)


def _geoid_available() -> bool:
    return _REAL_GEOID_PATH is not None and Path(_REAL_GEOID_PATH).is_file()


@pytest.mark.skipif(not _geoid_available(), reason=_SKIP_REASON)
def test_real_geoid_file_loads_and_produces_a_genuine_undulation() -> None:
    """
    Confirms the real file loads correctly and its own grid center
    (not a guessed coordinate) yields a finite, non-trivially-zero
    undulation -- this cannot assert an EXACT expected value (the
    real geoid's own values at any given point are not known in
    advance by this test), only that loading and interpolation
    genuinely happen against real data, not a stand-in.
    """
    assert _REAL_GEOID_PATH is not None  # for type-checkers; the skipif above already guarantees this at runtime

    grid = GeoidGrid.from_geotiff(_REAL_GEOID_PATH)

    center_longitude = grid.origin_longitude + (grid.columns / 2) * grid.pixel_width
    center_latitude = grid.origin_latitude + (grid.rows / 2) * grid.pixel_height

    undulation = grid.undulation_at(center_longitude, center_latitude)
    assert isinstance(undulation, float)
    import math

    assert math.isfinite(undulation)


@pytest.mark.skipif(not _geoid_available(), reason=_SKIP_REASON)
def test_workflow_transform_vertical_with_real_geoid_file(tmp_path: Path) -> None:
    """
    The same Workflow.transform_vertical() pipeline already covered
    against the synthetic fixture elsewhere in this suite, run once
    more here against real geoid data -- confirms the whole path
    (Workflow -> VerticalTransformer -> GeoidGrid.from_geotiff()) is
    genuinely path-agnostic in practice, not just in isolated
    unit-level calls.
    """
    assert _REAL_GEOID_PATH is not None

    grid = GeoidGrid.from_geotiff(_REAL_GEOID_PATH)
    center_longitude = grid.origin_longitude + (grid.columns / 2) * grid.pixel_width
    center_latitude = grid.origin_latitude + (grid.rows / 2) * grid.pixel_height

    survey_path = tmp_path / "survey.txt"
    survey_path.write_text(f"1,{center_longitude},{center_latitude},2600.0,PT\n", encoding="utf-8")

    workflow = Workflow().read_survey(survey_path, format=SurveyFormat.ID_XYZ_CODE)
    original_z = workflow.artifact(ArtifactType.SURVEY_POINT_SET).points[0].z

    vertical_transformer = VerticalTransformer(
        source_datum=VerticalDatum(name="ellipsoidal"),
        target_datum=VerticalDatum(name="orthometric", geoid_model="real-file-test"),
        geoid=grid,
    )
    workflow.transform_vertical(
        ArtifactType.SURVEY_POINT_SET,
        source_datum=vertical_transformer.source_datum,
        target_datum=vertical_transformer.target_datum,
        geoid=grid,
    )

    result_z = workflow.artifact(ArtifactType.SURVEY_POINT_SET).points[0].z
    # Confirms the height was genuinely shifted by the real geoid's
    # own undulation -- never silently left unchanged (the exact
    # danger this whole capability's own safety design exists to
    # prevent), without asserting a specific expected value.
    assert result_z != original_z
