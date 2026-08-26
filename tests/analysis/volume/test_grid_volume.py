"""
Targeted coverage suite for topocore.analysis.volume.grid_volume,
.cut_fill, .average_end_area, .prismoidal, .tin_volume, and .manager
-- PR20 coverage phase.

Includes one real, severe bug found and fixed in this session:

GridVolume.compute_from_dtm() previously used the INSTANCE's own
resolution/cell_area unconditionally, never checking it against the
DTMs' own actual resolution. Confirmed directly: a
GridVolume(resolution=1.0) fed real DTMs at resolution=2.0 silently
computed a cut volume exactly 4x too small (cell_area scales as
resolution-squared), with no error or warning -- a genuine risk of a
material earthwork-quantity error. Unlike CutFillVolume.compute_with_dtm()
(which sidesteps the whole issue by always deriving cell_area fresh
from the DTM), fixed here per explicit instruction by validating the
instance's resolution against the DTM's own and raising VolumeError
on mismatch, since GridVolume's resolution is a real, publicly
exposed configuration property that a caller relies on.

Also formalizes fresh manual verification (confirmed no further bugs)
into permanent regressions: AverageEndAreaVolume/PrismoidalVolume/
TINVolume verified against exact known values -- including
PrismoidalVolume's Simpson's-rule result matching an exact analytic
integral of a quadratic area profile (1365.333..., confirming the
PR19 fix documented in this module's own source comments remains
correct) -- plus VolumeAnalysis's full dispatcher (all 5 methods) and
VolumeMethod being genuinely a StrEnum (confirming the document's
"if it's a StrEnum, this is fine" conditional check was correct, no
change needed).

No other bugs found -- only test coverage was added elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis.exceptions import VolumeError
from topocore.analysis.types import VolumeMethod
from topocore.analysis.volume.average_end_area import AverageEndAreaVolume
from topocore.analysis.volume.cut_fill import CutFillVolume
from topocore.analysis.volume.grid_volume import GridVolume
from topocore.analysis.volume.manager import VolumeAnalysis
from topocore.analysis.volume.prismoidal import PrismoidalVolume
from topocore.analysis.volume.tin_volume import TINVolume
from topocore.geometry.point3d import Point3D
from topocore.terrain.dtm import DTM
from topocore.terrain.grid import Grid
from topocore.terrain.raster import Raster
from topocore.terrain.tin import TIN


def _make_dtm(grid: Grid, value: float) -> DTM:
    raster = Raster(grid=grid, values=np.full((grid.rows, grid.columns), value))
    return DTM(tin=None, grid=grid, raster=raster)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# GridVolume.compute_from_dtm() -- the real bug.
# ----------------------------------------------------------------------


def test_grid_volume_from_dtm_rejects_resolution_mismatch() -> None:
    """
    The exact regression: before the fix, a mismatched instance
    resolution silently computed a volume off by resolution-squared,
    with no error.
    """
    grid = Grid(min_x=0, min_y=0, max_x=10, max_y=10, resolution=2.0)
    existing_dtm = _make_dtm(grid, 110.0)
    proposed_dtm = _make_dtm(grid, 100.0)

    with pytest.raises(VolumeError, match="does not match"):
        GridVolume(resolution=1.0).compute_from_dtm(existing_dtm, proposed_dtm)


def test_grid_volume_from_dtm_succeeds_with_matching_resolution() -> None:
    grid = Grid(min_x=0, min_y=0, max_x=10, max_y=10, resolution=2.0)
    existing_dtm = _make_dtm(grid, 110.0)
    proposed_dtm = _make_dtm(grid, 100.0)

    result = GridVolume(resolution=2.0).compute_from_dtm(existing_dtm, proposed_dtm)

    # Confirmed value (grid dimensions from ceil((max-min)/resolution))
    assert result.cut_volume == pytest.approx(1440.0)


def test_grid_volume_from_dtm_rejects_mismatched_grid_geometry() -> None:
    grid_a = Grid(min_x=0, min_y=0, max_x=10, max_y=10, resolution=2.0)
    grid_b = Grid(min_x=0, min_y=0, max_x=20, max_y=20, resolution=2.0)
    existing_dtm = _make_dtm(grid_a, 110.0)
    proposed_dtm = _make_dtm(grid_b, 100.0)

    with pytest.raises(VolumeError, match="same grid geometry"):
        GridVolume(resolution=2.0).compute_from_dtm(existing_dtm, proposed_dtm)


def test_grid_volume_properties_and_validation() -> None:
    gv = GridVolume(resolution=2.0)
    assert gv.resolution == pytest.approx(2.0)
    assert gv.cell_area == pytest.approx(4.0)

    with pytest.raises(VolumeError, match="positive"):
        GridVolume(resolution=0.0)

    with pytest.raises(VolumeError, match="finite"):
        GridVolume(resolution=float("nan"))


def test_grid_volume_call_matches_compute() -> None:
    gv = GridVolume(resolution=1.0)
    existing = np.full((2, 2), 110.0)
    proposed = np.full((2, 2), 100.0)
    assert gv(existing, proposed).cut_volume == gv.compute(existing, proposed).cut_volume


# ----------------------------------------------------------------------
# CutFillVolume -- remaining gaps.
# ----------------------------------------------------------------------


def test_cut_fill_compute_with_dtm_derives_cell_area_from_dtm() -> None:
    grid = Grid(min_x=0, min_y=0, max_x=10, max_y=10, resolution=2.0)
    existing_dtm = _make_dtm(grid, 110.0)
    proposed_dtm = _make_dtm(grid, 100.0)

    # cell_area at construction (1.0) is irrelevant -- the DTM's own resolution (2.0) is used.
    result = CutFillVolume(cell_area=1.0).compute_with_dtm(existing_dtm, proposed_dtm)

    assert result.cut_volume == pytest.approx(1440.0)


def test_cut_fill_properties_and_validation() -> None:
    cfv = CutFillVolume(cell_area=4.0)
    assert cfv.cell_area == pytest.approx(4.0)

    with pytest.raises(VolumeError, match="positive"):
        CutFillVolume(cell_area=0.0)

    with pytest.raises(VolumeError, match="finite"):
        CutFillVolume(cell_area=float("inf"))


def test_cut_fill_call_matches_compute() -> None:
    cfv = CutFillVolume(cell_area=1.0)
    existing = np.full((2, 2), 110.0)
    proposed = np.full((2, 2), 100.0)
    assert cfv(existing, proposed).cut_volume == cfv.compute(existing, proposed).cut_volume


def test_cut_fill_with_dtm_rejects_mismatched_geometry() -> None:
    grid_a = Grid(min_x=0, min_y=0, max_x=10, max_y=10, resolution=2.0)
    grid_b = Grid(min_x=0, min_y=0, max_x=20, max_y=20, resolution=2.0)
    existing_dtm = _make_dtm(grid_a, 110.0)
    proposed_dtm = _make_dtm(grid_b, 100.0)

    with pytest.raises(VolumeError, match="same grid geometry"):
        CutFillVolume(cell_area=4.0).compute_with_dtm(existing_dtm, proposed_dtm)


# ----------------------------------------------------------------------
# AverageEndAreaVolume
# ----------------------------------------------------------------------


def test_average_end_area_known_three_section_volume() -> None:
    aea = AverageEndAreaVolume([(0.0, 100.0), (10.0, 200.0), (20.0, 100.0)])
    result = aea.compute()
    assert result.net_volume == pytest.approx(3000.0)
    assert aea.segment_volumes() == [pytest.approx(1500.0), pytest.approx(1500.0)]
    assert aea.section_count == 3


def test_average_end_area_call_matches_compute() -> None:
    aea = AverageEndAreaVolume([(0.0, 100.0), (10.0, 200.0)])
    assert aea().net_volume == aea.compute().net_volume


@pytest.mark.parametrize(
    ("sections", "match"),
    [
        ([(0.0, 100.0)], "at least 2"),
        ([(0.0, 100.0), (0.0, 100.0)], "increasing station"),
        ([(0.0, -5.0), (10.0, 100.0)], "cannot be negative"),
        ([(0.0, 100.0), (float("nan"), 100.0)], "station must be finite"),
    ],
)
def test_average_end_area_rejects_invalid_sections(sections: list, match: str) -> None:  # type: ignore[type-arg]
    with pytest.raises(VolumeError, match=match):
        AverageEndAreaVolume(sections)


# ----------------------------------------------------------------------
# PrismoidalVolume
# ----------------------------------------------------------------------


def test_prismoidal_matches_exact_analytic_integral() -> None:
    """Confirms the PR19 fix remains correct: Simpson's rule is exact for a quadratic area profile."""
    stations = [0.0, 4.0, 8.0, 12.0, 16.0]
    sections = [(s, s**2) for s in stations]
    pv = PrismoidalVolume(sections)

    result = pv.compute()
    exact = 16**3 / 3

    assert result.net_volume == pytest.approx(exact)
    assert sum(pv.segment_volumes()) == pytest.approx(result.net_volume)
    assert pv.section_count == 5


def test_prismoidal_call_matches_compute() -> None:
    pv = PrismoidalVolume([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])
    assert pv().net_volume == pv.compute().net_volume


@pytest.mark.parametrize(
    ("sections", "match"),
    [
        ([(0.0, 1.0), (1.0, 2.0)], "at least three"),
        ([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)], "odd number"),
        ([(0.0, 1.0), (1.0, 2.0), (0.5, 3.0)], "strictly increasing"),
        ([(0.0, 1.0), (1.0, 2.0), (3.0, 3.0)], "uniformly spaced"),
        ([(0.0, -1.0), (1.0, 2.0), (2.0, 3.0)], "cannot be negative"),
    ],
)
def test_prismoidal_rejects_invalid_sections(sections: list, match: str) -> None:  # type: ignore[type-arg]
    with pytest.raises(VolumeError, match=match):
        PrismoidalVolume(sections)


# ----------------------------------------------------------------------
# TINVolume
# ----------------------------------------------------------------------


def test_tin_volume_flat_square_known_volume() -> None:
    points = (
        Point3D(0, 0, 5.0),
        Point3D(10, 0, 5.0),
        Point3D(0, 10, 5.0),
        Point3D(10, 10, 5.0),
    )
    tin = TIN.from_points(points)

    result = TINVolume(datum=0.0).compute(tin)

    assert result.net_volume == pytest.approx(500.0)
    assert result.fill_volume == pytest.approx(0.0)


def test_tin_volume_below_datum_is_fill_not_cut() -> None:
    points = (
        Point3D(0, 0, 5.0),
        Point3D(10, 0, 5.0),
        Point3D(0, 10, 5.0),
        Point3D(10, 10, 5.0),
    )
    tin = TIN.from_points(points)

    result = TINVolume(datum=10.0).compute(tin)

    assert result.cut_volume == pytest.approx(0.0)
    assert result.fill_volume == pytest.approx(500.0)
    assert result.net_volume == pytest.approx(-500.0)


def test_tin_volume_datum_property() -> None:
    assert TINVolume(datum=5.0).datum == pytest.approx(5.0)


def test_tin_volume_call_matches_compute() -> None:
    points = (Point3D(0, 0, 1.0), Point3D(1, 0, 1.0), Point3D(0, 1, 1.0))
    tin = TIN.from_points(points)
    tv = TINVolume()
    assert tv(tin).net_volume == tv.compute(tin).net_volume


def test_tin_volume_rejects_nonfinite_datum() -> None:
    with pytest.raises(VolumeError, match="finite"):
        TINVolume(datum=float("nan"))


# ----------------------------------------------------------------------
# VolumeAnalysis manager -- dispatcher and VolumeMethod StrEnum.
# ----------------------------------------------------------------------


def test_volume_method_is_a_strenum() -> None:
    """Confirms the document's uncertainty: VolumeMethod.CUT_FILL.value == 'cut_fill', and == works directly."""
    assert VolumeMethod.CUT_FILL == "cut_fill"
    assert str(VolumeMethod.CUT_FILL) == "cut_fill"


def test_volume_analysis_dispatches_prismoidal() -> None:
    analysis = VolumeAnalysis(method="prismoidal")
    sections = [(0.0, 1.0), (1.0, 2.0), (2.0, 1.0)]
    result = analysis.compute(sections, method="prismoidal")
    assert result.method == "prismoidal"


def test_volume_analysis_dispatches_average_end_area() -> None:
    analysis = VolumeAnalysis()
    sections = [(0.0, 100.0), (10.0, 200.0)]
    result = analysis.compute(sections, method="average_end_area")
    assert result.net_volume == pytest.approx(1500.0)


def test_volume_analysis_dispatches_tin_volume() -> None:
    points = (
        Point3D(0, 0, 5.0),
        Point3D(10, 0, 5.0),
        Point3D(0, 10, 5.0),
        Point3D(10, 10, 5.0),
    )
    tin = TIN.from_points(points)
    analysis = VolumeAnalysis()
    result = analysis.compute(tin, 0.0, method="tin_volume")
    assert result.net_volume == pytest.approx(500.0)


def test_volume_analysis_dispatches_grid_volume() -> None:
    analysis = VolumeAnalysis()
    existing = np.full((2, 2), 110.0)
    proposed = np.full((2, 2), 100.0)
    result = analysis.compute(existing, proposed, 1.0, method="grid_volume")
    assert result.cut_volume == pytest.approx(40.0)


def test_volume_analysis_rejects_unknown_method_at_construction() -> None:
    with pytest.raises(VolumeError, match="Unsupported volume method"):
        VolumeAnalysis(method="bogus")


def test_volume_analysis_call_matches_compute() -> None:
    analysis = VolumeAnalysis()
    existing = np.full((2, 2), 110.0)
    proposed = np.full((2, 2), 100.0)
    assert (
        analysis(existing, proposed, 1.0, method="cut_fill").cut_volume
        == analysis.compute(existing, proposed, 1.0, method="cut_fill").cut_volume
    )


def test_volume_analysis_method_and_config_properties() -> None:
    analysis = VolumeAnalysis(method="prismoidal")
    assert analysis.method == "prismoidal"
    assert analysis.config is not None
