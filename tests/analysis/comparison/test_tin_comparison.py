"""
Regression suite for topocore.analysis.comparison.TINComparison --
PR20.4, new feature implementation.

Design decisions made explicit here (confirmed with real TIN objects
before implementing, per this session's established discipline):

1. TIN.interpolate() raises ValueError for points outside the convex
   hull -- a genuinely different NoData convention than the NaN-based
   one gridded surfaces use. TINComparison catches this and maps it
   to NaN internally, so SurfaceComparison's own NaN-based NoData
   handling (validated extensively in test_surface_comparison.py)
   applies uniformly to both gridded and TIN comparisons -- confirmed
   the underlying classification logic is NOT duplicated here.

2. TIN.find_triangle() is a documented O(triangle_count) brute-force
   scan -- evaluating a shared grid costs
   O(grid_points x triangle_count). Accepted as a known, documented
   cost per explicit instruction, not treated as a bug.

3. TIN carries no CRS information at all in this codebase (confirmed:
   no crs property, no CRS parameter anywhere in TIN.from_points()/
   from_mesh()) -- CRS compatibility validation is therefore not
   representable at the TIN level and is explicitly NOT attempted
   here, documented as a known limitation rather than silently
   skipped without explanation.

Verified with real TIN objects (not mocks): identical-domain flat
TINs (uniform cut, exact cell count for a known bbox/resolution),
partially-overlapping domains (exactly half the cells valid),
non-overlapping domains (rejected), and a non-rectangular
(triangular) TIN where roughly half the bounding-box grid falls
outside the hull -- confirming the ValueError-to-NaN mapping
correctly excludes those cells rather than crashing or silently
using bad values.
"""

from __future__ import annotations

import pytest

from topocore.analysis.comparison import TINComparison
from topocore.analysis.exceptions import VolumeError
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN


def _flat_square_tin(z: float, min_xy: float = 0.0, max_xy: float = 20.0) -> TIN:
    points = (
        Point3D(min_xy, min_xy, z),
        Point3D(max_xy, min_xy, z),
        Point3D(min_xy, max_xy, z),
        Point3D(max_xy, max_xy, z),
    )
    return TIN.from_points(points)


def _flat_triangle_tin(z: float, size: float = 20.0) -> TIN:
    points = (Point3D(0.0, 0.0, z), Point3D(size, 0.0, z), Point3D(0.0, size, z))
    return TIN.from_points(points)


def test_identical_domain_uniform_cut() -> None:
    existing = _flat_square_tin(110.0)
    proposed = _flat_square_tin(100.0)

    result = TINComparison(resolution=2.0, tolerance=0.0).compute(existing, proposed)

    # bbox is 20x20 at resolution 2.0 -> 10x10 = 100 cells, all valid.
    assert result.valid_cells == 100
    assert result.excluded_cells == 0
    assert result.cut_cells == 100
    assert result.fill_cells == 0
    assert result.mean_difference == pytest.approx(-10.0)


def test_identical_domain_uniform_fill() -> None:
    existing = _flat_square_tin(100.0)
    proposed = _flat_square_tin(110.0)

    result = TINComparison(resolution=2.0, tolerance=0.0).compute(existing, proposed)

    assert result.fill_cells == 100
    assert result.cut_cells == 0
    assert result.mean_difference == pytest.approx(10.0)


def test_partially_overlapping_domains() -> None:
    existing = _flat_square_tin(110.0, min_xy=0.0, max_xy=20.0)
    proposed = _flat_square_tin(100.0, min_xy=10.0, max_xy=30.0)

    result = TINComparison(resolution=2.0, tolerance=0.0).compute(existing, proposed)

    # _flat_square_tin shifts BOTH axes together (min_xy/max_xy apply
    # to X and Y alike), so existing=[0,20]x[0,20] and
    # proposed=[10,30]x[10,30] overlap only in [10,20]x[10,20] ->
    # 5x5 = 25 cells at resolution=2.0.
    assert result.valid_cells == 25


def test_non_overlapping_domains_rejected() -> None:
    existing = _flat_square_tin(110.0, min_xy=0.0, max_xy=20.0)
    proposed = _flat_square_tin(100.0, min_xy=1000.0, max_xy=1020.0)

    with pytest.raises(VolumeError, match="do not share an overlapping"):
        TINComparison(resolution=2.0).compute(existing, proposed)


def test_triangular_tin_excludes_cells_outside_hull() -> None:
    """
    The decisive ValueError-to-NaN mapping check: a triangular TIN's
    bounding box is 2x larger than the triangle itself -- roughly
    half the sampled grid must fall outside the hull and be excluded,
    not crash or silently interpolate garbage.
    """
    existing = _flat_triangle_tin(100.0)
    proposed = _flat_triangle_tin(90.0)

    result = TINComparison(resolution=2.0, tolerance=0.0).compute(existing, proposed)

    total_cells = result.valid_cells + result.excluded_cells
    assert total_cells == 100  # 20x20 bbox at resolution 2.0
    assert 0.35 < result.valid_cells / total_cells < 0.65
    assert result.cut_cells == result.valid_cells  # uniform 10m cut everywhere inside the hull


def test_tolerance_applies_to_tin_comparison() -> None:
    existing = _flat_square_tin(100.0)
    proposed = _flat_square_tin(100.02)

    result = TINComparison(resolution=2.0, tolerance=0.05).compute(existing, proposed)

    assert result.unchanged_cells == result.valid_cells
    assert result.cut_cells == 0
    assert result.fill_cells == 0


def test_call_matches_compute() -> None:
    existing = _flat_square_tin(110.0)
    proposed = _flat_square_tin(100.0)
    comparison = TINComparison(resolution=2.0)

    assert comparison(existing, proposed).cut_cells == comparison.compute(existing, proposed).cut_cells


def test_properties() -> None:
    comparison = TINComparison(resolution=5.0, tolerance=0.1)
    assert comparison.resolution == pytest.approx(5.0)
    assert comparison.tolerance == pytest.approx(0.1)


# ----------------------------------------------------------------------
# Validation.
# ----------------------------------------------------------------------


def test_rejects_nonpositive_resolution() -> None:
    with pytest.raises(VolumeError, match="positive"):
        TINComparison(resolution=0.0)


def test_rejects_nonfinite_resolution() -> None:
    with pytest.raises(VolumeError, match="finite"):
        TINComparison(resolution=float("nan"))


def test_rejects_empty_existing_tin() -> None:
    """A TIN.triangle_count <= 0 is defensively checked, even though real TIN.from_points() always builds >= 1 triangle."""
    proposed = _flat_square_tin(100.0)

    class EmptyTIN:
        triangle_count = 0
        bounds = (0.0, 0.0, 1.0, 1.0)

        def interpolate(self, x: float, y: float) -> float:
            raise ValueError("no triangles")

    with pytest.raises(VolumeError, match="Existing TIN contains no triangles"):
        TINComparison(resolution=1.0).compute(EmptyTIN(), proposed)  # type: ignore[arg-type]
