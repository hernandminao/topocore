"""
Regression tests closing a full family of defects in
topocore.analysis.statistics, found during this package's own first
audit (topocore.analysis had zero prior test coverage of any kind).

3 categories, kept distinct:

- A contract defect: `PointCloudData.elevation_array`/`.xy_array`
  were declared as `@property` while `TIN` (the sole concrete
  implementation) exposes both as plain methods. Fixed by changing
  the Protocol to match TIN's own real, working, tested behavior --
  never the other way around.
- 3 functional defects, all downstream of that same contract defect,
  each its own separately-reproducible failure before this fix:
  `ElevationStatistics.from_tin()`, `StatisticsAnalysis`'s own
  duck-typed elevation dispatch, and
  `DensityStatistics.compute_from_tin()`.
- 1 independent numeric defect in `SlopeStatistics`, unrelated to the
  Protocol/TIN contract: a near-degenerate (not exactly-zero) range
  crashing `np.histogram()`.

Scoped exactly to this fix contract: these tests exist to prove the
above are fixed and non-degenerate/non-uniform behavior is unchanged
-- not to establish general coverage for the rest of
topocore.analysis, which remains a separate, later effort.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from topocore.analysis.exceptions import StatisticsError
from topocore.analysis.statistics import (
    DensityStatistics,
    ElevationStatistics,
    SlopeStatistics,
    StatisticsAnalysis,
)
from topocore.analysis.types import ElevationStats
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN

# ----------------------------------------------------------------------
# Contract defect: PointCloudData now matches TIN's own real,
# working behavior (both `elevation_array` and `xy_array` are plain
# methods, not properties).
# ----------------------------------------------------------------------


def _sample_tin() -> TIN:
    vertices = (
        Point3D(0.0, 0.0, 0.0),
        Point3D(10.0, 0.0, 10.0),
        Point3D(0.0, 10.0, 20.0),
    )
    simplices = np.array([[0, 1, 2]], dtype=np.int32)
    return TIN.from_mesh(vertices, simplices)


def test_tin_satisfies_elevation_array_and_xy_array_contract() -> None:
    """
    Confirmed directly: this only became true after fixing
    `PointCloudData` to type `elevation_array`/`xy_array` as methods
    -- before that fix, `mypy` rejected passing a `TIN` anywhere
    either was declared this way (confirmed directly during the
    audit: "TIN is missing following PointCloudData protocol
    member... elevation_array: expected ndarray, got
    Callable[[], ndarray]").

    This does NOT assert full `PointCloudData` conformance -- `TIN`
    still has no `array` member at all (never did; a separate,
    pre-existing, deliberately out-of-scope gap, since `array` has no
    confirmed consumer requiring a decision either way -- see
    `contracts.md`). Calling `ElevationStatistics.from_tin()`/
    `DensityStatistics.compute_from_tin()` with a real `TIN` still
    needs `# type: ignore[arg-type]` for that unrelated reason alone,
    visible in this file's own calls below.
    """
    tin = _sample_tin()
    assert tin.elevation_array().size == 3
    assert tin.xy_array().shape == (3, 2)


# ----------------------------------------------------------------------
# Defect 1a/1b: `tin.elevation_array` was passed without calling it
# (a bound method, not the array) -- TypeError in np.isfinite().
# ----------------------------------------------------------------------


def test_elevation_statistics_from_tin_no_longer_raises() -> None:
    """
    Confirmed directly before this fix: `tin.elevation_array` (no
    call) was passed straight to `np.isfinite()`, raising TypeError
    unconditionally -- `from_tin()` never worked for any TIN.
    """
    tin = _sample_tin()
    # `array` (a separate PointCloudData member TIN never
    # implements, out of scope per this fix's own explicit decision
    # -- see contracts.md) is the only remaining reason this needs
    # an ignore; elevation_array/xy_array are now fully resolved.
    result = ElevationStatistics.from_tin(tin)  # type: ignore[arg-type]

    assert result.minimum == pytest.approx(0.0)
    assert result.maximum == pytest.approx(20.0)
    assert result.mean == pytest.approx(10.0)
    assert result.count == 3


def test_statistics_analysis_elevation_method_no_longer_raises() -> None:
    """
    Same underlying defect, a separate occurrence in
    StatisticsAnalysis's own duck-typed dispatch (statistics/manager.py)
    -- confirmed as its own, separately reproducible failure before
    this fix, not merely a re-test of from_tin() through another
    entry point.
    """
    tin = _sample_tin()
    result = StatisticsAnalysis(method="elevation").compute(tin)
    assert isinstance(result, ElevationStats)

    assert result.minimum == pytest.approx(0.0)
    assert result.maximum == pytest.approx(20.0)
    assert result.mean == pytest.approx(10.0)


def test_elevation_statistics_from_tin_and_manager_agree() -> None:
    """The 2 separately-fixed call paths must produce identical results for the same TIN."""
    tin = _sample_tin()
    direct = ElevationStatistics.from_tin(tin)  # type: ignore[arg-type]
    via_manager = StatisticsAnalysis(method="elevation").compute(tin)
    assert isinstance(via_manager, ElevationStats)

    assert direct == via_manager


# ----------------------------------------------------------------------
# Defect 1c (found only after insisting on a complete, zero-pending
# audit of every elevation_array/xy_array consumer, not just the 2
# already known): DensityStatistics.compute_from_tin() passed
# `tin.xy_array` without calling it -- the same defect family, in a
# method never exercised by the original audit pass.
# ----------------------------------------------------------------------


def test_density_statistics_compute_from_tin_no_longer_raises() -> None:
    """
    Confirmed directly before this fix: `tin.xy_array` (no call) was
    passed straight to `self.compute()`'s own internal coordinate
    conversion, raising TypeError unconditionally -- found only by
    auditing every consumer of the same Protocol members already
    known to be involved in the elevation_array defects above, not by
    this method being suspected on its own.
    """
    tin = _sample_tin()
    result = DensityStatistics(resolution=5.0).compute_from_tin(tin)  # type: ignore[arg-type]

    assert result.mean_density == pytest.approx(0.04)
    assert result.minimum_density == pytest.approx(0.04)
    assert result.maximum_density == pytest.approx(0.04)


# ----------------------------------------------------------------------
# Defect 2: SlopeStatistics.compute_from_array() raised
# "Cannot create N finite-sized bins" for a genuinely uniform slope.
#
# The real defect is narrower than "min == max exactly" -- a uniform
# slope computed via real floating-point arithmetic (np.gradient ->
# np.hypot -> np.arctan -> np.degrees) differs by ~1e-15 across the
# grid, not 0 exactly. np.histogram() only self-widens an EXACTLY
# degenerate range; this tiny nonzero range was what actually
# crashed. Confirmed directly during the audit: this is why merely
# mirroring DistributionStatistics's own (range=None) behavior would
# NOT have fixed this -- that class has never been exercised against
# a real, computed near-uniform array in this codebase, only an
# artificially exact-uniform one via np.full(). This defect is
# independent of the Protocol/TIN contract defect above -- it lives
# entirely inside SlopeStatistics's own histogram construction.
# ----------------------------------------------------------------------


def test_slope_statistics_real_uniform_ramp_no_longer_raises() -> None:
    """
    A perfectly uniform ramp (elevation increases identically along
    every row) -- confirmed directly, this is the exact scenario that
    raised ValueError before this fix, with a real (not artificial)
    ~1e-15 floating-point spread in the computed slope values.
    """
    size = 5
    elevations = np.tile(np.arange(size) * 0.1, (size, 1))

    result = SlopeStatistics(num_bins=3).compute_from_array(elevations, resolution=1.0)

    # minimum/maximum are reported exactly as computed -- the fix
    # stabilizes only the histogram's own internal binning range,
    # never the reported statistics themselves.
    assert result.minimum == pytest.approx(5.7106, abs=1e-3)
    assert result.maximum == pytest.approx(5.7106, abs=1e-3)
    assert result.minimum != result.maximum  # confirms this is genuinely the near-equal, not exact-equal, case
    assert sum(result.histogram_counts) == size * size
    assert max(result.histogram_counts) == size * size  # every value lands in the same bin


def test_slope_statistics_exact_uniform_values_still_works() -> None:
    """
    The exact-equality case (handled by np.histogram() itself even
    before this fix) must keep working identically -- this fix must
    not regress the case that already worked.
    """
    stats = SlopeStatistics(num_bins=3)
    exact_uniform = np.full(20, 5.0)

    result = stats._build_statistics(exact_uniform)

    assert result.minimum == 5.0
    assert result.maximum == 5.0
    assert sum(result.histogram_counts) == 20
    assert max(result.histogram_counts) == 20


def test_slope_statistics_non_uniform_terrain_result_unchanged() -> None:
    """
    Confirmed directly (same seed, same inputs) against this fix's
    own pre-change verification run -- a non-degenerate case must
    produce exactly the same statistics after the fix as before it.
    """
    rng = np.random.default_rng(0)
    size = 5
    elevations = np.cumsum(rng.uniform(0.05, 0.3, (size, size)), axis=1)

    result = SlopeStatistics(num_bins=3).compute_from_array(elevations, resolution=1.0)

    assert result.minimum == pytest.approx(5.0885486187461, abs=1e-9)
    assert result.maximum == pytest.approx(28.784459663606146, abs=1e-9)
    assert result.mean == pytest.approx(13.15305891238565, abs=1e-9)
    # A genuinely spread-out histogram, unlike the degenerate cases above.
    assert len([count for count in result.histogram_counts if count > 0]) > 1


# ----------------------------------------------------------------------
# Defect found during this project's own Point-3 manager/dispatcher
# audit (a later phase of the same PR22 documentation effort, not
# the original 3-defect fix above): `StatisticsAnalysis.slope()`
# only accepted a `GriddedSurface`, despite `SlopeStatistics` (the
# class it wraps) publicly supporting a raw array too -- a narrower
# facade than the class it represents, and inconsistent with this
# same manager's own `elevation()` method, which already branches on
# input type. Passing a raw array used to raise a bare
# `AttributeError` ("'numpy.ndarray' object has no attribute
# 'elevations'"), not a `StatisticsError`.
# ----------------------------------------------------------------------


def test_statistics_analysis_slope_accepts_a_raw_array_with_resolution() -> None:
    """
    Confirmed directly before this fix: this raised AttributeError
    unconditionally for a raw array -- `.slope()` only ever called
    `compute_from_dtm()` internally, regardless of input type.
    """
    from topocore.analysis.statistics import StatisticsAnalysis

    size = 5
    elevations = np.tile(np.arange(size) * 0.1, (size, 1))

    result = StatisticsAnalysis(method="slope").slope(elevations, resolution=1.0)

    assert result.minimum == pytest.approx(5.7106, abs=1e-3)
    assert result.maximum == pytest.approx(5.7106, abs=1e-3)


def test_statistics_analysis_slope_raw_array_without_resolution_raises_statistics_error() -> None:
    """
    A raw array requires `resolution` (unlike a GriddedSurface, which
    carries its own) -- confirmed this is now a clear, domain-specific
    `StatisticsError`, not the bare `AttributeError` from before this
    fix.
    """
    from topocore.analysis.statistics import StatisticsAnalysis

    elevations = np.zeros((5, 5))

    with pytest.raises(StatisticsError, match="resolution is required"):
        StatisticsAnalysis(method="slope").slope(elevations)


def test_statistics_analysis_slope_via_compute_dispatch_matches_direct_call() -> None:
    """The generic `.compute()` dispatch path must agree exactly with calling `.slope()` directly."""
    from topocore.analysis.statistics import StatisticsAnalysis

    size = 5
    elevations = np.tile(np.arange(size) * 0.1, (size, 1))
    analysis = StatisticsAnalysis(method="slope")

    direct = analysis.slope(elevations, resolution=1.0)
    via_compute = analysis.compute(elevations, resolution=1.0, method="slope")

    assert direct == via_compute


def test_statistics_analysis_slope_gridded_surface_path_unchanged() -> None:
    """
    The pre-existing `GriddedSurface` path (e.g. a real DTM) must be
    completely unaffected by adding the raw-array path above --
    verified against a minimal object satisfying `GriddedSurface`
    structurally, matching `SlopeStatistics.compute_from_dtm()`
    called directly on the same object.
    """
    from dataclasses import dataclass

    from topocore.analysis.statistics import StatisticsAnalysis
    from topocore.terrain.grid import Grid

    @dataclass
    class _FakeGriddedSurface:
        grid: Grid
        resolution: float
        elevations: NDArray[np.float64]

    rng = np.random.default_rng(0)
    surface = _FakeGriddedSurface(
        grid=Grid(min_x=0.0, min_y=0.0, max_x=5.0, max_y=5.0, resolution=1.0),
        resolution=1.0,
        elevations=np.cumsum(rng.uniform(0.05, 0.3, (5, 5)), axis=1),
    )

    via_manager = StatisticsAnalysis(method="slope").slope(surface)
    direct = SlopeStatistics(num_bins=10).compute_from_dtm(surface)

    assert via_manager == direct
