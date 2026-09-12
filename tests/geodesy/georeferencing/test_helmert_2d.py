"""
Tests for topocore.geodesy.georeferencing.helmert_2d.fit_helmert_2d().

Special attention to the exact scope this module's own docstring
promises: rx/ry are always 0.0 (never estimated), Z = Z + tz only
(never scaled/rotated), and degeneracy/reflection detection are each
this module's own independent concern -- not derived from
fit_helmert_3d()'s own system.
"""

from __future__ import annotations

import pytest
from topocore.geodesy.georeferencing import ControlPoint, apply_helmert_3d
from topocore.geodesy.georeferencing.exceptions import (
    DegenerateGeometryError,
    InsufficientControlPointsError,
    LinearizationInvalidError,
)
from topocore.geodesy.georeferencing.helmert_2d import fit_helmert_2d
from topocore.geodesy.helmert import HelmertParameters

_SOURCES = [(100.0, 0.0, 0.0), (0.0, 100.0, 50.0), (50.0, 50.0, 25.0)]


def _controls_from_2d_transform(
    sources: list[tuple[float, float, float]],
    tx: float,
    ty: float,
    tz: float,
    rz: float,
    scale: float,
) -> list[ControlPoint]:
    """
    Builds targets using ONLY the 2D formula (X/Y rotated+scaled+
    translated, Z translated by tz alone, never scaled) -- distinct
    from using apply_helmert_3d() directly with rx=ry=0, which WOULD
    also apply `scale` to Z (confirmed directly, during this
    capability's own construction, that this creates a small but
    real mismatch against what fit_helmert_2d() itself assumes about
    Z). This constructs test data that genuinely matches this
    module's own model exactly, for exact-recovery assertions.
    """
    import math

    rz_rad = math.radians(rz / 3600.0)
    s = scale * 1e-6
    controls = []
    for i, (xs, ys, zs) in enumerate(sources):
        xt = xs + tx + s * xs - rz_rad * ys
        yt = ys + ty + s * ys + rz_rad * xs
        zt = zs + tz
        controls.append(ControlPoint(id=str(i), source=(xs, ys, zs), target=(xt, yt, zt)))
    return controls


def test_known_2d_transformation_is_recovered_exactly() -> None:
    controls = _controls_from_2d_transform(_SOURCES, tx=10.0, ty=20.0, tz=5.0, rz=15.0, scale=100.0)
    result = fit_helmert_2d(controls)

    assert result.parameters.tx == pytest.approx(10.0)
    assert result.parameters.ty == pytest.approx(20.0)
    assert result.parameters.tz == pytest.approx(5.0)
    assert result.parameters.rz == pytest.approx(15.0)
    assert result.parameters.scale == pytest.approx(100.0)
    assert result.rms == pytest.approx(0.0, abs=1e-6)


def test_rx_and_ry_are_always_zero_never_estimated() -> None:
    controls = _controls_from_2d_transform(_SOURCES, tx=1.0, ty=2.0, tz=3.0, rz=5.0, scale=10.0)
    result = fit_helmert_2d(controls)

    assert result.parameters.rx == 0.0
    assert result.parameters.ry == 0.0


def test_z_is_translated_only_never_scaled_or_rotated() -> None:
    """
    The core scope assertion: Z' = Z + tz exactly, regardless of how
    large rz/scale are -- confirmed directly against a hand-computed
    tz, not merely "close to" some value influenced by scale.
    """
    controls = _controls_from_2d_transform(_SOURCES, tx=0.0, ty=0.0, tz=7.5, rz=500.0, scale=2000.0)
    result = fit_helmert_2d(controls)

    assert result.parameters.tz == pytest.approx(7.5, abs=1e-9)

    # Confirm every single source Z, once shifted by tz alone, matches its own target Z exactly.
    for control in controls:
        assert control.source[2] + result.parameters.tz == pytest.approx(control.target[2])


def test_reapplying_fitted_parameters_matches_xy_exactly_but_not_z_when_scale_nonzero() -> None:
    """
    Confirmed a genuine, permanent structural characteristic, not a
    test artifact: `apply_helmert_3d()` -- the shared, general 3D
    application function -- always applies `scale` to Z too (`zt =
    zs + tz + s*zs`, even with `rx=ry=0`), while this fit's own model
    assumes `Z' = Z + tz` alone (see this module's own top-level
    docstring). Reapplying a HELMERT_2D-fitted `HelmertParameters` via
    `apply_helmert_3d()` therefore reproduces X/Y exactly, but Z only
    approximately whenever `scale != 0` -- this is an inherent
    consequence of representing a HELMERT_2D result with the same
    shared `HelmertParameters` type the full 3D fit uses, not
    something to "fix" here. How `Workflow.georeference()` should
    apply a HELMERT_2D result to real data later is a separate,
    forward-looking design question for that phase, not this one.
    """
    controls = _controls_from_2d_transform(_SOURCES, tx=10.0, ty=-5.0, tz=3.0, rz=8.0, scale=50.0)
    result = fit_helmert_2d(controls)

    mismatches = 0
    for control in controls:
        reapplied = apply_helmert_3d(control.source, result.parameters)
        assert reapplied[0] == pytest.approx(control.target[0], abs=1e-6)
        assert reapplied[1] == pytest.approx(control.target[1], abs=1e-6)
        # Z matches exactly only when source.z == 0 (since the
        # spurious s*zs term is then itself zero) -- confirmed
        # directly per-point, not asserted as a blanket mismatch for
        # every point in _SOURCES (one of which does have z=0.0).
        if control.source[2] == 0.0:
            assert reapplied[2] == pytest.approx(control.target[2], abs=1e-6)
        else:
            assert reapplied[2] != pytest.approx(control.target[2], abs=1e-6)
            mismatches += 1

    assert mismatches > 0  # confirms the discrepancy was genuinely exercised, not accidentally absent


def test_points_with_distinct_xy_but_shared_z_still_work() -> None:
    """Distinct XY is all that matters for this fit -- shared Z across all points is irrelevant to it."""
    sources = [(0.0, 0.0, 42.0), (10.0, 0.0, 42.0), (0.0, 10.0, 42.0)]
    controls = _controls_from_2d_transform(sources, tx=5.0, ty=5.0, tz=1.0, rz=10.0, scale=100.0)
    result = fit_helmert_2d(controls)

    assert result.parameters.tx == pytest.approx(5.0)
    assert result.parameters.rz == pytest.approx(10.0)


def test_vertical_line_same_xy_raises_degenerate_geometry() -> None:
    """Confirmed directly earlier: points sharing the same XY projection give rank_2d=2, regardless of point count or Z spread."""
    vertical_sources = [(5.0, 5.0, 0.0), (5.0, 5.0, 10.0), (5.0, 5.0, 20.0)]
    controls = [
        ControlPoint(id=str(i), source=s, target=(s[0] + 1, s[1] + 1, s[2] + 1)) for i, s in enumerate(vertical_sources)
    ]

    with pytest.raises(DegenerateGeometryError, match="XY position"):
        fit_helmert_2d(controls)


def test_reflection_raises_linearization_invalid() -> None:
    sources = [(100.0, 0.0, 0.0), (0.0, 100.0, 0.0)]
    reflected_targets = [(-x, y, z) for x, y, z in sources]
    controls = [
        ControlPoint(id=str(i), source=s, target=t)
        for i, (s, t) in enumerate(zip(sources, reflected_targets, strict=True))
    ]

    with pytest.raises(LinearizationInvalidError, match="exceeds"):
        fit_helmert_2d(controls)


def test_insufficient_control_points_raises() -> None:
    with pytest.raises(InsufficientControlPointsError, match="at least 2"):
        fit_helmert_2d([ControlPoint(id="1", source=(0.0, 0.0, 0.0), target=(1.0, 1.0, 1.0))])


def test_zero_control_points_raises() -> None:
    with pytest.raises(InsufficientControlPointsError):
        fit_helmert_2d([])


def test_two_control_points_with_distinct_xy_is_sufficient() -> None:
    """The minimum case: exactly 2 points, distinct in XY, is enough (4 equations for 4 unknowns)."""
    sources = [(0.0, 0.0, 0.0), (10.0, 5.0, 3.0)]
    controls = _controls_from_2d_transform(sources, tx=1.0, ty=1.0, tz=1.0, rz=5.0, scale=10.0)
    result = fit_helmert_2d(controls)

    assert result.parameters.tx == pytest.approx(1.0)
    assert result.rms == pytest.approx(0.0, abs=1e-6)


def test_apply_helmert_3d_with_full_3d_parameters_confirms_expected_z_mismatch() -> None:
    """
    Documents a real, understood distinction discovered while
    building this module: generating test targets via
    apply_helmert_3d() directly (a genuinely 3D transform, which DOES
    apply scale to Z) against fit_helmert_2d()'s own Z=Z+tz-only
    model produces a small, expected mismatch in the fitted tz --
    NOT a bug in fit_helmert_2d(), simply a model mismatch when the
    "known" transformation used to generate test data isn't itself
    a genuine 2D one.
    """
    full_3d_params = HelmertParameters(tx=10.0, ty=20.0, tz=5.0, rz=15.0, scale=100.0)
    targets = [apply_helmert_3d(point, full_3d_params) for point in _SOURCES]
    controls = [
        ControlPoint(id=str(i), source=s, target=t) for i, (s, t) in enumerate(zip(_SOURCES, targets, strict=True))
    ]

    result = fit_helmert_2d(controls)

    # tz is NOT exactly 5.0 here -- the "known" data included scale
    # applied to Z (via apply_helmert_3d's own s*zs term), which this
    # fit's own Z=Z+tz-only model cannot capture.
    assert result.parameters.tz != pytest.approx(5.0, abs=1e-6)
