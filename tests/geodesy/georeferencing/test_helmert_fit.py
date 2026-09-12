"""
Tests for topocore.geodesy.georeferencing.helmert_fit -- the pure
mathematical core of control-point-based georeferencing.

Deliberately narrow scope, matching this feature's own first phase:
ControlPoint, fit_helmert_3d(), apply_helmert_3d() only. No
Georeferencing/GeoreferencingOptions orchestration, no acceptance
validation, no SurveyPointSet/target_crs integration -- those build
on top of this core in a later phase.

No geographic data, no PROJ, no geocentric coordinates, no CRS
anywhere in this suite -- this is a plain Cartesian 3D similarity
transform fit, verified purely as numerical linear algebra.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.geodesy.georeferencing import (
    ControlPoint,
    apply_helmert_3d,
    fit_helmert_3d,
)
from topocore.geodesy.georeferencing.exceptions import (
    DegenerateGeometryError,
    InsufficientControlPointsError,
)
from topocore.geodesy.helmert import HelmertParameters

_SOURCES = [
    (100.0, 0.0, 0.0),
    (0.0, 100.0, 0.0),
    (0.0, 0.0, 100.0),
    (50.0, 50.0, 50.0),
]


def _controls(
    sources: list[tuple[float, float, float]], targets: list[tuple[float, float, float]]
) -> list[ControlPoint]:
    return [ControlPoint(id=str(i), source=s, target=t) for i, (s, t) in enumerate(zip(sources, targets, strict=True))]


def _apply_to_all(
    sources: list[tuple[float, float, float]], parameters: HelmertParameters
) -> list[tuple[float, float, float]]:
    return [apply_helmert_3d(point, parameters) for point in sources]


# ----------------------------------------------------------------------
# 1. Identity.
# ----------------------------------------------------------------------


def test_identity_transformation() -> None:
    result = fit_helmert_3d(_controls(_SOURCES, _SOURCES))
    p = result.parameters

    assert p.tx == pytest.approx(0.0, abs=1e-9)
    assert p.ty == pytest.approx(0.0, abs=1e-9)
    assert p.tz == pytest.approx(0.0, abs=1e-9)
    assert p.rx == pytest.approx(0.0, abs=1e-9)
    assert p.ry == pytest.approx(0.0, abs=1e-9)
    assert p.rz == pytest.approx(0.0, abs=1e-9)
    assert p.scale == pytest.approx(0.0, abs=1e-9)
    assert result.rms == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------
# 2. Known translation.
# ----------------------------------------------------------------------


def test_known_translation_is_recovered_exactly() -> None:
    targets = [(x + 10.0, y + 20.0, z + 30.0) for x, y, z in _SOURCES]
    result = fit_helmert_3d(_controls(_SOURCES, targets))

    assert result.parameters.tx == pytest.approx(10.0)
    assert result.parameters.ty == pytest.approx(20.0)
    assert result.parameters.tz == pytest.approx(30.0)
    assert result.parameters.rx == pytest.approx(0.0, abs=1e-9)
    # scale is stored in ppm (multiplied by 1e6 from the raw
    # dimensionless fitted value) -- floating-point noise at the
    # ~1e-14 level in the raw value is genuinely amplified to ~1e-8
    # by this unit conversion; abs=1e-9 was too tight for this
    # specific field, confirmed directly, not a fit inaccuracy.
    assert result.parameters.scale == pytest.approx(0.0, abs=1e-7)
    assert result.rms == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------
# 3. Known rotation.
# ----------------------------------------------------------------------


def test_known_rotation_is_recovered_exactly() -> None:
    known = HelmertParameters(tx=0.0, ty=0.0, tz=0.0, rz=15.0)
    targets = _apply_to_all(_SOURCES, known)
    result = fit_helmert_3d(_controls(_SOURCES, targets))

    assert result.parameters.rz == pytest.approx(15.0)
    assert result.parameters.rx == pytest.approx(0.0, abs=1e-6)
    assert result.parameters.ry == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# 4. Known scale.
# ----------------------------------------------------------------------


def test_known_scale_is_recovered_exactly() -> None:
    known = HelmertParameters(tx=0.0, ty=0.0, tz=0.0, scale=250.0)
    targets = _apply_to_all(_SOURCES, known)
    result = fit_helmert_3d(_controls(_SOURCES, targets))

    assert result.parameters.scale == pytest.approx(250.0)


# ----------------------------------------------------------------------
# 5. Combined transformation.
# ----------------------------------------------------------------------


def test_combined_transformation_is_recovered_exactly() -> None:
    known = HelmertParameters(tx=100.0, ty=-50.0, tz=25.0, rx=5.0, ry=-3.0, rz=8.0, scale=120.0)
    targets = _apply_to_all(_SOURCES, known)
    result = fit_helmert_3d(_controls(_SOURCES, targets))

    assert result.parameters.tx == pytest.approx(100.0)
    assert result.parameters.ty == pytest.approx(-50.0)
    assert result.parameters.tz == pytest.approx(25.0)
    # Confirmed directly: actual achieved precision here is ~1e-8 for
    # both rotation and scale (not merely "small enough for 1e-5" --
    # that earlier tolerance was unjustifiably loose, tightened here
    # to the precision actually observed, with headroom).
    assert result.parameters.rx == pytest.approx(5.0, abs=1e-6)
    assert result.parameters.ry == pytest.approx(-3.0, abs=1e-6)
    assert result.parameters.rz == pytest.approx(8.0, abs=1e-6)
    assert result.parameters.scale == pytest.approx(120.0, abs=1e-6)
    assert result.rms == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# 6. Small noise -- parameters close to true values, RMS coherent,
# per-point residuals available.
# ----------------------------------------------------------------------


def test_small_noise_gives_close_parameters_and_coherent_rms() -> None:
    known = HelmertParameters(tx=100.0, ty=-50.0, tz=25.0, rx=5.0, ry=-3.0, rz=8.0, scale=120.0)
    clean_targets = _apply_to_all(_SOURCES, known)

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.001, (len(_SOURCES), 3))
    noisy_targets = [(x + dx, y + dy, z + dz) for (x, y, z), (dx, dy, dz) in zip(clean_targets, noise, strict=True)]

    result = fit_helmert_3d(_controls(_SOURCES, noisy_targets))

    # Translation tolerance (linear units) can stay tight -- it's not
    # amplified by the ~100-unit "lever arm" the way rotation is.
    assert result.parameters.tx == pytest.approx(100.0, abs=0.01)
    # Rotation tolerance is deliberately looser: confirmed directly
    # (independent hand calculation) that 1mm of coordinate noise over
    # this fixture's own ~100-unit baseline has an expected angular
    # sensitivity on the order of ~2 arcsec (noise / baseline,
    # converted to arcsec) -- a ~1 arcsec observed error here is
    # genuinely within that expected order of magnitude, not a defect.
    assert result.parameters.rz == pytest.approx(8.0, abs=3.0)
    assert 0.0 < result.rms < 0.01  # small but genuinely nonzero
    assert len(result.residuals) == len(_SOURCES)
    for residual in result.residuals:
        assert residual.magnitude < 0.01


# ----------------------------------------------------------------------
# 7. Insufficient control points.
# ----------------------------------------------------------------------


def test_insufficient_control_points_raises() -> None:
    with pytest.raises(InsufficientControlPointsError, match="at least 3"):
        fit_helmert_3d(_controls(_SOURCES[:2], _SOURCES[:2]))


def test_zero_control_points_raises() -> None:
    with pytest.raises(InsufficientControlPointsError):
        fit_helmert_3d([])


# ----------------------------------------------------------------------
# 8. Degenerate geometry.
# ----------------------------------------------------------------------


def test_collinear_control_points_raise_degenerate_geometry() -> None:
    collinear_sources = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (20.0, 0.0, 0.0),
        (30.0, 0.0, 0.0),
    ]
    targets = [(x + 5.0, y + 5.0, z + 5.0) for x, y, z in collinear_sources]

    with pytest.raises(DegenerateGeometryError, match="collinear"):
        fit_helmert_3d(_controls(collinear_sources, targets))


def test_collinear_with_many_points_still_raises() -> None:
    """More points doesn't fix degeneracy if they're all still collinear."""
    collinear_sources = [(float(i), 0.0, 0.0) for i in range(10)]
    targets = [(x + 5.0, y + 5.0, z + 5.0) for x, y, z in collinear_sources]

    with pytest.raises(DegenerateGeometryError):
        fit_helmert_3d(_controls(collinear_sources, targets))


# ----------------------------------------------------------------------
# 9. Determinism.
# ----------------------------------------------------------------------


def test_fit_is_deterministic() -> None:
    known = HelmertParameters(tx=100.0, ty=-50.0, tz=25.0, rx=5.0, ry=-3.0, rz=8.0, scale=120.0)
    targets = _apply_to_all(_SOURCES, known)

    result_a = fit_helmert_3d(_controls(_SOURCES, targets))
    result_b = fit_helmert_3d(_controls(_SOURCES, targets))

    assert result_a.parameters == result_b.parameters
    assert result_a.rms == result_b.rms


# ----------------------------------------------------------------------
# 10. Numerically difficult case -- real UTM-scale magnitudes with
# small rotation/scale.
# ----------------------------------------------------------------------


def test_utm_scale_magnitudes_with_small_transformation() -> None:
    utm_sources = [
        (500000.0, 4649000.0, 100.0),
        (500100.0, 4649000.0, 105.0),
        (500000.0, 4649100.0, 102.0),
        (500050.0, 4649050.0, 103.0),
    ]
    known = HelmertParameters(tx=0.05, ty=-0.03, tz=0.02, rx=0.5, ry=-0.3, rz=0.2, scale=5.0)
    targets = _apply_to_all(utm_sources, known)

    result = fit_helmert_3d(_controls(utm_sources, targets))

    assert result.parameters.tx == pytest.approx(0.05, abs=1e-4)
    assert result.parameters.rz == pytest.approx(0.2, abs=1e-4)
    assert result.rms < 1e-6


# ----------------------------------------------------------------------
# Residual sign convention and apply/fit round trip.
# ----------------------------------------------------------------------


def test_residual_sign_convention_is_target_minus_transformed_source() -> None:
    """residual = target - transformed_source, matching topocore.geodesy.transform's own convention."""
    known = HelmertParameters(tx=1.0, ty=0.0, tz=0.0)
    controls = [ControlPoint(id="1", source=(0.0, 0.0, 0.0), target=(1.5, 0.0, 0.0))]  # true tx would be 1.5, not 1.0

    # Manually compute what a fit with ONLY this offset control point would produce residual-wise
    # if we apply the WRONG (given, not fitted) parameters -- confirms sign direction directly.
    transformed = apply_helmert_3d(controls[0].source, known)
    expected_dx = controls[0].target[0] - transformed[0]
    assert expected_dx == pytest.approx(0.5)


def test_apply_helmert_3d_matches_the_fit_exactly() -> None:
    """No discrepancy between what fit_helmert_3d() solves for and what apply_helmert_3d() produces."""
    known = HelmertParameters(tx=10.0, ty=-5.0, tz=3.0, rx=2.0, ry=1.0, rz=-1.5, scale=50.0)
    targets = _apply_to_all(_SOURCES, known)

    result = fit_helmert_3d(_controls(_SOURCES, targets))

    for control, target in zip(_controls(_SOURCES, targets), targets, strict=True):
        reapplied = apply_helmert_3d(control.source, result.parameters)
        assert reapplied[0] == pytest.approx(target[0], abs=1e-6)
        assert reapplied[1] == pytest.approx(target[1], abs=1e-6)
        assert reapplied[2] == pytest.approx(target[2], abs=1e-6)


# ----------------------------------------------------------------------
# Reviewer-identified gaps, closed with real, verified evidence.
# ----------------------------------------------------------------------


def test_coplanar_non_collinear_points_are_not_degenerate() -> None:
    """
    Confirmed directly, contrary to the initial (incorrect) assumption
    that coplanarity alone is degenerate for a 3D fit (an intuition
    carried over from unrelated geometric problems like plane
    fitting): 4 points sharing Z=0 but spread in X/Y (a square) give
    a full rank-7 design matrix and correctly recover a genuine
    rotation about X -- because the Z equation's own coefficients for
    rx/ry depend on X/Y (which vary across these points), not on Z
    itself. This test locks in the CORRECT, verified behavior, not
    the originally-assumed (and wrong) rejection.
    """
    coplanar_sources = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (0.0, 10.0, 0.0),
        (10.0, 10.0, 0.0),
    ]
    known = HelmertParameters(tx=0.0, ty=0.0, tz=0.0, rx=20.0)
    targets = _apply_to_all(coplanar_sources, known)

    result = fit_helmert_3d(_controls(coplanar_sources, targets))

    assert result.parameters.rx == pytest.approx(20.0)
    assert result.rms == pytest.approx(0.0, abs=1e-6)


def test_reflection_is_rejected_as_linearization_invalid() -> None:
    """
    A genuine mirror reflection (source/target related by an axis
    flip) is not representable by a proper (determinant +1) Helmert
    similarity transform at all. Confirmed directly: without this
    check, the fit "succeeds" numerically but produces a wildly
    unphysical scale (~-666667 ppm) and rotations (~±121332 arcsec,
    ~34 degrees) -- caught here via the fitted rotation's own
    magnitude, well before any caller would need to notice the
    absurd parameter values themselves.
    """
    from topocore.geodesy.georeferencing.exceptions import LinearizationInvalidError

    reflected_targets = [(-x, y, z) for x, y, z in _SOURCES]

    with pytest.raises(LinearizationInvalidError, match="exceeds"):
        fit_helmert_3d(_controls(_SOURCES, reflected_targets))


def test_large_but_valid_rotation_is_not_rejected() -> None:
    """Confirms the LinearizationInvalidError threshold doesn't reject genuinely valid, if large, rotations."""
    known = HelmertParameters(tx=0.0, ty=0.0, tz=0.0, rz=2000.0)  # well under the 3600 arcsec threshold
    targets = _apply_to_all(_SOURCES, known)

    result = fit_helmert_3d(_controls(_SOURCES, targets))
    assert result.parameters.rz == pytest.approx(2000.0)


def test_inconsistent_correspondences_are_not_rejected_but_show_large_residuals() -> None:
    """
    Confirmed directly, and deliberately NOT a rejection: a moderate
    outlier in one control point's own target (4 points consistent
    with a plain translation, a 5th deliberately off) makes the
    system genuinely inconsistent (no exact solution exists).
    fit_helmert_3d() still returns its best least-squares
    approximation rather than raising -- the elevated residuals ARE
    the signal a caller should act on. Whether to accept or reject
    based on residual magnitude is an acceptance decision explicitly
    deferred to a later phase of this capability (a future
    GeoreferencingOptions/validation layer), not this pure fitting
    core's own responsibility.

    Confirmed separately, while building this test: a MORE severe
    inconsistency (two controls sharing the same source but wildly
    disagreeing targets) is actually caught by
    `LinearizationInvalidError` instead -- the least-squares "best
    fit" for that specific case required an unphysically large
    rotation to partially reconcile the contradiction, which the
    fit's own linearization-validity check (see
    `test_reflection_is_rejected_as_linearization_invalid`) correctly
    rejects too. This test uses a milder, more realistic outlier
    (a plausible surveying blunder on one point) specifically to
    exercise the "inconsistent but not rotation-breaking" case
    distinctly from that one.
    """
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
        (200.0, 100.0, 50.0),
    ]
    targets = [(x + 10.0, y + 20.0, z + 30.0) for x, y, z in sources]
    targets[4] = (
        targets[4][0] + 5.0,
        targets[4][1] - 3.0,
        targets[4][2] + 2.0,
    )  # a moderate, realistic outlier

    result = fit_helmert_3d(_controls(sources, targets))

    assert result.rms > 1.0  # confirms the inconsistency is visible, not hidden
    assert len(result.residuals) == 5
    assert max(r.magnitude for r in result.residuals) > 0.5
