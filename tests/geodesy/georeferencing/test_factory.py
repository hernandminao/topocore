"""
Tests for topocore.geodesy.georeferencing.factory.fit_georeferencing --
the single orchestrator selecting among HELMERT_3D/HELMERT_2D/
TRANSLATION_ONLY based purely on each fit's own mathematical
determinacy (never on len(controls) alone), gated by explicit
caller authorization (GeoreferencingOptions.accept_underconstrained)
for any under-determined strategy.

Every test checks WHICH strategy was actually selected
(result.strategy), not just that the numeric output looks
plausible -- this is the entire point of this factory's own safety
design: it must never silently produce a plausible-looking but
mathematically under-determined result without the caller's explicit
authorization.
"""

from __future__ import annotations

import pytest
from topocore.geodesy.georeferencing import (
    ControlPoint,
    GeoreferencingOptions,
    GeoreferencingStrategy,
    apply_helmert_3d,
    fit_georeferencing,
)
from topocore.geodesy.georeferencing.exceptions import (
    ControlPointsTooCloseError,
    InsufficientControlPointsError,
    UnderconstrainedGeoreferencingError,
)
from topocore.geodesy.helmert import HelmertParameters

_KNOWN_3D = HelmertParameters(tx=10.0, ty=20.0, tz=5.0, rx=2.0, ry=-1.0, rz=3.0, scale=50.0)


def _controls_from(sources: list[tuple[float, float, float]], parameters: HelmertParameters) -> list[ControlPoint]:
    targets = [apply_helmert_3d(point, parameters) for point in sources]
    return [ControlPoint(id=str(i), source=s, target=t) for i, (s, t) in enumerate(zip(sources, targets, strict=True))]


# ----------------------------------------------------------------------
# A. 3D determined (rank_3d == 7) -- no authorization needed.
# ----------------------------------------------------------------------


def test_a_well_distributed_points_select_helmert_3d_without_authorization() -> None:
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    controls = _controls_from(sources, _KNOWN_3D)

    result = fit_georeferencing(controls, GeoreferencingOptions())  # accept_underconstrained defaults to False

    assert result.strategy is GeoreferencingStrategy.HELMERT_3D
    assert result.warning is None
    assert result.rms is not None
    assert result.control_count == 4
    assert result.parameters.tx == pytest.approx(10.0)
    assert result.parameters.rx == pytest.approx(2.0, abs=1e-5)


# ----------------------------------------------------------------------
# B. 3D degenerate but 2D determined (rank_3d < 7, rank_2d == 4).
# ----------------------------------------------------------------------


def test_b_horizontally_collinear_points_fall_back_to_helmert_2d() -> None:
    """Confirmed directly earlier: collinear points along a horizontal line give rank_3d=6 but rank_2d=4."""
    sources = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (30.0, 0.0, 0.0)]
    known_2d = HelmertParameters(tx=5.0, ty=3.0, tz=2.0, rz=10.0, scale=20.0)
    controls = _controls_from(sources, known_2d)

    with pytest.raises(UnderconstrainedGeoreferencingError):
        fit_georeferencing(controls, GeoreferencingOptions())

    result = fit_georeferencing(controls, GeoreferencingOptions(accept_underconstrained=True))

    assert result.strategy is GeoreferencingStrategy.HELMERT_2D
    assert result.warning is not None
    assert "3D" in result.warning
    assert result.control_count == 4


# ----------------------------------------------------------------------
# C. Both degenerate (rank_3d < 7, rank_2d < 4) -- the critical
# counter-example that motivated verifying this cascade in the first
# place.
# ----------------------------------------------------------------------


def test_c_vertical_line_falls_all_the_way_back_to_translation_only() -> None:
    """Confirmed directly earlier: collinear points along a purely vertical line give rank_3d=6 AND rank_2d=2."""
    sources = [(5.0, 5.0, 0.0), (5.0, 5.0, 10.0), (5.0, 5.0, 20.0)]
    known_translation = HelmertParameters(tx=1.0, ty=2.0, tz=3.0)
    controls = _controls_from(sources, known_translation)

    with pytest.raises(UnderconstrainedGeoreferencingError):
        fit_georeferencing(controls, GeoreferencingOptions())

    result = fit_georeferencing(controls, GeoreferencingOptions(accept_underconstrained=True))

    assert result.strategy is GeoreferencingStrategy.TRANSLATION_ONLY
    assert result.warning is not None
    assert "2D" in result.warning or "3D" in result.warning
    assert result.control_count == 3
    assert result.rms is not None  # n=3 > 1, so genuine redundancy exists here


# ----------------------------------------------------------------------
# D. Single control point.
# ----------------------------------------------------------------------


def test_d_single_control_point_requires_authorization_and_gives_none_rms() -> None:
    control = ControlPoint(id="1", source=(0.0, 0.0, 0.0), target=(5.0, 3.0, 2.0))

    with pytest.raises(UnderconstrainedGeoreferencingError):
        fit_georeferencing([control], GeoreferencingOptions())

    result = fit_georeferencing([control], GeoreferencingOptions(accept_underconstrained=True))

    assert result.strategy is GeoreferencingStrategy.TRANSLATION_ONLY
    assert result.rms is None  # None specifically in the public GeoreferencingResult, per this module's own contract
    assert result.control_count == 1
    assert result.warning is not None
    assert result.parameters.tx == pytest.approx(5.0)


# ----------------------------------------------------------------------
# E. Zero control points.
# ----------------------------------------------------------------------


def test_e_zero_control_points_raises_regardless_of_options() -> None:
    with pytest.raises(InsufficientControlPointsError):
        fit_georeferencing([], GeoreferencingOptions())

    with pytest.raises(InsufficientControlPointsError):
        fit_georeferencing([], GeoreferencingOptions(accept_underconstrained=True))


# ----------------------------------------------------------------------
# accept_underconstrained: rejects vs authorizes, never silently
# degrades.
# ----------------------------------------------------------------------


def test_accept_underconstrained_false_rejects_2_points() -> None:
    sources = [(100.0, 0.0, 0.0), (0.0, 100.0, 50.0)]
    controls = _controls_from(sources, HelmertParameters(tx=1.0, ty=2.0, tz=3.0, rz=10.0, scale=20.0))

    with pytest.raises(UnderconstrainedGeoreferencingError, match="HELMERT_2D"):
        fit_georeferencing(controls, GeoreferencingOptions(accept_underconstrained=False))


def test_accept_underconstrained_true_authorizes_2_points_with_warning() -> None:
    sources = [(100.0, 0.0, 0.0), (0.0, 100.0, 50.0)]
    controls = _controls_from(sources, HelmertParameters(tx=1.0, ty=2.0, tz=3.0, rz=10.0, scale=20.0))

    result = fit_georeferencing(controls, GeoreferencingOptions(accept_underconstrained=True))

    assert result.strategy is GeoreferencingStrategy.HELMERT_2D
    assert result.warning is not None


def test_helmert_3d_never_requires_authorization_even_when_options_default() -> None:
    """A fully-determined 3D fit runs regardless of accept_underconstrained -- it never needed authorization at all."""
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    controls = _controls_from(sources, _KNOWN_3D)

    result_default = fit_georeferencing(controls, GeoreferencingOptions())
    result_authorized = fit_georeferencing(controls, GeoreferencingOptions(accept_underconstrained=True))

    assert result_default.strategy is GeoreferencingStrategy.HELMERT_3D
    assert result_authorized.strategy is GeoreferencingStrategy.HELMERT_3D
    assert result_default.warning is None
    assert result_authorized.warning is None


# ----------------------------------------------------------------------
# rms=None is specific to TRANSLATION_ONLY -- confirmed not to leak
# into the other 2 strategies.
# ----------------------------------------------------------------------


def test_rms_is_never_none_for_helmert_3d() -> None:
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    controls = _controls_from(sources, _KNOWN_3D)
    result = fit_georeferencing(controls, GeoreferencingOptions())
    assert result.rms is not None


def test_rms_is_never_none_for_helmert_2d() -> None:
    sources = [(100.0, 0.0, 0.0), (0.0, 100.0, 50.0)]
    controls = _controls_from(sources, HelmertParameters(tx=1.0, ty=2.0, tz=3.0, rz=10.0, scale=20.0))
    result = fit_georeferencing(controls, GeoreferencingOptions(accept_underconstrained=True))
    assert result.rms is not None


def test_rms_is_none_only_for_single_point_translation_only() -> None:
    """Confirms translation_only from MORE than 1 point gets a real rms, not None -- only n=1 gets None."""
    sources = [(5.0, 5.0, 0.0), (5.0, 5.0, 10.0), (5.0, 5.0, 20.0)]
    controls = _controls_from(sources, HelmertParameters(tx=1.0, ty=2.0, tz=3.0))
    result = fit_georeferencing(controls, GeoreferencingOptions(accept_underconstrained=True))
    assert result.strategy is GeoreferencingStrategy.TRANSLATION_ONLY
    assert result.rms is not None  # n=3, not n=1


# ----------------------------------------------------------------------
# control_count and returned parameters usable with apply_helmert_3d().
# ----------------------------------------------------------------------


def test_control_count_matches_exactly_what_was_given() -> None:
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
        (10.0, 20.0, 30.0),
    ]
    controls = _controls_from(sources, _KNOWN_3D)
    result = fit_georeferencing(controls, GeoreferencingOptions())
    assert result.control_count == 5
    assert len(result.residuals) == 5


def test_parameters_are_directly_usable_with_apply_helmert_3d() -> None:
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    controls = _controls_from(sources, _KNOWN_3D)
    result = fit_georeferencing(controls, GeoreferencingOptions())

    # Just confirms this doesn't raise and returns a usable 3-tuple -- the numeric
    # correctness of apply_helmert_3d() itself is already covered by test_helmert_fit.py.
    reapplied = apply_helmert_3d(controls[0].source, result.parameters)
    assert len(reapplied) == 3


# ----------------------------------------------------------------------
# LinearizationInvalidError propagates directly, never triggers fallback.
# ----------------------------------------------------------------------


def test_reflection_propagates_without_falling_back_to_2d_or_translation() -> None:
    """
    A reflection is caught by fit_helmert_3d() itself and must
    propagate directly -- confirmed the factory does NOT catch this
    and attempt HELMERT_2D or TRANSLATION_ONLY instead, since a lower-
    dimensional fit would not fix a reflection, only mask it with a
    plausible-looking but meaningless result.
    """
    from topocore.geodesy.georeferencing.exceptions import LinearizationInvalidError

    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    reflected_targets = [(-x, y, z) for x, y, z in sources]
    controls = [
        ControlPoint(id=str(i), source=s, target=t)
        for i, (s, t) in enumerate(zip(sources, reflected_targets, strict=True))
    ]

    with pytest.raises(LinearizationInvalidError):
        fit_georeferencing(controls, GeoreferencingOptions(accept_underconstrained=True))


# ----------------------------------------------------------------------
# minimum_control_distance.
# ----------------------------------------------------------------------


def test_minimum_control_distance_none_performs_no_check() -> None:
    """None means no check at all -- even control points 1 unit apart in XY are accepted."""
    sources = [(0.0, 0.0, 0.0), (1.0, 0.0, 50.0)]
    controls = _controls_from(sources, HelmertParameters(tx=1.0, ty=2.0, tz=3.0, rz=5.0, scale=10.0))

    result = fit_georeferencing(
        controls,
        GeoreferencingOptions(accept_underconstrained=True, minimum_control_distance=None),
    )
    assert result.strategy is GeoreferencingStrategy.HELMERT_2D


def test_minimum_control_distance_positive_rejects_close_points() -> None:
    sources = [(0.0, 0.0, 0.0), (1.0, 0.0, 50.0)]  # 1.0 apart in XY
    controls = _controls_from(sources, HelmertParameters(tx=1.0, ty=2.0, tz=3.0, rz=5.0, scale=10.0))

    with pytest.raises(ControlPointsTooCloseError):
        fit_georeferencing(
            controls,
            GeoreferencingOptions(accept_underconstrained=True, minimum_control_distance=5.0),
        )


def test_minimum_control_distance_uses_xy_only_not_3d() -> None:
    """
    Two points with identical XY but very different Z (50 units apart
    in Z alone) must still be rejected by even a small
    minimum_control_distance -- confirms XY-only measurement, not 3D
    euclidean distance (which would be dominated by the large Z
    separation and could wrongly appear to satisfy the threshold).
    """
    sources = [(5.0, 5.0, 0.0), (5.0, 5.0, 50.0)]
    controls = _controls_from(sources, HelmertParameters(tx=1.0, ty=2.0, tz=3.0))

    with pytest.raises(ControlPointsTooCloseError):
        fit_georeferencing(
            controls,
            GeoreferencingOptions(accept_underconstrained=True, minimum_control_distance=1.0),
        )


def test_minimum_control_distance_satisfied_by_sufficiently_separated_points() -> None:
    sources = [(0.0, 0.0, 0.0), (10.0, 0.0, 50.0)]  # 10.0 apart in XY
    controls = _controls_from(sources, HelmertParameters(tx=1.0, ty=2.0, tz=3.0, rz=5.0, scale=10.0))

    result = fit_georeferencing(
        controls,
        GeoreferencingOptions(accept_underconstrained=True, minimum_control_distance=5.0),
    )
    assert result.strategy is GeoreferencingStrategy.HELMERT_2D


# ----------------------------------------------------------------------
# Determinism.
# ----------------------------------------------------------------------


def test_fit_georeferencing_is_deterministic() -> None:
    sources = [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (50.0, 50.0, 50.0),
    ]
    controls = _controls_from(sources, _KNOWN_3D)

    result_a = fit_georeferencing(controls, GeoreferencingOptions())
    result_b = fit_georeferencing(controls, GeoreferencingOptions())

    assert result_a.strategy == result_b.strategy
    assert result_a.parameters == result_b.parameters
    assert result_a.rms == result_b.rms


# ----------------------------------------------------------------------
# Reviewer-identified gap, closed with real, verified evidence.
# ----------------------------------------------------------------------


def test_two_vertical_points_fall_back_to_translation_with_precise_warning() -> None:
    """
    Confirmed directly: exactly 2 control points sharing the same XY
    (a vertical pair) never even attempt HELMERT_3D (which requires
    n >= 3) -- the warning must say so precisely, not claim "neither
    3D nor 2D worked" when 3D was genuinely never tried at all.
    """
    controls = [
        ControlPoint(id="1", source=(5.0, 5.0, 0.0), target=(6.0, 7.0, 3.0)),
        ControlPoint(id="2", source=(5.0, 5.0, 10.0), target=(6.0, 7.0, 13.0)),
    ]

    result = fit_georeferencing(controls, GeoreferencingOptions(accept_underconstrained=True))

    assert result.strategy is GeoreferencingStrategy.TRANSLATION_ONLY
    assert result.rms is not None  # n=2 > 1, genuine redundancy exists
    assert result.warning is not None
    assert "never attempted" in result.warning
    assert "either a 3D or a 2D" not in result.warning  # that phrasing belongs only to the n >= 3 case


def test_three_or_more_vertical_points_mention_both_3d_and_2d_attempts() -> None:
    """Contrast case: with n >= 3, both HELMERT_3D and HELMERT_2D genuinely were attempted and failed."""
    controls = [
        ControlPoint(id="1", source=(5.0, 5.0, 0.0), target=(6.0, 7.0, 3.0)),
        ControlPoint(id="2", source=(5.0, 5.0, 10.0), target=(6.0, 7.0, 13.0)),
        ControlPoint(id="3", source=(5.0, 5.0, 20.0), target=(6.0, 7.0, 23.0)),
    ]

    result = fit_georeferencing(controls, GeoreferencingOptions(accept_underconstrained=True))

    assert result.strategy is GeoreferencingStrategy.TRANSLATION_ONLY
    assert result.warning is not None
    assert "either a 3D or a 2D" in result.warning
