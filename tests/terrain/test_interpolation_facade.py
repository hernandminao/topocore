"""
Regression suite for topocore.terrain.interpolation.TerrainInterpolator
-- PR19.

Includes a real bug found and fixed in this session: enums.py and
interpolation.py each declared their own InterpolationMethod (same
name, overlapping-but-different members -- NATURAL_NEIGHBOR, never
implemented anywhere in the codebase, vs. NEAREST, which is). Since
TerrainInterpolator.interpolate() compared method values with `is`
(identity) rather than `==`, passing enums.InterpolationMethod.LINEAR
(a natural import path, since every other Terrain enum lives in
enums.py) silently fell through to NEAREST instead of LINEAR -- no
error, just a wrong elevation. Confirmed against a real Delaunay TIN:
7.5 (correct linear) vs. 10.0 (silently NEAREST) for the identical
query. Fixed by consolidating to a single InterpolationMethod
(enums.py is now the sole definition; interpolation.py imports it).
"""

from __future__ import annotations

import pytest

from topocore.geometry.point3d import Point3D
from topocore.terrain.enums import InterpolationMethod as EnumsInterpolationMethod
from topocore.terrain.interpolation import InterpolationMethod, TerrainInterpolator
from topocore.terrain.tin import TIN


@pytest.fixture
def asymmetric_tin() -> TIN:
    """
    A TIN whose 4 corner elevations are all different, so LINEAR and
    NEAREST genuinely disagree at the query point (0.5, 0.5) -- the
    exact configuration that exposed the bug.
    """
    points = (
        Point3D(0.0, 0.0, 10.0),
        Point3D(1.0, 0.0, 20.0),
        Point3D(0.0, 1.0, 30.0),
        Point3D(1.0, 1.0, 5.0),
    )
    return TIN.from_points(points)


def test_enums_and_interpolation_are_the_same_class() -> None:
    """
    The exact regression this fix targets: there must be only ONE
    InterpolationMethod, not two same-named classes that happen to
    share some string values.
    """
    assert EnumsInterpolationMethod is InterpolationMethod


def test_method_from_enums_module_gives_correct_linear_result(
    asymmetric_tin: TIN,
) -> None:
    """
    Before the fix, this exact call silently used NEAREST (10.0)
    instead of LINEAR (7.5) -- passing the enum from the "natural"
    import location (enums.py, alongside every other Terrain enum).
    """
    interpolator = TerrainInterpolator(asymmetric_tin, method=EnumsInterpolationMethod.LINEAR)
    result = interpolator.interpolate(0.5, 0.5)

    assert result == pytest.approx(7.5)
    assert result != pytest.approx(10.0)  # 10.0 is what silent-NEAREST would have given


@pytest.mark.parametrize("method", list(InterpolationMethod))
def test_every_method_is_reachable_via_either_import_path(method: InterpolationMethod, asymmetric_tin: TIN) -> None:
    """
    For every method, importing it from enums.py or interpolation.py
    must produce the identical interpolator behavior -- there is
    only one enum now, so this is nearly tautological, but verifies
    no method was dropped/renamed in the consolidation (NEAREST was
    kept; NATURAL_NEIGHBOR, unimplemented, was not).
    """
    from_interpolation = TerrainInterpolator(asymmetric_tin, method=method).interpolate(0.5, 0.5)
    from_enums = TerrainInterpolator(asymmetric_tin, method=EnumsInterpolationMethod(method.value)).interpolate(
        0.5, 0.5
    )
    assert from_interpolation == from_enums


def test_method_can_be_changed_after_construction(asymmetric_tin: TIN) -> None:
    interpolator = TerrainInterpolator(asymmetric_tin, method=InterpolationMethod.LINEAR)
    linear_result = interpolator.interpolate(0.5, 0.5)

    interpolator.method = InterpolationMethod.NEAREST
    nearest_result = interpolator.interpolate(0.5, 0.5)

    assert linear_result != nearest_result


def test_barycentric_and_linear_agree() -> None:
    """
    On a TIN, linear and barycentric interpolation are mathematically
    identical (see LinearInterpolator's own docstring) -- confirmed
    through the facade, not just the underlying classes directly.
    """
    points = (Point3D(0.0, 0.0, 10.0), Point3D(1.0, 0.0, 20.0), Point3D(0.0, 1.0, 30.0))
    tin = TIN.from_points(points)

    linear = TerrainInterpolator(tin, method=InterpolationMethod.LINEAR).interpolate(0.25, 0.25)
    barycentric = TerrainInterpolator(tin, method=InterpolationMethod.BARYCENTRIC).interpolate(0.25, 0.25)

    assert linear == pytest.approx(barycentric)


def test_interpolate_point_matches_interpolate(asymmetric_tin: TIN) -> None:
    interpolator = TerrainInterpolator(asymmetric_tin, method=InterpolationMethod.IDW)
    point = Point3D(0.5, 0.5, 0.0)  # z is irrelevant, only x/y are used

    assert interpolator.interpolate_point(point) == interpolator.interpolate(0.5, 0.5)


def test_callable_matches_interpolate(asymmetric_tin: TIN) -> None:
    interpolator = TerrainInterpolator(asymmetric_tin, method=InterpolationMethod.NEAREST)
    assert interpolator(0.5, 0.5) == interpolator.interpolate(0.5, 0.5)


def test_idw_power_is_configurable(asymmetric_tin: TIN) -> None:
    # (0.5, 0.5) is equidistant from all 4 corners -- power can't
    # matter there (weights end up equal regardless of exponent).
    # Use an off-center point where distances genuinely differ.
    default_power = TerrainInterpolator(asymmetric_tin, method=InterpolationMethod.IDW).interpolate(0.2, 0.8)
    high_power = TerrainInterpolator(asymmetric_tin, method=InterpolationMethod.IDW, power=10.0).interpolate(0.2, 0.8)
    assert default_power != high_power  # different power -> different weighting -> different result
