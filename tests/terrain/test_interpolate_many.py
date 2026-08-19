"""
Regression suite for TD-001: BaseInterpolator.interpolate_many() --
re-implemented in this session (PR19) after confirming, against
Hernán's own freshly re-exported real repository dump, that this
fix (originally described as already completed) was never actually
persisted -- only topocore.terrain.tin.TIN.from_mesh() and the four
interpolator classes' EXISTING scalar interpolate() behavior were
present. TD-002/003/004 WERE confirmed present and correct in the
same dump; this gap was isolated to TD-001 specifically.

Primary verification strategy, matching the project's own established
convention: interpolate_many() is checked against interpolate()
called once per point (not against a hand-computed value) -- the
same convention already used for TD-001's original design.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.geometry.point3d import Point3D
from topocore.terrain.barycentric import BarycentricInterpolator
from topocore.terrain.base import BaseInterpolator
from topocore.terrain.exceptions import InterpolationError
from topocore.terrain.idw import IDWInterpolator
from topocore.terrain.linear import LinearInterpolator
from topocore.terrain.nearest import NearestInterpolator
from topocore.terrain.tin import TIN

_INTERPOLATOR_CLASSES = [
    NearestInterpolator,
    IDWInterpolator,
    BarycentricInterpolator,
    LinearInterpolator,
]


@pytest.fixture
def two_triangle_tin() -> TIN:
    vertices = (
        Point3D(0.0, 0.0, 10.0),
        Point3D(1.0, 0.0, 5.0),
        Point3D(0.0, 1.0, 10.0),
        Point3D(1.0, 1.0, 5.0),
    )
    simplices = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return TIN.from_mesh(vertices, simplices)


# ----------------------------------------------------------------------
# Inheritance: the actual TD-001 bug -- none of the 4 classes
# inherited from BaseInterpolator at all, confirmed via
# issubclass(), not just "does it have the method".
# ----------------------------------------------------------------------


@pytest.mark.parametrize("interpolator_class", _INTERPOLATOR_CLASSES)
def test_interpolator_inherits_base_interpolator(interpolator_class: type) -> None:
    assert issubclass(interpolator_class, BaseInterpolator)


# ----------------------------------------------------------------------
# interpolate_many() vs interpolate() called per-point -- the
# project's own established verification convention.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("interpolator_class", _INTERPOLATOR_CLASSES)
def test_interpolate_many_matches_scalar_interpolate(interpolator_class: type, two_triangle_tin: TIN) -> None:
    interpolator = interpolator_class(two_triangle_tin)

    xs = np.array([0.1, 0.5, 0.9, 0.5, 0.25])
    ys = np.array([0.1, 0.5, 0.9, 0.9, 0.75])

    many_result = interpolator.interpolate_many(xs, ys)
    scalar_result = np.array([interpolator.interpolate(float(x), float(y)) for x, y in zip(xs, ys, strict=True)])

    np.testing.assert_allclose(many_result, scalar_result)


@pytest.mark.parametrize("interpolator_class", _INTERPOLATOR_CLASSES)
def test_interpolate_many_single_point(interpolator_class: type, two_triangle_tin: TIN) -> None:
    interpolator = interpolator_class(two_triangle_tin)

    result = interpolator.interpolate_many(np.array([0.5]), np.array([0.5]))

    assert result.shape == (1,)
    assert result[0] == pytest.approx(interpolator.interpolate(0.5, 0.5))


@pytest.mark.parametrize("interpolator_class", _INTERPOLATOR_CLASSES[:3])  # Linear delegates, same result
def test_interpolate_many_returns_float64_array(interpolator_class: type, two_triangle_tin: TIN) -> None:
    interpolator = interpolator_class(two_triangle_tin)
    result = interpolator.interpolate_many(np.array([0.5]), np.array([0.5]))
    assert result.dtype == np.float64


# ----------------------------------------------------------------------
# NearestInterpolator: tie-breaking must match min()'s "first
# occurrence" behavior exactly, not an arbitrary numpy tie.
# ----------------------------------------------------------------------


def test_nearest_interpolate_many_tie_breaks_to_first_vertex() -> None:
    """
    Two vertices exactly equidistant from the query point -- both
    interpolate() (via Python's min()) and interpolate_many() (via
    np.argmin()) must resolve to the FIRST one in vertices order,
    and must agree with each other.
    """
    vertices = (
        Point3D(-1.0, 0.0, 100.0),  # index 0, distance 1 from (0,0)
        Point3D(1.0, 0.0, 200.0),  # index 1, distance 1 from (0,0) -- exact tie
        Point3D(0.0, 2.0, 300.0),
    )
    simplices = np.array([[0, 1, 2]], dtype=np.int32)
    tin = TIN.from_mesh(vertices, simplices)
    interpolator = NearestInterpolator(tin)

    scalar_result = interpolator.interpolate(0.0, 0.0)
    many_result = interpolator.interpolate_many(np.array([0.0]), np.array([0.0]))

    assert scalar_result == 100.0  # first vertex wins the tie
    assert many_result[0] == scalar_result


# ----------------------------------------------------------------------
# IDWInterpolator: exact-vertex-match short-circuit must survive
# vectorization (not silently become a division involving inf).
# ----------------------------------------------------------------------


def test_idw_interpolate_many_exact_vertex_match(two_triangle_tin: TIN) -> None:
    interpolator = IDWInterpolator(two_triangle_tin)

    # Query exactly at vertex 0's coordinates (0.0, 0.0) -> z=10.0
    # exactly, not a weighted blend with other vertices.
    result = interpolator.interpolate_many(np.array([0.0]), np.array([0.0]))

    assert result[0] == pytest.approx(10.0)


def test_idw_interpolate_many_mixed_exact_and_normal_rows(
    two_triangle_tin: TIN,
) -> None:
    """
    A batch with BOTH an exact-vertex-match query and a normal
    (weighted) query must handle each row independently -- the
    normal row's result must not be corrupted by the exact-match
    row's special case, or vice versa.
    """
    interpolator = IDWInterpolator(two_triangle_tin)

    xs = np.array([0.0, 0.5])  # row 0: exact match; row 1: normal
    ys = np.array([0.0, 0.5])

    many_result = interpolator.interpolate_many(xs, ys)
    scalar_result = np.array([interpolator.interpolate(float(x), float(y)) for x, y in zip(xs, ys, strict=True)])

    np.testing.assert_allclose(many_result, scalar_result)
    assert many_result[0] == pytest.approx(10.0)  # exact vertex 0


# ----------------------------------------------------------------------
# BarycentricInterpolator: interpolate_many() propagates
# InterpolationError for out-of-hull points, same as interpolate().
# ----------------------------------------------------------------------


def test_barycentric_interpolate_many_raises_for_out_of_hull_point(
    two_triangle_tin: TIN,
) -> None:
    interpolator = BarycentricInterpolator(two_triangle_tin)

    with pytest.raises(InterpolationError):
        interpolator.interpolate_many(np.array([100.0]), np.array([100.0]))


# ----------------------------------------------------------------------
# LinearInterpolator: interpolate_many() genuinely delegates to
# BarycentricInterpolator's interpolate_many(), not a separate
# reimplementation that could silently drift from it.
# ----------------------------------------------------------------------


def test_linear_interpolate_many_matches_barycentric_exactly(
    two_triangle_tin: TIN,
) -> None:
    linear = LinearInterpolator(two_triangle_tin)
    barycentric = BarycentricInterpolator(two_triangle_tin)

    xs = np.array([0.2, 0.6, 0.8])
    ys = np.array([0.3, 0.4, 0.7])

    np.testing.assert_array_equal(
        linear.interpolate_many(xs, ys),
        barycentric.interpolate_many(xs, ys),
    )
