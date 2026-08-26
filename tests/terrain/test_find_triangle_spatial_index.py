"""
Regression suite for topocore.terrain.tin.TIN.find_triangle()'s
PR21.5 spatial-index acceleration.

The governing rule for PR21: no optimization is accepted just
because it "feels faster" -- it must be proven both correct (this
file) and faster (see benchmarks/benchmark_tin.py, run manually via
``python -m benchmarks.benchmark_tin``).

Correctness is verified against an independent reference
implementation: a plain brute-force O(triangle_count) scan using the
exact same _point_in_triangle test find_triangle() always used,
confirming the spatial index changes ONLY the search strategy, never
the result -- exercised over thousands of random query points
(including points outside the convex hull), plus the specific edge
cases already audited during PR20 (points on a shared triangle edge,
points exactly on the TIN's own vertices, points at the domain
boundary).

find_triangle()'s own public contract (signature, return value
semantics: triangle index or -1) is unchanged -- confirmed by every
existing test across the whole session that calls
find_triangle()/interpolate()/contains()/locate() (all delegate to
find_triangle() internally) continuing to pass unmodified.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN, _point_in_triangle


def _brute_force_find_triangle(tin: TIN, x: float, y: float) -> int:
    """Independent reference implementation -- the exact pre-PR21 algorithm, reimplemented here."""
    for index in range(tin.triangle_count):
        p1, p2, p3 = tin.triangle_vertices(index)
        if _point_in_triangle(x, y, p1, p2, p3):
            return index
    return -1


def _random_tin(n_points: int, seed: int) -> TIN:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 100, n_points)
    ys = rng.uniform(0, 100, n_points)
    zs = rng.uniform(0, 10, n_points)
    points = tuple(Point3D(float(x), float(y), float(z)) for x, y, z in zip(xs, ys, zs, strict=True))
    return TIN.from_points(points)


# ----------------------------------------------------------------------
# Correctness: identical results to the independent brute-force reference.
# ----------------------------------------------------------------------


def test_matches_brute_force_reference_over_many_random_queries() -> None:
    tin = _random_tin(n_points=500, seed=42)

    rng = np.random.default_rng(99)
    query_xs = rng.uniform(-20, 120, 3000)  # includes points outside the hull
    query_ys = rng.uniform(-20, 120, 3000)

    for qx, qy in zip(query_xs, query_ys, strict=True):
        fast = tin.find_triangle(float(qx), float(qy))
        reference = _brute_force_find_triangle(tin, float(qx), float(qy))
        assert fast == reference, f"mismatch at ({qx}, {qy}): indexed={fast}, brute-force={reference}"


def test_matches_reference_at_exact_tin_vertices() -> None:
    tin = _random_tin(n_points=200, seed=7)

    for vertex in tin.vertices[:20]:
        fast = tin.find_triangle(vertex.x, vertex.y)
        reference = _brute_force_find_triangle(tin, vertex.x, vertex.y)
        assert fast == reference


def test_matches_reference_on_shared_triangle_edge() -> None:
    """The exact PR20.5 case: a square split into 2 triangles along a diagonal."""
    points = (
        Point3D(0.0, 0.0, 0.0),
        Point3D(10.0, 0.0, 10.0),
        Point3D(0.0, 10.0, 20.0),
        Point3D(10.0, 10.0, 30.0),
    )
    tin = TIN.from_points(points)

    for t in np.linspace(0.01, 9.99, 30):
        fast = tin.find_triangle(float(t), float(t))
        reference = _brute_force_find_triangle(tin, float(t), float(t))
        assert fast == reference


def test_matches_reference_outside_convex_hull() -> None:
    tin = _random_tin(n_points=100, seed=3)

    for x, y in [(-50.0, -50.0), (500.0, 500.0), (-1.0, 50.0), (150.0, 50.0)]:
        assert tin.find_triangle(x, y) == -1
        assert tin.find_triangle(x, y) == _brute_force_find_triangle(tin, x, y)


def test_matches_reference_at_grid_cell_boundaries() -> None:
    """Points falling exactly on the spatial index's own internal cell boundaries -- must not be missed."""
    tin = _random_tin(n_points=300, seed=11)

    min_x, min_y, max_x, max_y = tin.bounds
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = min_x + fraction * (max_x - min_x)
        y = min_y + fraction * (max_y - min_y)
        assert tin.find_triangle(x, y) == _brute_force_find_triangle(tin, x, y)


# ----------------------------------------------------------------------
# find_triangle()'s contract is unchanged: interpolate/contains/locate.
# ----------------------------------------------------------------------


def test_interpolate_and_contains_still_delegate_correctly() -> None:
    tin = _random_tin(n_points=50, seed=1)
    x, y = 50.0, 50.0

    triangle_index = tin.find_triangle(x, y)
    if triangle_index >= 0:
        elevation = tin.interpolate(x, y)
        assert np.isfinite(elevation)
        assert tin.contains(x, y) is True
    else:
        with pytest.raises(ValueError):
            tin.interpolate(x, y)
        assert tin.contains(x, y) is False


def test_locate_delegates_to_find_triangle() -> None:
    tin = _random_tin(n_points=50, seed=2)
    point = Point3D(50.0, 50.0, 0.0)
    assert tin.locate(point) == tin.find_triangle(point.x, point.y)


# ----------------------------------------------------------------------
# Performance sanity: the index must not scale linearly with triangle count.
# ----------------------------------------------------------------------


def test_find_triangle_time_does_not_scale_linearly_with_triangle_count() -> None:
    """
    Not a strict micro-benchmark (too flaky for CI) -- just confirms
    the qualitative O(T) -> ~O(1) improvement demonstrated in
    benchmarks/benchmark_tin.py: querying a TIN with 10x more
    triangles must not take anywhere near 10x longer.
    """
    import time

    small_tin = _random_tin(n_points=50, seed=5)
    large_tin = _random_tin(n_points=2000, seed=5)  # ~40x more points -> ~40x more triangles

    rng = np.random.default_rng(6)
    query_xs = rng.uniform(10, 90, 200)
    query_ys = rng.uniform(10, 90, 200)

    small_tin.find_triangle(50.0, 50.0)
    large_tin.find_triangle(50.0, 50.0)

    start = time.perf_counter()
    for x, y in zip(query_xs, query_ys, strict=True):
        small_tin.find_triangle(float(x), float(y))
    small_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    for x, y in zip(query_xs, query_ys, strict=True):
        large_tin.find_triangle(float(x), float(y))
    large_elapsed = time.perf_counter() - start

    # A brute-force O(T) scan would take ~(large_tin.triangle_count /
    # small_tin.triangle_count) times longer. The indexed version
    # should be far closer to constant time -- allow generous margin
    # (5x) to keep this robust against CI noise while still failing
    # if the index genuinely isn't helping.
    scale_factor = large_tin.triangle_count / small_tin.triangle_count
    assert large_elapsed < small_elapsed * (scale_factor / 5.0) + 0.05
