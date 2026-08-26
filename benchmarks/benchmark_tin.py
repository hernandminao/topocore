"""
benchmarks.benchmark_tin
=========================

Reproducible benchmark for topocore.terrain.tin.TIN.find_triangle()
and topocore.analysis.comparison.TINComparison -- the bottleneck
demonstrated during PR20's audit (documented O(triangle_count)
brute-force scan, multiplied by grid points in TINComparison, giving
O(grid_points x triangle_count) overall).

Run directly:

    python -m benchmarks.benchmark_tin

Per PR21's own governing rule: "no optimization is accepted just
because it feels faster" -- this script is the reproducible evidence
a PR21 change must be measured against, both for speed (this module)
and for correctness (see tests/terrain/test_find_triangle_index.py,
which confirms the accelerated find_triangle() returns results
identical to a reference brute-force scan).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from topocore.analysis.comparison import TINComparison
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One benchmark measurement."""

    label: str
    n_points: int
    triangle_count: int
    elapsed_seconds: float
    extra: str = ""

    def __str__(self) -> str:
        extra = f"  {self.extra}" if self.extra else ""
        return (
            f"{self.label:<24} n_points={self.n_points:>6}  "
            f"triangles={self.triangle_count:>6}  "
            f"time={self.elapsed_seconds:>9.4f}s{extra}"
        )


def _make_tin(n_points: int, z_base: float, seed: int) -> TIN:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 100, n_points)
    ys = rng.uniform(0, 100, n_points)
    zs = z_base + rng.uniform(0, 2, n_points)
    points = tuple(Point3D(float(x), float(y), float(z)) for x, y, z in zip(xs, ys, zs, strict=True))
    return TIN.from_points(points)


def benchmark_find_triangle(n_points: int, n_queries: int = 1000, seed: int = 0) -> BenchmarkResult:
    """
    Benchmark TIN.find_triangle() over `n_queries` random points
    inside the TIN's domain. The FIRST call (which lazily builds the
    PR21.5 spatial index, if present) is excluded from the timed
    loop, since that one-time cost is amortized across every
    subsequent query on the same TIN -- exactly how it is actually
    used by real callers (TINComparison, DTM.from_tin(), etc.).
    """
    tin = _make_tin(n_points, z_base=0.0, seed=seed)

    rng = np.random.default_rng(seed + 1)
    query_xs = rng.uniform(10, 90, n_queries)
    query_ys = rng.uniform(10, 90, n_queries)

    tin.find_triangle(50.0, 50.0)  # warm up / build any lazy index

    start = time.perf_counter()
    for x, y in zip(query_xs, query_ys, strict=True):
        tin.find_triangle(float(x), float(y))
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        label="find_triangle",
        n_points=n_points,
        triangle_count=tin.triangle_count,
        elapsed_seconds=elapsed,
        extra=f"({n_queries} queries, {elapsed / n_queries * 1000:.4f}ms/query)",
    )


def benchmark_tin_comparison(n_points: int, resolution: float, seed: int = 0) -> BenchmarkResult:
    """Benchmark a full TINComparison.compute() call end to end."""
    existing = _make_tin(n_points, z_base=100.0, seed=seed)
    proposed = _make_tin(n_points, z_base=90.0, seed=seed + 1)

    start = time.perf_counter()
    result = TINComparison(resolution=resolution, max_grid_cells=10_000_000).compute(existing, proposed)
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        label="TINComparison",
        n_points=n_points,
        triangle_count=existing.triangle_count,
        elapsed_seconds=elapsed,
        extra=f"(resolution={resolution}, grid_cells={result.difference.size})",
    )


def run_all() -> list[BenchmarkResult]:
    """Run the full benchmark suite and return every result."""
    results: list[BenchmarkResult] = []

    for n_points in (50, 200, 500, 1000, 2000):
        results.append(benchmark_find_triangle(n_points))

    for n_points, resolution in ((200, 5.0), (500, 5.0), (1000, 5.0), (1000, 2.0)):
        results.append(benchmark_tin_comparison(n_points, resolution))

    return results


if __name__ == "__main__":
    for result in run_all():
        print(result)
