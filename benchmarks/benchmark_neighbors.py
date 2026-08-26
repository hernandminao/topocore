"""
benchmarks.benchmark_neighbors
================================

Reproducible benchmark for the PR21.2 opt-in `workers` parallelism
in topocore.processing.neighbors.kdtree.KDTreeNeighborSearch.

Run directly:

    python -m benchmarks.benchmark_neighbors

Measures, at several realistic point-cloud sizes:
    - KD-tree construction time
    - knn_many() query time (workers=1 vs workers=-1)
    - covariance + eigendecomposition time (already fully vectorized,
      included here only for context on where total pipeline time
      actually goes)

Per PR21's own governing rule: run this on the target multi-core
machine to confirm the actual speedup workers=-1 provides -- the
audit that led to this change was performed on a single-core
sandbox, where workers=-1 could not be shown to help (nothing to
parallelize across), even though the underlying scipy mechanism is
sound and the numerical-equivalence regression suite (see
tests/processing/neighbors/test_workers_parallelism.py) confirms it
never changes results.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import time

import numpy as np

from topocore.processing.config import NeighborConfig
from topocore.processing.neighbors.manager import NeighborhoodManager


def _make_points(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 100, (n, 3))


def run_all() -> None:
    print(f"{'n_points':>10} {'workers':>8} {'build':>10} {'knn_many':>10} {'pca_math':>10} {'total':>10}")

    for n in (10_000, 50_000, 100_000, 500_000):
        points = _make_points(n)

        for workers in (1, -1):
            t0 = time.perf_counter()
            manager = NeighborhoodManager.from_array(points, config=NeighborConfig(workers=workers))
            t_build = time.perf_counter() - t0

            t0 = time.perf_counter()
            indices, _distances = manager.knn_many(k=10)
            t_knn = time.perf_counter() - t0

            neighbor_points = points[indices]
            t0 = time.perf_counter()
            centroids = neighbor_points.mean(axis=1, keepdims=True)
            centered = neighbor_points - centroids
            covariances = np.einsum("nki,nkj->nij", centered, centered) / 10
            np.linalg.eigh(covariances)
            t_pca_math = time.perf_counter() - t0

            total = t_build + t_knn + t_pca_math
            print(f"{n:>10} {workers:>8} {t_build:>9.3f}s {t_knn:>9.3f}s {t_pca_math:>9.3f}s {total:>9.3f}s")


if __name__ == "__main__":
    run_all()
