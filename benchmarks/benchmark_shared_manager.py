"""
benchmarks.benchmark_shared_manager
=====================================

Reproducible benchmark for PR21.3.4: does sharing one
NeighborhoodManager across NormalManager, PCAFeatures, and
RuleBasedClassifier (Case B) save meaningful time over each building
its own internally (Case A)?

Run directly:

    python -m benchmarks.benchmark_shared_manager

Deliberately uses DIFFERENT k per module (Normals k=10, PCAFeatures
k=8, RuleBasedClassifier's own pca_neighbors=20) to confirm sharing
the underlying KDTree/manager does not force a shared k -- k belongs
to the query (knn_many(k=...)), not the manager's identity, per the
PR21.3.2 audit's own finding.

Findings from this session's own run (single-core sandbox):

    n=20,000:  Case A=0.865s  Case B=0.663s  savings=23.3%
    n=100,000: Case A=5.477s  Case B=4.057s  savings=25.9%
    n=500,000: Case A=49.27s  Case B=42.03s  savings=14.7%

Savings DECLINE at larger n not because sharing helps less, but
because RuleBasedClassifier.classify() itself comes to dominate
total cost (74% of Case A's total time at n=500,000) with work
unrelated to KDTree construction (GroundManager, noise filtering,
rule evaluation) -- the KDTree-sharing saving is a roughly fixed
absolute amount, shrinking as a fraction of a total dominated by a
different, larger cost. Per PR21's own governing rule ("no
optimizar por intuicion"), this argues for measuring
RuleBasedClassifier's own internal cost breakdown (a distinct,
separate profiling question) before deciding whether a shared-cache
INFRASTRUCTURE (PR21.3.5) is worth building, versus simply
documenting that a caller who wants the already-available savings
can build one NeighborhoodManager and pass it explicitly to each
stage today -- no new caching code required, since PR21.3.3 already
made every consumer accept an external manager.

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

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.rules import RuleBasedClassifier
from topocore.processing.features.pca import PCAFeatures
from topocore.processing.neighbors.manager import NeighborhoodManager
from topocore.processing.normals.manager import NormalManager


@dataclass(frozen=True, slots=True)
class SharedManagerBenchmarkResult:
    n_points: int
    separate_total: float
    shared_total: float

    @property
    def savings_percent(self) -> float:
        return (1.0 - self.shared_total / self.separate_total) * 100.0

    def __str__(self) -> str:
        return (
            f"n={self.n_points:>7}  separate={self.separate_total:>8.3f}s  "
            f"shared={self.shared_total:>8.3f}s  savings={self.savings_percent:>5.1f}%"
        )


def _make_cloud(n: int, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 100, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 100, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 5, n)
    cloud.add_chunk(chunk)
    return cloud


def benchmark(n_points: int) -> SharedManagerBenchmarkResult:
    cloud = _make_cloud(n_points)

    # Case A: separate -- each module builds its own manager internally.
    normal_manager_a = NormalManager(method="pca", k=10)
    pca_features_a = PCAFeatures(k=8)
    classifier_a = RuleBasedClassifier(pca_neighbors=20)

    start = time.perf_counter()
    normal_manager_a.estimate(cloud)
    pca_features_a.compute_all(cloud)
    classifier_a.classify(cloud)
    separate_total = time.perf_counter() - start

    # Case B: shared -- one manager, built once, passed explicitly to each.
    normal_manager_b = NormalManager(method="pca", k=10)
    pca_features_b = PCAFeatures(k=8)
    classifier_b = RuleBasedClassifier(pca_neighbors=20)

    start = time.perf_counter()
    shared_manager = NeighborhoodManager.from_point_cloud(cloud)
    normal_manager_b.estimate(cloud, manager=shared_manager)
    pca_features_b.compute_all(cloud, manager=shared_manager)
    classifier_b.classify(cloud, manager=shared_manager)
    shared_total = time.perf_counter() - start

    return SharedManagerBenchmarkResult(n_points, separate_total, shared_total)


def run_all() -> None:
    for n in (20_000, 100_000):
        print(benchmark(n))


if __name__ == "__main__":
    run_all()
