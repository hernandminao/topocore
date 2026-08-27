"""
benchmarks.benchmark_grid_ground
===================================

Reproducible benchmark for PR21.7.8: GridGroundClassifier/
GridGroundElevationEstimator's rewritten two-pass, chunk-wise
accumulator versus the prior concatenate-everything approach.

Run directly:

    python -m benchmarks.benchmark_grid_ground

Unlike VoxelSampler/StratifiedSampler (PR21.7.5/7.6), this module did
NOT have the O(N x G) complexity bug -- confirmed via direct
benchmarking BEFORE this fix that its `np.minimum.at`-based per-cell
reduction was already genuinely linear (50x the points gave ~52.8x
the time, not the quadratic-ish scaling voxel.py/stratified.py had).
This fix targets only the memory overhead from concatenating every
chunk's X/Y/Z into one global array.

Honest finding: the improvement here is REAL but MODEST, unlike
CropBox's (PR21.7.4) dramatic 9.5x-30x reduction -- because this
algorithm genuinely needs TWO passes with per-cell dict-based
aggregation (pass 1: accumulate minimums; pass 2: look up each
point's cell minimum to build the final boolean mask/elevation
array), unlike CropBox's single, independent-per-point test. The
per-point Python loop needed in pass 2 (to combine each point's own
Z with its cell's minimum) has real, if smaller, overhead of its own.

Findings from this session's own run (single-core sandbox, each size
measured in its own subprocess):

BEFORE:
    n=100,000:   chunks=+3.3MB   classify_delta=+11.5MB  (3.5x)  time=0.137s
    n=500,000:   chunks=+15.3MB  classify_delta=+51.6MB  (3.4x)  time=0.712s
    n=1,000,000: chunks=+30.7MB  classify_delta=+101.7MB (3.3x)  time=1.365s

AFTER:
    n=100,000:   chunks=+3.2MB   classify_delta=+11.6MB  (3.6x)  time=0.159s
    n=500,000:   chunks=+15.5MB  classify_delta=+40.1MB  (2.6x)  time=0.729s
    n=1,000,000: chunks=+30.7MB  classify_delta=+78.9MB  (2.6x)  time=1.823s

Memory overhead ratio improved at larger scale (3.3x-3.5x -> 2.6x),
though not dramatically -- and wall time increased slightly, a real,
honestly-reported trade-off from the added per-point Python loop in
pass 2. Numerical equivalence to the pre-PR21.7.8 reference algorithm
was verified in this PR's own regression suite -- see
tests/processing/ground/test_grid_chunked.py.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.ground.grid import GridGroundClassifier


@dataclass(frozen=True, slots=True)
class GridGroundBenchmarkResult:
    n_points: int
    chunks_rss_delta_mb: float
    classify_rss_delta_mb: float
    elapsed_seconds: float
    ground_count: int

    @property
    def memory_ratio(self) -> float:
        return self.classify_rss_delta_mb / self.chunks_rss_delta_mb

    def __str__(self) -> str:
        return (
            f"n={self.n_points:>9}  chunks=+{self.chunks_rss_delta_mb:>5.1f}MB  "
            f"classify_delta=+{self.classify_rss_delta_mb:>6.1f}MB  ratio={self.memory_ratio:>4.1f}x  "
            f"time={self.elapsed_seconds:>7.4f}s  ground={self.ground_count}"
        )


def _rss_mb() -> float:
    """Return peak resident-set size in MB on Linux."""
    usage = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
    max_rss = int(usage.ru_maxrss)
    return float(max_rss) / 1024.0


def _make_cloud_chunked(n_total: int, chunk_size: int = 1_000_000, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    remaining = n_total
    while remaining > 0:
        n = min(chunk_size, remaining)
        chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
        chunk[PointAttribute.X][:] = rng.uniform(0, 1000, n)
        chunk[PointAttribute.Y][:] = rng.uniform(0, 1000, n)
        chunk[PointAttribute.Z][:] = rng.uniform(0, 100, n)
        cloud.add_chunk(chunk)
        remaining -= n
    return cloud


def benchmark(n_points: int) -> GridGroundBenchmarkResult:
    rss_before_cloud = _rss_mb()
    cloud = _make_cloud_chunked(n_points)
    rss_after_cloud = _rss_mb()

    start = time.perf_counter()
    mask = GridGroundClassifier(cell_size=5.0).classify(cloud)
    elapsed = time.perf_counter() - start

    rss_after_classify = _rss_mb()

    return GridGroundBenchmarkResult(
        n_points=n_points,
        chunks_rss_delta_mb=rss_after_cloud - rss_before_cloud,
        classify_rss_delta_mb=rss_after_classify - rss_after_cloud,
        elapsed_seconds=elapsed,
        ground_count=int(mask.sum()),
    )


def run_all() -> None:
    """Each size in its own subprocess -- ru_maxrss is a per-process watermark, per this session's established methodology."""
    import subprocess
    import sys

    for n in (100_000, 500_000, 1_000_000):
        code = f"from benchmarks.benchmark_grid_ground import benchmark; print(benchmark({n}))"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout.strip())


if __name__ == "__main__":
    run_all()
