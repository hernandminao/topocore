"""
benchmarks.benchmark_voxel_sampler
=====================================

Reproducible benchmark for PR21.7.5: VoxelSampler's "centroid" and
"closest" methods, rewritten from an O(N x n_groups) (effectively
quadratic for realistic data) per-group boolean-mask-scan loop to a
chunk-wise, vectorized O(N + G) accumulator.

Run directly:

    python -m benchmarks.benchmark_voxel_sampler

This was a genuine, severe, PRE-EXISTING complexity bug, independent
of the chunking/memory work elsewhere in PR21.7 -- confirmed via
direct measurement before this fix: 20,000 points took 1.267s;
100,000 points took 28.66s (~22.6x for a 5x increase in N, consistent
with quadratic scaling, not linear or O(N x G) with G roughly
constant). 500,000 points did not complete within 60 seconds.

Findings from this session's own run (single-core sandbox, each
size/method measured in its own subprocess):

BEFORE the accumulator fix:
    n=20,000,  centroid: time=1.267s
    n=100,000, centroid: time=28.658s
    n=100,000, closest:  time=25.556s
    n=500,000: did not complete within 60s (either method)

AFTER the accumulator fix:
    n=20,000,   centroid: time=0.074s   closest: time=0.033s
    n=100,000,  centroid: time=0.318s   closest: time=0.127s
    n=500,000,  centroid: time=2.212s   closest: time=0.638s

Roughly 17x-90x faster for centroid, up to ~200x faster for closest,
at the sizes where the old implementation could complete at all.

Honest trade-off, not hidden: the dict-based accumulator (keyed by
voxel coordinate, merging small per-chunk vectorized results) carries
more per-voxel-group Python-object overhead than the old single
merged-array approach. Peak memory at these same sizes is HIGHER,
not lower, than before this specific change -- this fix targets the
catastrophic time complexity, not memory (memory-focused work is
PR21.7.3/7.4's own concern, for writers/CropBox, which had no
comparable complexity bug to begin with). Numerical/order-of-merge
equivalence to the pre-PR21.7.5 reference algorithm (including
"closest"'s exact tie-breaking rule) was verified in this PR's own
regression suite -- see
tests/processing/sampling/test_voxel_chunked_accumulator.py.

"random" and "all" are unchanged by this PR (see that test suite's
own module docstring for why).

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
from typing import Literal

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.sampling.voxel import VoxelSampler


@dataclass(frozen=True, slots=True)
class VoxelBenchmarkResult:
    n_points: int
    method: str
    elapsed_seconds: float
    voxel_count: int

    def __str__(self) -> str:
        return (
            f"n={self.n_points:>9}  method={self.method:<9}  "
            f"time={self.elapsed_seconds:>8.4f}s  voxels={self.voxel_count}"
        )


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


def benchmark(n_points: int, method: Literal["centroid", "closest"]) -> VoxelBenchmarkResult:
    cloud = _make_cloud_chunked(n_points)

    start = time.perf_counter()
    result = VoxelSampler(voxel_size=5.0, method=method).sample(cloud)
    elapsed = time.perf_counter() - start

    return VoxelBenchmarkResult(
        n_points=n_points,
        method=method,
        elapsed_seconds=elapsed,
        voxel_count=result.point_count,
    )


def run_all() -> None:
    """Each size/method in its own subprocess -- matches this session's established, corrected methodology."""
    import subprocess
    import sys

    for method in ("centroid", "closest"):
        for n in (20_000, 100_000, 500_000):
            code = f"from benchmarks.benchmark_voxel_sampler import benchmark; print(benchmark({n}, {method!r}))"
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
            )
            print(result.stdout.strip())


if __name__ == "__main__":
    run_all()
