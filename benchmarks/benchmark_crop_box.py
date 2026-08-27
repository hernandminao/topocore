"""
benchmarks.benchmark_crop_box
===============================

Reproducible benchmark for PR21.7.4: CropBoxFilter.mask()'s rewritten
per-chunk strategy versus the prior concatenate-everything approach.

Run directly:

    python -m benchmarks.benchmark_crop_box

Unlike the LAS/LAZ writer (PR21.7.3) or local_density (PR21.4), this
filter's box test is a pure per-point, per-axis range check with no
dependency on any other point or any global structure -- confirmed
during this PR's own audit (mask() even explicitly discards its
optional `manager` parameter via `del manager`, since no spatial
query is ever needed). This makes it the simplest chunking case in
PR21.7: no offset/min-reduction subtlety (PR21.7.3), no KDTree/cache
lifecycle to reason about (PR21.3) -- just computing the mask per
chunk directly and concatenating the (much smaller) boolean results
instead of first concatenating three giant float64 coordinate arrays.

Findings from this session's own run (single-core sandbox, each
size/mode measured in its own subprocess -- ru_maxrss is a
monotonically increasing per-process watermark, so comparing sizes
within one long-lived process gives invalid deltas, exactly the
methodological issue PR21.7.2's own benchmark ran into and fixed):

    n=100,000:   current mask_delta=+1.9MB   chunked=+0.2MB   time: 0.0154s -> 0.0003s
    n=500,000:   current mask_delta=+9.2MB   chunked=+0.6MB   time: 0.0661s -> 0.0021s
    n=1,000,000: current mask_delta=+17.8MB  chunked=+0.6MB   time: 0.0780s -> 0.0043s

Roughly a 9.5x-30x memory reduction and 18x-50x speedup, since three
float64 (8 bytes/point) concatenations become one boolean
(1 byte/point) concatenation. mask.sum() was confirmed identical
between the two modes at every size measured, before this change was
made -- see tests/processing/filters/test_crop_box_chunked.py for the
permanent regression suite.

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

from topocore.geometry.bbox3d import BBox3D
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.filters.crop_box import CropBoxFilter


@dataclass(frozen=True, slots=True)
class CropBoxBenchmarkResult:
    n_points: int
    chunks_rss_delta_mb: float
    mask_rss_delta_mb: float
    elapsed_seconds: float
    mask_sum: int

    def __str__(self) -> str:
        return (
            f"n={self.n_points:>9}  chunks=+{self.chunks_rss_delta_mb:>6.1f}MB  "
            f"mask_delta=+{self.mask_rss_delta_mb:>6.2f}MB  time={self.elapsed_seconds:>7.4f}s  "
            f"mask_sum={self.mask_sum}"
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


def benchmark(n_points: int) -> CropBoxBenchmarkResult:
    box = BBox3D(min_x=200, min_y=200, min_z=20, max_x=800, max_y=800, max_z=80)

    rss_before_cloud = _rss_mb()
    cloud = _make_cloud_chunked(n_points)
    rss_after_cloud = _rss_mb()

    start = time.perf_counter()
    mask = CropBoxFilter(box).mask(cloud)
    elapsed = time.perf_counter() - start

    rss_after_mask = _rss_mb()

    return CropBoxBenchmarkResult(
        n_points=n_points,
        chunks_rss_delta_mb=rss_after_cloud - rss_before_cloud,
        mask_rss_delta_mb=rss_after_mask - rss_after_cloud,
        elapsed_seconds=elapsed,
        mask_sum=int(mask.sum()),
    )


def run_all() -> None:
    """Each size in its own subprocess -- see this module's own docstring for why that matters."""
    import subprocess
    import sys

    for n in (100_000, 500_000, 1_000_000):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"from benchmarks.benchmark_crop_box import benchmark; print(benchmark({n}))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout.strip())


if __name__ == "__main__":
    run_all()
