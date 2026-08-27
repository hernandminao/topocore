"""
benchmarks.benchmark_stratified_sampler
==========================================

Reproducible benchmark for PR21.7.6: StratifiedSampler's "centroid"
and "closest" methods, rewritten from an O(N x n_cells) (super-linear
for realistic data) per-cell boolean-mask-scan loop to a two-pass,
chunk-wise, vectorized accumulator.

Run directly:

    python -m benchmarks.benchmark_stratified_sampler

The SAME `for cell_idx in range(len(unique_cells)): mask =
group_labels == cell_idx` bug independently found and fixed in
VoxelSampler (PR21.7.5) was found here too, in a sibling module.
Milder scaling was observed here than in VoxelSampler's near-1-point-
per-voxel synthetic benchmark (2D cells vs. 3D voxels means fewer,
larger groups for the same point density), but still clearly
super-linear before this fix: 2,000 points took 0.054s; 20,000 points
(10x) took 1.180s (~21.9x, well above linear).

Two passes (not one, unlike VoxelSampler's "centroid") are needed
here because this method's "centroid" doesn't just return the
arithmetic centroid -- it finds the REAL point closest to it, which
requires the centroid to be fully known (pass 1) before searching for
that real point (pass 2); "closest" needs each cell's mean Z (pass 1)
before computing per-point distances (pass 2). Both passes iterate
the same already-in-memory Chunk objects PointCloud holds -- no data
is re-read, and neither pass concatenates chunk data into one global
array.

Findings from this session's own run (single-core sandbox, each
size/method measured in its own subprocess):

BEFORE the accumulator fix:
    n=2,000,  centroid: time=0.054s
    n=5,000,  centroid: time=0.137s
    n=10,000, centroid: time=0.395s
    n=20,000, centroid: time=1.180s

AFTER the accumulator fix:
    n=2,000,   centroid: 0.010s   closest: 0.012s
    n=5,000,   centroid: 0.022s   closest: 0.039s
    n=10,000,  centroid: 0.054s   closest: 0.064s
    n=20,000,  centroid: 0.095s   closest: 0.112s
    n=100,000, centroid: 0.380s   closest: 0.460s

Roughly 5x-12x+ faster at the sizes where both could be measured, and
scaling from 2,000 to 100,000 (50x the points) took only ~38x longer
for centroid -- close to linear, versus the old implementation's
clearly super-linear growth.

Honest note on "closest" specifically: unlike "centroid" (which only
needs O(G) memory for its running best-candidate accumulator) and
unlike VoxelSampler's own "closest" (a single best-candidate per
voxel), THIS "closest" supports `samples_per_cell > 1` via
NumPy's own (not formally stable-guaranteed) argsort tie-break --
preserving that exact behavior requires bucketing every point's
(index, distance) by cell rather than a slimmer streaming top-1
accumulator, so peak memory for "closest" here remains O(N)
(reorganized into per-cell buckets, not one single N-length array,
but not reduced to O(G) either). This targets the time-complexity
bug specifically, matching PR21.7.5's own precedent for VoxelSampler
of not conflating a time fix with an unrelated memory goal.

"random" is unchanged in this PR -- see
tests/processing/sampling/test_stratified_two_pass.py's own module
docstring for why (the same reservoir-sampling/RNG-reproducibility
question already deferred for VoxelSampler's "random").

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
from topocore.processing.sampling.stratified import StratifiedSampler


@dataclass(frozen=True, slots=True)
class StratifiedBenchmarkResult:
    n_points: int
    method: str
    elapsed_seconds: float
    selected_count: int

    def __str__(self) -> str:
        return (
            f"n={self.n_points:>9}  method={self.method:<9}  "
            f"time={self.elapsed_seconds:>8.4f}s  selected={self.selected_count}"
        )


def _make_cloud(n_points: int, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n_points, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 1000, n_points)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 1000, n_points)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 100, n_points)
    cloud.add_chunk(chunk)
    return cloud


def benchmark(n_points: int, method: Literal["centroid", "closest"]) -> StratifiedBenchmarkResult:
    cloud = _make_cloud(n_points)

    start = time.perf_counter()
    result = StratifiedSampler(cell_size=5.0, samples_per_cell=1, method=method, seed=0).sample(cloud)
    elapsed = time.perf_counter() - start

    return StratifiedBenchmarkResult(
        n_points=n_points,
        method=method,
        elapsed_seconds=elapsed,
        selected_count=result.point_count,
    )


def run_all() -> None:
    """Each size/method in its own subprocess -- matches this session's established, corrected methodology."""
    import subprocess
    import sys

    for method in ("centroid", "closest"):
        for n in (2_000, 20_000, 100_000):
            code = f"from benchmarks.benchmark_stratified_sampler import benchmark; print(benchmark({n}, {method!r}))"
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
            )
            print(result.stdout.strip())


if __name__ == "__main__":
    run_all()
