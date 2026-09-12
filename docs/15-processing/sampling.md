# Sampling

5 downsampling strategies, each reducing point count by a different
criterion.

```python
from topocore.processing.sampling import RandomSampler

sampled = RandomSampler(fraction=0.1, seed=42).sample(cloud)
```

**The method is `.sample(cloud)`, not `.apply(cloud)`** — a different
contract from `filters` (`Filter.apply()`) and `ground`
(`GroundManager.extract()`), even though all 3 conceptually reduce a
`PointCloud` to a subset. Confirmed directly: calling `.apply()` on a
`Sampler` raises `AttributeError`, not a deprecation warning or an
alias.

## The 5 samplers and their real constructor signatures

```python
RandomSampler(fraction: float | None = None, count: int | None = None, seed: int | None = None)
UniformSampler(step: int, offset: int = 0)
VoxelSampler(voxel_size: float, method: Literal["centroid","closest","random","all"] = "centroid", seed: int | None = None)
StratifiedSampler(cell_size: float, samples_per_cell: int = 1, method: Literal["random","centroid","closest"] = "random", seed: int | None = None)
DensitySampler(target_density: float, radius: float = 1.0, min_fraction: float = 0.1, max_fraction: float = 1.0, seed: int | None = None)
```

Verified directly on a 1,000-point uniform-random cloud:
`RandomSampler(fraction=0.1)` and `UniformSampler(step=10)` both
produce exactly 100 points; `VoxelSampler(voxel_size=1.0)` produced
646 (voxel-based sampling doesn't target an exact count — it groups
points into `voxel_size`-sided cells and keeps one representative per
occupied cell, so the result depends on the cloud's own spatial
distribution, not a requested fraction/count).

## `RandomSampler`: `fraction` and `count` are mutually exclusive, not combinable

Pass exactly one of `fraction` (a proportion of the total) or `count`
(an absolute number of points) — not both, and not neither.

## `VoxelSampler`'s 4 selection methods, within each occupied voxel

- `"centroid"` — a synthetic point at the voxel's own geometric mean (not a real point in the source cloud).
- `"closest"` — the real point nearest to that centroid.
- `"random"` — one real point, chosen uniformly at random.
- `"all"` — every point in the voxel is kept (this mode does not
  actually downsample by count; it's for voxel-based grouping/analysis
  where you still want the original points).

## `StratifiedSampler` vs. `VoxelSampler` — both grid-based, different intent

Both partition space into cells, but `StratifiedSampler` targets a
fixed `samples_per_cell` (potentially several points per cell, chosen
by its own `method`), while `VoxelSampler` always reduces each
occupied voxel to exactly one representative. Use `StratifiedSampler`
when you want a controlled, non-uniform-density result with more than
one sample per region; use `VoxelSampler` for straightforward spatial
decimation.

## `DensitySampler`

Targets a specific output point density (points per unit
area/volume, matching `neighbors.NeighborhoodManager.local_density()`'s
own convention) rather than a fixed fraction or count — the actual
sampling rate varies locally, bounded by `min_fraction`/`max_fraction`,
so a sparse region isn't thinned as aggressively as a dense one.

## `SamplingManager`

Selects between the 5 samplers by name, matching the manager pattern
used throughout this section.
