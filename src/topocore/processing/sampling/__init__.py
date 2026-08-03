"""
topocore.processing.sampling
============================

Point cloud sampling and downsampling.

This package provides various methods for reducing point cloud density
while preserving essential characteristics:

- Random: random selection without replacement
- Uniform: systematic decimation (every N-th point)
- Voxel: voxel grid downsampling (centroid, closest, random)
- Stratified: stratified sampling by 2D grid cells
- Density: density-adaptive sampling

The samplers are designed to be interchangeable and can be used
in processing pipelines.

Public API
----------
- Sampler: abstract base class for samplers
- RandomSampler: random sampling
- UniformSampler: uniform decimation
- VoxelSampler: voxel grid downsampling
- StratifiedSampler: stratified sampling
- DensitySampler: density-adaptive sampling
- SamplingManager: high-level manager with method selection

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .base import Sampler
from .density import DensitySampler
from .manager import SamplingManager
from .random import RandomSampler
from .stratified import StratifiedSampler
from .uniform import UniformSampler
from .voxel import VoxelSampler

__all__ = [
    "Sampler",
    "RandomSampler",
    "UniformSampler",
    "VoxelSampler",
    "StratifiedSampler",
    "DensitySampler",
    "SamplingManager",
]
