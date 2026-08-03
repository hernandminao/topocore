"""
topocore.processing
===================

Point cloud processing.

This is where TopoCore starts to resemble CloudCompare: spatial
neighbor search, sampling/downsampling, filtering, normal/curvature
estimation, and ground/vegetation/building/noise classification.

Implemented components
-----------------------

* Neighbors (KD-tree: KNN, radius search)
* Sampling (Random, Uniform, Voxel)
* Filters (Statistical/Radius Outlier, Pass Through, Crop Box,
  Clip Polygon)
* Normals (PCA-based normal + curvature estimation)
* Classification (ASPRS Ground / Low-High Vegetation / Building /
  Noise -- Water and Powerlines intentionally out of scope, see
  ``classification`` module docstring)

Not yet implemented: Registration (ICP).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .config import (
    DEFAULT_PROCESSING_CONFIG,
    ClassifierConfig,
    NeighborConfig,
    NormalConfig,
    OutlierConfig,
    ProcessingConfig,
    RegistrationConfig,
    SamplingConfig,
    SegmentationConfig,
)
from .exceptions import (
    FilterError,
    GroundError,
    NeighborError,
    NormalError,
    ProcessingError,
    RegistrationError,
    SamplingError,
    SegmentationError,
)
from .protocols import (
    Classifier,
    Estimator,
    Masker,
    Sampler,
    Segmenter,
    Transformer,
)
from .types import (
    BoolArray1D,
    FloatArray1D,
    FloatArray2D,
    IntArray1D,
    IntArray2D,
    MaskFunction,
    Pipeline,
    PipelineStep,
    Processor,
    TransformationMatrix,
    TransformFunction,
    ValueFunction,
)

__all__ = [
    # Configuration
    "NeighborConfig",
    "OutlierConfig",
    "NormalConfig",
    "ClassifierConfig",
    "SamplingConfig",
    "RegistrationConfig",
    "SegmentationConfig",
    "ProcessingConfig",
    "DEFAULT_PROCESSING_CONFIG",
    # Exceptions
    "ProcessingError",
    "NeighborError",
    "NormalError",
    "GroundError",
    "FilterError",
    "SamplingError",
    "RegistrationError",
    "SegmentationError",
    # Protocols
    "Transformer",
    "Masker",
    "Estimator",
    "Sampler",
    "Segmenter",
    "Classifier",
    # Types
    "FloatArray1D",
    "FloatArray2D",
    "BoolArray1D",
    "IntArray1D",
    "IntArray2D",
    "Processor",
    "MaskFunction",
    "ValueFunction",
    "TransformFunction",
    "PipelineStep",
    "Pipeline",
    "TransformationMatrix",
]
