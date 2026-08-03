"""
topocore.processing.types
=========================

Type aliases used throughout the processing module.

This module centralizes the type definitions for the processing subsystem,
providing clear and consistent type hints across all modules. This helps
with static type checking and improves code readability.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Any, Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray

from topocore.core.types import (
    Vector3D,
)

# # ============================================================================
# # Numeric Arrays
# # ============================================================================

# #: A one-dimensional NumPy array of 64-bit floats.
FloatArray1D: TypeAlias = NDArray[np.float64]

# #: A two-dimensional NumPy array of 64-bit floats (e.g., Nx3 coordinates).
FloatArray2D: TypeAlias = NDArray[np.float64]

# #: A one-dimensional NumPy array of boolean values (masks).
BoolArray1D: TypeAlias = NDArray[np.bool_]

# #: A one-dimensional NumPy array of integer indices.
IntArray1D: TypeAlias = NDArray[np.int64]

# #: A two-dimensional NumPy array of integer indices.
IntArray2D: TypeAlias = NDArray[np.int64]

# ============================================================================
# Processing Types
# ============================================================================

#: A callable that transforms a point cloud and returns a modified point cloud.
Processor: TypeAlias = Callable[["PointCloud"], "PointCloud"]  # type: ignore[name-defined] # noqa: F821

#: A callable that computes a mask (boolean array) for a point cloud.
MaskFunction: TypeAlias = Callable[["PointCloud"], BoolArray1D]  # type: ignore[name-defined] # noqa: F821

#: A callable that extracts a value (e.g., a scalar) from a point cloud.
ValueFunction: TypeAlias = Callable[["PointCloud"], Any]  # type: ignore[name-defined] # noqa: F821

#: A generic point cloud transformation function.
TransformFunction: TypeAlias = Callable[["PointCloud"], "PointCloud"]  # type: ignore[name-defined] # noqa: F821

# ============================================================================
# Pipeline Types
# ============================================================================

#: A step in a processing pipeline, consisting of a name and a callable.
PipelineStep: TypeAlias = tuple[str, Processor]

#: A sequence of pipeline steps to be executed in order.
Pipeline: TypeAlias = list[PipelineStep]

# ============================================================================
# Neighbor Types
# ============================================================================

#: A list of neighbor indices for a single point.
NeighborIndices: TypeAlias = IntArray1D

#: A list of neighbor distances for a single point.
NeighborDistances: TypeAlias = FloatArray1D

#: A batch of neighbor indices for multiple points.
NeighborIndexBatch: TypeAlias = IntArray2D

#: A batch of neighbor distances for multiple points.
NeighborDistanceBatch: TypeAlias = FloatArray2D

# ============================================================================
# Feature Types
# ============================================================================

#: A scalar feature value for a single point.
ScalarFeature: TypeAlias = float

#: A vector feature value for a single point (e.g., a normal).
VectorFeature: TypeAlias = tuple[float, ...] | NDArray[np.floating]

#: A dictionary mapping attribute names to feature values.
FeatureDict: TypeAlias = dict[str, Any]

# ============================================================================
# Registration Types
# ============================================================================

#: A 4x4 homogeneous transformation matrix.
TransformationMatrix: TypeAlias = NDArray[np.float64]

# ============================================================================
# Segmentation Types
# ============================================================================

#: A list of point indices that belong to the same segment.
SegmentIndices: TypeAlias = IntArray1D

#: A list of segments (each segment is a list/array of point indices).
Segments: TypeAlias = list[SegmentIndices]

__all__ = [
    "FloatArray1D",
    "FloatArray2D",
    "BoolArray1D",
    "IntArray1D",
    "IntArray2D",
    "Vector3D",
    "Processor",
    "MaskFunction",
    "ValueFunction",
    "TransformFunction",
    "PipelineStep",
    "Pipeline",
    "NeighborIndices",
    "NeighborDistances",
    "NeighborIndexBatch",
    "NeighborDistanceBatch",
    "ScalarFeature",
    "VectorFeature",
    "FeatureDict",
    "TransformationMatrix",
    "SegmentIndices",
    "Segments",
]
