from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ============================================================================
# Scalar Types
# ============================================================================

type Coordinate = float
type Elevation = float
type Distance = float
type Angle = float

# ============================================================================
# NumPy Array Types
# ============================================================================

type FloatArray1D = NDArray[np.float64]
type FloatArray2D = NDArray[np.float64]
type FloatArray3D = NDArray[np.float64]

type IntArray1D = NDArray[np.int64]
type IntArray2D = NDArray[np.int64]

type UInt8Array1D = NDArray[np.uint8]
type UInt16Array1D = NDArray[np.uint16]
type UInt32Array1D = NDArray[np.uint32]

type BoolArray1D = NDArray[np.bool_]

type Matrix3x3 = NDArray[np.float64]
type Matrix4x4 = NDArray[np.float64]

type Vector3D = NDArray[np.float64]

type PointAttributeArray = (
    FloatArray1D
    | FloatArray2D
    | FloatArray3D
    | IntArray1D
    | IntArray2D
    | UInt8Array1D
    | UInt16Array1D
    | UInt32Array1D
    | BoolArray1D
    | Matrix3x3
    | Matrix4x4
    | Vector3D
)
