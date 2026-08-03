from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# ============================================================================
# Scalar Types
# ============================================================================

Coordinate: TypeAlias = float
Elevation: TypeAlias = float
Distance: TypeAlias = float
Angle: TypeAlias = float

# ============================================================================
# NumPy Array Types
# ============================================================================

FloatArray1D: TypeAlias = NDArray[np.float64]
FloatArray2D: TypeAlias = NDArray[np.float64]
FloatArray3D: TypeAlias = NDArray[np.float64]

IntArray1D: TypeAlias = NDArray[np.int64]
IntArray2D: TypeAlias = NDArray[np.int64]

UInt8Array1D: TypeAlias = NDArray[np.uint8]
UInt16Array1D: TypeAlias = NDArray[np.uint16]
UInt32Array1D: TypeAlias = NDArray[np.uint32]

BoolArray1D: TypeAlias = NDArray[np.bool_]

Matrix3x3: TypeAlias = NDArray[np.float64]
Matrix4x4: TypeAlias = NDArray[np.float64]

Vector3D: TypeAlias = NDArray[np.float64]

PointAttributeArray: TypeAlias = (
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
