"""
topocore.io.ply.enums
=====================

Enumerations used by the PLY module.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from enum import Enum
from typing import Final

import numpy as np


class PLYFormat(str, Enum):
    """
    Supported PLY storage formats.
    """

    ASCII = "ascii"

    BINARY_LITTLE_ENDIAN = "binary_little_endian"

    BINARY_BIG_ENDIAN = "binary_big_endian"


class PLYScalarType(str, Enum):
    """
    Scalar types defined by the PLY specification.
    """

    CHAR = "char"
    INT8 = "int8"

    UCHAR = "uchar"
    UINT8 = "uint8"

    SHORT = "short"
    INT16 = "int16"

    USHORT = "ushort"
    UINT16 = "uint16"

    INT = "int"
    INT32 = "int32"

    UINT = "uint"
    UINT32 = "uint32"

    FLOAT = "float"
    FLOAT32 = "float32"

    DOUBLE = "double"
    FLOAT64 = "float64"

    @property
    def numpy_dtype(
        self,
    ) -> np.dtype:
        """
        Return the equivalent NumPy dtype.
        """
        return PLY_NUMPY_DTYPES[self]


PLY_NUMPY_DTYPES: Final[
    dict[
        PLYScalarType,
        np.dtype,
    ]
] = {

    PLYScalarType.CHAR: np.dtype("i1"),
    PLYScalarType.INT8: np.dtype("i1"),

    PLYScalarType.UCHAR: np.dtype("u1"),
    PLYScalarType.UINT8: np.dtype("u1"),

    PLYScalarType.SHORT: np.dtype("i2"),
    PLYScalarType.INT16: np.dtype("i2"),

    PLYScalarType.USHORT: np.dtype("u2"),
    PLYScalarType.UINT16: np.dtype("u2"),

    PLYScalarType.INT: np.dtype("i4"),
    PLYScalarType.INT32: np.dtype("i4"),

    PLYScalarType.UINT: np.dtype("u4"),
    PLYScalarType.UINT32: np.dtype("u4"),

    PLYScalarType.FLOAT: np.dtype("f4"),
    PLYScalarType.FLOAT32: np.dtype("f4"),

    PLYScalarType.DOUBLE: np.dtype("f8"),
    PLYScalarType.FLOAT64: np.dtype("f8"),
}


__all__ = [
    "PLYFormat",
    "PLYScalarType",
    "PLY_NUMPY_DTYPES",
]