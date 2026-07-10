"""
topocore.io.ascii.type_inference
================================

Utilities for inferring NumPy data types for ASCII columns.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import DTypeLike, NDArray

COLUMN_DTYPES: Final[dict[str, DTypeLike]] = {
    # Coordinates
    "x": np.float64,
    "y": np.float64,
    "z": np.float64,
    "easting": np.float64,
    "northing": np.float64,
    "elevation": np.float64,
    "height": np.float64,
    "gps_time": np.float64,
    # Colors
    "red": np.uint8,
    "green": np.uint8,
    "blue": np.uint8,
    # LAS
    "classification": np.uint8,
    "return_number": np.uint8,
    "number_of_returns": np.uint8,
    "intensity": np.int32,
    "point_source_id": np.int32,
}


class TypeInferer:
    """
    Converts ASCII string values into typed NumPy arrays.
    """

    @classmethod
    def infer(
        cls,
        column_name: str,
        values: list[str],
    ) -> NDArray:

        dtype = COLUMN_DTYPES.get(column_name.lower())

        if dtype is not None:
            return np.asarray(values, dtype=dtype)

        #
        # Automatic inference
        #

        try:
            return np.asarray(values, dtype=np.int32)

        except ValueError:
            pass

        try:
            return np.asarray(values, dtype=np.float64)

        except ValueError:
            pass

        return np.asarray(values, dtype=str)
