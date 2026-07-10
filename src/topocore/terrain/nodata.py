"""
topocore.terrain.nodata
=======================

NoData utilities.

Provides helper functions for handling missing values in raster terrain
models.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


DEFAULT_NODATA = float("nan")


def is_nodata(
    value: float,
) -> bool:
    """
    Return whether a value represents NoData.

    Parameters
    ----------
    value
        Elevation value.

    Returns
    -------
    bool
    """
    return not math.isfinite(value)


def valid_mask(
    array: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """
    Return a mask of valid cells.

    Parameters
    ----------
    array
        Raster values.

    Returns
    -------
    ndarray
    """
    return np.isfinite(array)


def nodata_mask(
    array: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """
    Return a NoData mask.

    Parameters
    ----------
    array
        Raster values.

    Returns
    -------
    ndarray
    """
    return ~np.isfinite(array)


def valid_count(
    array: NDArray[np.float64],
) -> int:
    """
    Count valid cells.
    """
    return int(
        np.count_nonzero(
            np.isfinite(array),
        )
    )


def nodata_count(
    array: NDArray[np.float64],
) -> int:
    """
    Count NoData cells.
    """
    return int(
        np.count_nonzero(
            ~np.isfinite(array),
        )
    )


def replace_nodata(
    array: NDArray[np.float64],
    value: float,
) -> NDArray[np.float64]:
    """
    Replace NoData values.

    Parameters
    ----------
    array
        Raster values.
    value
        Replacement value.

    Returns
    -------
    ndarray
    """
    result = array.copy()

    result[
        ~np.isfinite(result)
    ] = value

    return result


def fill_nodata(
    array: NDArray[np.float64],
    value: float = 0.0,
) -> NDArray[np.float64]:
    """
    Fill NoData cells.

    Parameters
    ----------
    array
        Raster values.
    value
        Fill value.

    Returns
    -------
    ndarray
    """
    return replace_nodata(
        array,
        value,
    )


__all__ = [
    "DEFAULT_NODATA",
    "is_nodata",
    "valid_mask",
    "nodata_mask",
    "valid_count",
    "nodata_count",
    "replace_nodata",
    "fill_nodata",
]