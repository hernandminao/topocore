"""
topocore.geodesy.validation
===========================

Validation utilities for geodesy inputs.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from topocore.geodesy.exceptions import ValidationError

ArrayLike: TypeAlias = Sequence[float] | NDArray[np.floating]


def _validate_finite(value: float, name: str) -> None:
    """
    Validate that a numeric value is finite.

    Parameters
    ----------
    value
        Value to validate.
    name
        Parameter name.

    Raises
    ------
    ValidationError
        If the value is not finite.
    """
    if value is None:
        raise ValidationError(f"{name} cannot be None.")

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be numeric.") from exc

    if not math.isfinite(numeric):
        raise ValidationError(f"{name} must be finite.")


def validate_lat_lon(latitude: float, longitude: float) -> None:
    """
    Validate latitude and longitude.

    Parameters
    ----------
    latitude
        Latitude in degrees.
    longitude
        Longitude in degrees.

    Raises
    ------
    ValidationError
        If the coordinates are invalid.
    """
    _validate_finite(latitude, "Latitude")
    _validate_finite(longitude, "Longitude")

    if not (-90.0 <= latitude <= 90.0):
        raise ValidationError(
            f"Latitude {latitude} out of bounds [-90, 90]."
        )

    if not (-180.0 <= longitude <= 180.0):
        raise ValidationError(
            f"Longitude {longitude} out of bounds [-180, 180]."
        )


def validate_epsg(epsg: int) -> None:
    """
    Validate an EPSG code.

    Parameters
    ----------
    epsg
        EPSG identifier.

    Raises
    ------
    ValidationError
        If the EPSG code is invalid.
    """
    if isinstance(epsg, bool):
        raise ValidationError(
            "EPSG code must be an integer, not bool."
        )

    if not isinstance(epsg, int):
        raise ValidationError(
            f"Invalid EPSG code: {epsg!r}."
        )

    if epsg <= 0:
        raise ValidationError(
            f"EPSG code must be positive, got {epsg}."
        )


def validate_bbox(
    bbox: tuple[float, float, float, float],
) -> None:
    """
    Validate a bounding box.

    Parameters
    ----------
    bbox
        Bounding box as (minx, miny, maxx, maxy).

    Raises
    ------
    ValidationError
        If the bbox is invalid.
    """
    if len(bbox) != 4:
        raise ValidationError(
            "BBox must contain exactly 4 values "
            "(minx, miny, maxx, maxy)."
        )

    minx, miny, maxx, maxy = bbox

    _validate_finite(minx, "minx")
    _validate_finite(miny, "miny")
    _validate_finite(maxx, "maxx")
    _validate_finite(maxy, "maxy")

    if minx > maxx:
        raise ValidationError(
            f"BBox minx ({minx}) cannot be greater than maxx ({maxx})."
        )

    if miny > maxy:
        raise ValidationError(
            f"BBox miny ({miny}) cannot be greater than maxy ({maxy})."
        )


def validate_array(
    arr: ArrayLike,
    dims: int,
) -> NDArray[np.float64]:
    """
    Validate and normalize a coordinate array.

    Parameters
    ----------
    arr
        Coordinate array.
    dims
        Expected number of columns.

    Returns
    -------
    numpy.ndarray
        Normalized float64 array.

    Raises
    ------
    ValidationError
        If the array shape is invalid.
    """
    array = np.asarray(arr, dtype=np.float64)

    if array.ndim != 2:
        raise ValidationError(
            f"Expected a 2D array, got {array.ndim}D."
        )

    if array.shape[1] != dims:
        raise ValidationError(
            f"Expected shape (N, {dims}), got {array.shape}."
        )

    if not np.isfinite(array).all():
        raise ValidationError(
            "Coordinate array contains NaN or infinite values."
        )

    return array


def validate_coordinate_arrays(
    *arrays: ArrayLike,
) -> tuple[NDArray[np.float64], ...]:
    """
    Validate multiple coordinate arrays.

    All arrays must:

    - have identical length;
    - contain only finite values.

    Parameters
    ----------
    arrays
        Coordinate arrays.

    Returns
    -------
    tuple[numpy.ndarray, ...]

    Raises
    ------
    ValidationError
        If validation fails.
    """
    normalized = tuple(
        np.asarray(arr, dtype=np.float64)
        for arr in arrays
    )

    if not normalized:
        raise ValidationError("No coordinate arrays provided.")

    expected = normalized[0].shape

    for arr in normalized:
        if arr.ndim != 1:
            raise ValidationError(
                "Coordinate arrays must be one-dimensional."
            )

        if arr.shape != expected:
            raise ValidationError(
                "Coordinate arrays must have identical length."
            )

        if not np.isfinite(arr).all():
            raise ValidationError(
                "Coordinate arrays contain NaN or infinite values."
            )

    return normalized


__all__ = [
    "ArrayLike",
    "validate_lat_lon",
    "validate_epsg",
    "validate_bbox",
    "validate_array",
    "validate_coordinate_arrays",
]