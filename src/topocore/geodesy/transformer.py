"""
topocore.geodesy.transformer
============================

Coordinate transformation utilities.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pyproj import Transformer as PyprojTransformer

from topocore.geodesy.crs import CRS
from topocore.geodesy.exceptions import TransformationError
from topocore.geodesy.validation import (
    validate_bbox,
    validate_coordinate_arrays,
)
from . import _cache


class CoordinateTransformer:
    """
    Transforms coordinates from a source CRS to a target CRS.
    """

    __slots__ = (
        "_source_crs",
        "_target_crs",
        "_transformer",
    )

    def __init__(self, source_crs: CRS, target_crs: CRS) -> None:
        self._source_crs = source_crs
        self._target_crs = target_crs

        src_epsg = source_crs.epsg
        tgt_epsg = target_crs.epsg

        try:
            if src_epsg is not None and tgt_epsg is not None:
                self._transformer = _cache.get_transformer(
                    src_epsg,
                    tgt_epsg,
                )
            else:
                self._transformer = PyprojTransformer.from_crs(
                    source_crs._native,
                    target_crs._native,
                    always_xy=True,
                )
        except Exception as exc:
            raise TransformationError(
                "Failed to create coordinate transformer."
            ) from exc

    @property
    def source_crs(self) -> CRS:
        """Return the source CRS."""
        return self._source_crs

    @property
    def target_crs(self) -> CRS:
        """Return the target CRS."""
        return self._target_crs

    def transform_point(
        self,
        x: float,
        y: float,
        z: float | None = None,
    ) -> tuple[float, float, float | None]:
        """
        Transform a single point.

        Parameters
        ----------
        x
            X coordinate (longitude/easting).
        y
            Y coordinate (latitude/northing).
        z
            Optional height.

        Returns
        -------
        tuple
            (x, y, z) in the destination CRS.
        """
        try:
            if z is None:
                x_new, y_new = self._transformer.transform(x, y)
                return (
                    float(x_new),
                    float(y_new),
                    None,
                )

            x_new, y_new, z_new = self._transformer.transform(
                x,
                y,
                z,
            )

            return (
                float(x_new),
                float(y_new),
                float(z_new),
            )

        except Exception as exc:
            raise TransformationError(
                "Point transformation failed."
            ) from exc

    def transform_array(
        self,
        x: Sequence[float],
        y: Sequence[float],
        z: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Transform arrays of coordinates.
        """
        try:

            if z is None:
                
                x_arr, y_arr = validate_coordinate_arrays(
		    x,
		    y,
		)

                x_new, y_new = self._transformer.transform(
                    x_arr,
                    y_arr,
                )

                return (
                    np.asarray(x_new, dtype=np.float64),
                    np.asarray(y_new, dtype=np.float64),
                    None,
                )

            x_arr, y_arr, z_arr = validate_coordinate_arrays(
	        x,
	        y,
	        z,
            )

            x_new, y_new, z_new = self._transformer.transform(
                x_arr,
                y_arr,
                z_arr,
            )

            return (
                np.asarray(x_new, dtype=np.float64),
                np.asarray(y_new, dtype=np.float64),
                np.asarray(z_new, dtype=np.float64),
            )

        except Exception as exc:
            raise TransformationError(
                "Array transformation failed."
            ) from exc

    def transform_bbox(
        self,
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """
        Transform a bounding box.

        Parameters
        ----------
        bbox
            Bounding box in the form
            (minx, miny, maxx, maxy).

        Returns
        -------
        tuple
            Transformed bounding box
            (minx, miny, maxx, maxy).
        """
        validate_bbox(bbox)

        try:
            minx, miny, maxx, maxy = bbox

            left, bottom, _ = self.transform_point(
                minx,
                miny,
            )
            right, top, _ = self.transform_point(
                maxx,
                maxy,
            )

            return (
                min(left, right),
                min(bottom, top),
                max(left, right),
                max(bottom, top),
            )

        except Exception as exc:
            raise TransformationError(
                "BBox transformation failed."
            ) from exc


__all__ = [
    "CoordinateTransformer",
] 