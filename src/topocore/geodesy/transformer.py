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
from topocore.geodesy.operation import CoordinateOperation
from topocore.geodesy.operation_type import OperationType
from topocore.geodesy.validation import (
    validate_bbox,
    validate_coordinate_arrays,
)

from . import _cache


class CoordinateTransformer:
    """
    Transforms coordinates from a source CRS to a target CRS.

    `from_operation()` scope (PR18A.4) -- read before extending
    -----------------------------------------------------------
    This alternate constructor intentionally supports only the
    subset of `CoordinateOperation` that can be executed with the
    information a `CoordinateOperation` itself carries -- nothing
    more is silently assumed or approximated.

    Supported:
        - IDENTITY
        - HELMERT, static (7-parameter) form, geographic source AND
          target CRS. Verified numerically against EPSG formula 9606
          (position_vector convention) and against hand-derivable
          geometry (a pure rz rotation and a pure tx translation each
          checked against their expected lon/lat/height signature at
          (lon=0, lat=0), where geocentric X is the local vertical
          and geocentric Y is the local east/longitude direction).

    Explicitly UNsupported -- boundaries, not missing implementations:
        - HELMERT, time-dependent (14-parameter) form. Applying the
          rate terms correctly needs an observation epoch for the
          DATA being transformed (distinct from Helmert's own
          `reference_epoch`), which `transform_point()`/
          `transform_array()` don't accept as a parameter today.
          There is no way to execute this correctly without that
          missing piece of information -- adding it is real,
          additional API surface for a future PR.
        - HELMERT between projected CRS. The verified pipeline goes
          through geographic-to-geocentric conversion; projected
          coordinates would need an additional projection/
          deprojection step this version doesn't build.
        - GRID_SHIFT. `GridShift` itself (see `grid_shift.py`)
          doesn't load or parse a grid file yet -- there is nothing
          to execute against.

    None of the three above are TODOs quietly waiting to be filled
    in casually: each is blocked by a specific, named piece of
    information or capability TopoCore doesn't have yet. See
    `test_transformer_from_operation.py` for one regression test per
    row of this table.
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
            raise TransformationError("Failed to create coordinate transformer.") from exc

    @classmethod
    def from_operation(cls, operation: CoordinateOperation) -> CoordinateTransformer:
        """
        Build a `CoordinateTransformer` from a `CoordinateOperation`'s
        described parameters, rather than a plain CRS-to-CRS pyproj
        lookup. `CoordinateTransformer(source_crs, target_crs)` is
        never replaced -- this is an alternate constructor, per the
        frozen PR18A contract.

        Scope in this version
        ----------------------
        - `IDENTITY`: delegates directly to the existing constructor.
        - `HELMERT`: only for geographic source AND target CRS, and
          only the static (7-parameter) form. A time-dependent
          (14-parameter) transform needs an observation epoch for
          the DATA being transformed (not just Helmert's own
          `reference_epoch`) to apply the rate terms correctly --
          `transform_point()`/`transform_array()` don't accept one
          today. That's real, additional API surface for a future
          PR, not something to guess at here.
        - `GRID_SHIFT`: not implemented -- `GridShift` itself
          explicitly defers loading/applying a grid file to a later
          PR (see `grid_shift.py`); this can't apply what doesn't
          exist yet.

        Raises
        ------
        TransformationError
            If `operation.operation_type` is `GRID_SHIFT`, is a
            time-dependent `HELMERT`, or either CRS isn't geographic
            for a `HELMERT` operation.
        """
        if operation.operation_type is OperationType.IDENTITY:
            return cls(operation.source_crs, operation.target_crs)

        if operation.operation_type is OperationType.GRID_SHIFT:
            raise TransformationError(
                "CoordinateTransformer.from_operation(): GRID_SHIFT is not yet "
                "implemented -- GridShift itself doesn't load or apply grid "
                "files yet (see topocore.geodesy.grid_shift)."
            )

        # HELMERT -- CoordinateOperation.__post_init__ already guarantees
        # operation.helmert is not None whenever operation_type is HELMERT.
        helmert = operation.helmert
        assert helmert is not None

        if helmert.is_time_dependent:
            raise TransformationError(
                "CoordinateTransformer.from_operation(): time-dependent "
                "(14-parameter) Helmert transformations are not yet "
                "supported -- applying the rate terms correctly requires an "
                "observation epoch for the data being transformed, which "
                "transform_point()/transform_array() don't accept today. "
                "Use a static (7-parameter) HelmertParameters instead."
            )

        if not (operation.source_crs.is_geographic and operation.target_crs.is_geographic):
            raise TransformationError(
                "CoordinateTransformer.from_operation(): HELMERT is only "
                "supported between geographic CRS in this version -- got "
                f"source_crs.is_geographic={operation.source_crs.is_geographic}, "
                f"target_crs.is_geographic={operation.target_crs.is_geographic}."
            )

        source_ellipsoid = operation.source_crs.ellipsoid
        target_ellipsoid = operation.target_crs.ellipsoid
        if source_ellipsoid is None or target_ellipsoid is None:
            raise TransformationError(
                "CoordinateTransformer.from_operation(): source_crs or target_crs has no ellipsoid."
            )

        # Geographic (deg) -> radians -> geocentric (cartesian) -> Helmert
        # (position_vector convention, matching EPSG's documented formula
        # 9606 -- verified against it numerically) -> inverse geocentric ->
        # radians -> geographic (deg). Ellipsoid parameters come straight
        # from CRS.ellipsoid (already real domain objects), never from a
        # PROJ "+ellps=<name>" lookup, avoiding any name-matching ambiguity.
        pipeline = (
            "+proj=pipeline "
            "+step +proj=unitconvert +xy_in=deg +xy_out=rad "
            f"+step +proj=cart +a={source_ellipsoid.semi_major_axis} "
            f"+rf={source_ellipsoid.inverse_flattening} "
            f"+step +proj=helmert +x={helmert.tx} +y={helmert.ty} +z={helmert.tz} "
            f"+rx={helmert.rx} +ry={helmert.ry} +rz={helmert.rz} "
            f"+s={helmert.scale} +convention=position_vector "
            f"+step +inv +proj=cart +a={target_ellipsoid.semi_major_axis} "
            f"+rf={target_ellipsoid.inverse_flattening} "
            "+step +proj=unitconvert +xy_in=rad +xy_out=deg"
        )

        try:
            pyproj_transformer = PyprojTransformer.from_pipeline(pipeline)
        except Exception as exc:
            raise TransformationError("Failed to build Helmert transformation pipeline.") from exc

        instance = cls.__new__(cls)
        instance._source_crs = operation.source_crs
        instance._target_crs = operation.target_crs
        instance._transformer = pyproj_transformer
        return instance

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
            raise TransformationError("Point transformation failed.") from exc

    def transform_array(
        self,
        x: Sequence[float] | np.ndarray,
        y: Sequence[float] | np.ndarray,
        z: Sequence[float] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Transform arrays of coordinates.
        """
        if z is None:
            x_arr, y_arr = validate_coordinate_arrays(
                x,
                y,
            )
        else:
            x_arr, y_arr, z_arr = validate_coordinate_arrays(
                x,
                y,
                z,
            )

        try:
            if z is None:
                x_new, y_new = self._transformer.transform(
                    x_arr,
                    y_arr,
                )

                return (
                    np.asarray(x_new, dtype=np.float64),
                    np.asarray(y_new, dtype=np.float64),
                    None,
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
            raise TransformationError("Array transformation failed.") from exc

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
            raise TransformationError("BBox transformation failed.") from exc


__all__ = [
    "CoordinateTransformer",
]
