"""
topocore.io.ascii.converter
===========================

Converts ASCIIRecordBatch objects into TopoCore Chunk instances.

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

from topocore.io.exceptions import CorruptedFileError
from topocore.pointcloud.attributes import ATTRIBUTE_DTYPES, PointAttribute
from topocore.pointcloud.chunk import Chunk

from .records import ASCIIRecordBatch

#
# Scalar attributes
#

ATTRIBUTE_MAPPING: Final[dict[str, PointAttribute]] = {
    "x": PointAttribute.X,
    "y": PointAttribute.Y,
    "z": PointAttribute.Z,
    "intensity": PointAttribute.INTENSITY,
    "classification": PointAttribute.CLASSIFICATION,
    "return_number": PointAttribute.RETURN_NUMBER,
    "number_of_returns": PointAttribute.NUMBER_OF_RETURNS,
    "scan_angle": PointAttribute.SCAN_ANGLE,
    "gps_time": PointAttribute.GPS_TIME,
    "nir": PointAttribute.NIR,
    "user_data": PointAttribute.USER_DATA,
    "point_source_id": PointAttribute.POINT_SOURCE_ID,
}


def _validate_range(
    values: np.ndarray,
    target_dtype: np.dtype,
    column_name: str,
) -> None:
    """
    Reject values that don't fit in the destination integer dtype.

    Found and fixed in PR19: ASCIIConverter is a standalone
    implementation (like LASConverter) that does NOT go through
    topocore.io.common.base_converter.BasePointConverter -- so the
    range-validation fix already applied there does not protect
    ASCII sources at all. Confirmed directly: an "intensity" column
    value of 70000 (a plausible unit/scale mismatch in a real ASCII
    export) silently became 4464 (``70000 % 65536``) via a plain
    ``.astype()`` cast to the target uint16 dtype, with no error or
    warning. Fixed with the same range check as base_converter.py --
    including rejecting NaN/infinite values for integer-target
    columns, since NumPy's own float->int cast silently turns NaN
    into 0 with only a RuntimeWarning, not an exception.
    """

    if not np.issubdtype(target_dtype, np.integer) or values.size == 0:
        return

    info = np.iinfo(target_dtype)

    if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
        raise CorruptedFileError(
            f"Column '{column_name}' contains NaN or infinite values, which cannot be "
            f"represented in its {target_dtype} integer representation."
        )

    actual_min = values.min()
    actual_max = values.max()

    if actual_min < info.min or actual_max > info.max:
        raise CorruptedFileError(
            f"Column '{column_name}' has values out of range for its {target_dtype} "
            f"representation: expected [{info.min}, {info.max}], got [{actual_min}, {actual_max}]."
        )


class ASCIIConverter:
    """
    Converts ASCIIRecordBatch objects into Chunk instances.
    """

    def __call__(
        self,
        batch: ASCIIRecordBatch,
        *,
        source_id: int = 0,
    ) -> Chunk:

        return self.convert(
            batch,
            source_id=source_id,
        )

    @classmethod
    def convert(
        cls,
        batch: ASCIIRecordBatch,
        *,
        source_id: int = 0,
    ) -> Chunk:
        """
        Convert one ASCIIRecordBatch into a Chunk.
        """

        arrays: dict[
            PointAttribute,
            np.ndarray,
        ] = {}

        #
        # Scalar attributes
        #

        for (
            column_name,
            values,
        ) in batch.columns.items():
            attribute = ATTRIBUTE_MAPPING.get(
                column_name,
            )

            if attribute is None:
                continue

            # Forzar el dtype canónico de TopoCore
            target_dtype = ATTRIBUTE_DTYPES[attribute]

            _validate_range(
                np.asarray(values),
                target_dtype,
                column_name,
            )

            arrays[attribute] = np.asarray(values, dtype=target_dtype)

        #
        # RGB -> COLOR
        #

        cls._build_color(
            batch,
            arrays,
        )

        #
        # NX NY NZ -> NORMAL
        #

        cls._build_normal(
            batch,
            arrays,
        )

        chunk = Chunk(
            size=batch.size,
            attributes=list(arrays.keys()),
            source_id=source_id,
        )

        for (
            attribute,
            values,
        ) in arrays.items():
            chunk[attribute][:] = values

        return chunk

    @staticmethod
    def _build_color(
        batch: ASCIIRecordBatch,
        arrays: dict[
            PointAttribute,
            np.ndarray,
        ],
    ) -> None:
        """
        Assemble RGB columns into COLOR.
        """

        columns = batch.columns

        if not {
            "red",
            "green",
            "blue",
        }.issubset(columns):
            return

        # Asegurar que el stack tenga el dtype correcto (uint16)
        target_dtype = ATTRIBUTE_DTYPES[PointAttribute.COLOR]
        stacked = np.column_stack(
            (
                columns["red"],
                columns["green"],
                columns["blue"],
            )
        )
        _validate_range(stacked, target_dtype, "red/green/blue")
        arrays[PointAttribute.COLOR] = stacked.astype(target_dtype)

    @staticmethod
    def _build_normal(
        batch: ASCIIRecordBatch,
        arrays: dict[
            PointAttribute,
            np.ndarray,
        ],
    ) -> None:
        """
        Assemble NX, NY and NZ into NORMAL.
        """

        columns = batch.columns

        if not {
            "nx",
            "ny",
            "nz",
        }.issubset(columns):
            return

        # Asegurar que el stack tenga el dtype correcto (float32)
        target_dtype = ATTRIBUTE_DTYPES[PointAttribute.NORMAL]
        arrays[PointAttribute.NORMAL] = np.column_stack(
            (
                columns["nx"],
                columns["ny"],
                columns["nz"],
            )
        ).astype(target_dtype)
