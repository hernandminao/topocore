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
        arrays[
            PointAttribute.COLOR
        ] = np.column_stack(
            (
                columns["red"],
                columns["green"],
                columns["blue"],
            )
        ).astype(target_dtype)

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
        arrays[
            PointAttribute.NORMAL
        ] = np.column_stack(
            (
                columns["nx"],
                columns["ny"],
                columns["nz"],
            )
        ).astype(target_dtype)