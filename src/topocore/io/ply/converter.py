"""
topocore.io.ply.converter
=========================

PLY to Chunk converter.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import numpy as np

from topocore.io.common.base_converter import BasePointConverter
from topocore.io.common.records import PointRecordBatch
from topocore.pointcloud.attributes import ATTRIBUTE_DTYPES
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk


class PLYConverter(BasePointConverter):
    """
    Converts PLY PointRecordBatch objects into TopoCore Chunk objects.

    Generic scalar attributes (X, Y, Z, intensity, classification,
    GPS time, etc.) are handled automatically by BasePointConverter.

    This converter only handles vector attributes that require
    combining multiple PLY properties.
    """

    @property
    def attribute_mapping(
        self,
    ) -> dict[str, PointAttribute]:
        """
        PLY-specific attribute mapping.

        Scalar attributes are resolved by the common attribute mapping.
        Only vector attributes are declared here.
        """

        return {

            # RGB color

            "red": PointAttribute.COLOR,
            "green": PointAttribute.COLOR,
            "blue": PointAttribute.COLOR,

            # Surface normals

            "nx": PointAttribute.NORMAL,
            "ny": PointAttribute.NORMAL,
            "nz": PointAttribute.NORMAL,

        }

    def _populate_special_attributes(
        self,
        chunk: Chunk,
        batch: PointRecordBatch,
    ) -> None:
        """
        Populate vector attributes.
        """

        self._populate_color(
            chunk,
            batch,
        )

        self._populate_normals(
            chunk,
            batch,
        )

    def _populate_color(
        self,
        chunk: Chunk,
        batch: PointRecordBatch,
    ) -> None:
        """
        Populate the COLOR attribute from red, green and blue arrays.
        """

        if not all(
            name in batch
            for name in (
                "red",
                "green",
                "blue",
            )
        ):
            return

        if not chunk.has_attribute(
            PointAttribute.COLOR,
        ):
            return

        rgb = np.column_stack(
            (
                batch["red"],
                batch["green"],
                batch["blue"],
            )
        ).astype(
            ATTRIBUTE_DTYPES[
                PointAttribute.COLOR
            ],
            copy=False,
        )

        chunk[
            PointAttribute.COLOR
        ][:] = rgb

    def _populate_normals(
        self,
        chunk: Chunk,
        batch: PointRecordBatch,
    ) -> None:
        """
        Populate the NORMAL attribute from nx, ny and nz arrays.
        """

        if not all(
            name in batch
            for name in (
                "nx",
                "ny",
                "nz",
            )
        ):
            return

        if not chunk.has_attribute(
            PointAttribute.NORMAL,
        ):
            return

        normals = np.column_stack(
            (
                batch["nx"],
                batch["ny"],
                batch["nz"],
            )
        ).astype(
            ATTRIBUTE_DTYPES[
                PointAttribute.NORMAL
            ],
            copy=False,
        )

        chunk[
            PointAttribute.NORMAL
        ][:] = normals


__all__ = [
    "PLYConverter",
]