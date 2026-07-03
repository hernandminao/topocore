"""
topocore.io.common.base_converter
=================================

Base converter used by all point cloud readers.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np

from topocore.io.common.attribute_mapping import resolve_attribute
from topocore.io.common.records import PointRecordBatch
from topocore.pointcloud.attributes import ATTRIBUTE_DTYPES
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk


class BasePointConverter(ABC):
    """
    Base class for point cloud converters.

    Concrete converters only need to provide mappings that are
    specific to their format. Canonical attribute names are resolved
    automatically through the common attribute mapping.
    """

    @property
    @abstractmethod
    def attribute_mapping(
        self,
    ) -> dict[str, PointAttribute]:
        """
        Mapping between source property names and PointAttribute.

        This mapping should only contain format-specific aliases.
        Generic names (x, y, z, intensity, classification, etc.)
        are resolved automatically by the common resolver.
        """
        raise NotImplementedError

    def convert(
        self,
        batch: PointRecordBatch,
    ) -> Chunk:
        """
        Convert a PointRecordBatch into a Chunk.
        """

        attributes = self._collect_attributes(batch)

        chunk = Chunk(
            size=batch.size,
            attributes=attributes,
            source_id=batch.source_id,
        )

        self._populate_chunk(
            chunk,
            batch,
        )

        return chunk

    def _collect_attributes(
        self,
        batch: PointRecordBatch,
    ) -> list[PointAttribute]:
        """
        Determine the Chunk attributes present in the batch.
        """

        result: list[PointAttribute] = []

        for name in batch:

            attribute = (
                self.attribute_mapping.get(name)
                or resolve_attribute(name)
            )

            if attribute is None:
                continue

            if attribute not in result:
                result.append(attribute)

        return result

    def _populate_chunk(
        self,
        chunk: Chunk,
        batch: PointRecordBatch,
    ) -> None:
        """
        Copy scalar attributes into the destination Chunk.
        """

        processed: set[str] = set()

        for source_name in batch:

            attribute = (
                self.attribute_mapping.get(source_name)
                or resolve_attribute(source_name)
            )

            if attribute is None:
                continue

            if attribute in (
                PointAttribute.COLOR,
                PointAttribute.NORMAL,
            ):
                continue

            if not chunk.has_attribute(attribute):
                continue

            target_dtype = ATTRIBUTE_DTYPES[attribute]

            chunk[attribute][:] = np.asarray(
                batch[source_name],
                dtype=target_dtype,
            )

            processed.add(source_name)

        self._populate_special_attributes(
            chunk,
            batch,
        )

    def _populate_special_attributes(
        self,
        chunk: Chunk,
        batch: PointRecordBatch,
    ) -> None:
        """
        Hook for vector attributes such as COLOR or NORMAL.

        Concrete converters may override this method.
        """

        pass