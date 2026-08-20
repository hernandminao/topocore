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

from abc import ABC, abstractmethod

import numpy as np

from topocore.io.common.attribute_mapping import resolve_attribute
from topocore.io.common.records import PointRecordBatch
from topocore.io.exceptions import CorruptedFileError
from topocore.pointcloud.attributes import ATTRIBUTE_DTYPES, PointAttribute
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
            attribute = self.attribute_mapping.get(name) or resolve_attribute(name)

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
            attribute = self.attribute_mapping.get(source_name) or resolve_attribute(source_name)

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

            source_array = np.asarray(batch[source_name])

            self._validate_range(
                source_array,
                target_dtype,
                attribute,
                source_name,
            )

            chunk[attribute][:] = source_array.astype(target_dtype)

            processed.add(source_name)

        self._populate_special_attributes(
            chunk,
            batch,
        )

    @staticmethod
    def _validate_range(
        source_array: np.ndarray,
        target_dtype: np.dtype,
        attribute: PointAttribute,
        source_name: str,
    ) -> None:
        """
        Reject values that don't fit in the destination integer dtype.

        Found and fixed in PR19: converting a source array to a
        narrower integer dtype (e.g. ``int32`` -> ``uint16``, the
        exact case for "intensity") via a plain ``.astype()`` call
        does NOT raise -- NumPy silently wraps out-of-range integers
        modulo the target type's range. Confirmed directly: a value
        of 70000 read from an ASCII or PLY source silently became
        4464 (``70000 % 65536``) in the resulting Chunk, with no
        error or warning anywhere, for ANY format going through this
        shared converter (confirmed reproducible via both PLY and
        ASCII sources, since both route through this same method).
        Fixed by validating the actual value range against the
        target dtype's representable range BEFORE casting, raising a
        clear, domain-specific error instead of silently corrupting
        the data.

        Also rejects NaN/infinite values in a float source destined
        for an integer attribute: confirmed separately that NumPy's
        own float->int cast turns NaN into 0 with only a
        ``RuntimeWarning`` (not an exception) -- easy to miss, and
        the same "corrupt data must fail loud" principle applies,
        since a NaN in a column like "intensity" or "classification"
        (unlike elevation, which legitimately uses NaN for NoData)
        indicates a genuinely malformed source file.
        """

        if not np.issubdtype(target_dtype, np.integer) or source_array.size == 0:
            return

        info = np.iinfo(target_dtype)

        if np.issubdtype(source_array.dtype, np.floating) and not np.isfinite(source_array).all():
            raise CorruptedFileError(
                f"Attribute '{attribute.value}' (source column '{source_name}') contains NaN "
                f"or infinite values, which cannot be represented in its {target_dtype} integer "
                "representation."
            )

        actual_min = source_array.min()
        actual_max = source_array.max()

        if actual_min < info.min or actual_max > info.max:
            raise CorruptedFileError(
                f"Attribute '{attribute.value}' (source column '{source_name}') has values "
                f"out of range for its {target_dtype} representation: expected "
                f"[{info.min}, {info.max}], got [{actual_min}, {actual_max}]."
            )

    @abstractmethod
    def _populate_special_attributes(
        self,
        chunk: Chunk,
        batch: PointRecordBatch,
    ) -> None:
        """
        Hook for vector attributes such as COLOR or NORMAL.

        Concrete converters may override this method.
        """
