"""
topocore.io.e57.reader
======================

Reader for ASTM E57 point cloud files.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pye57  # type: ignore[import-not-found]

from topocore.io.exceptions import PointCloudIOError
from topocore.pointcloud.chunk import Chunk

from .base_reader import BaseE57Reader
from .converter import E57Converter


class E57Reader(BaseE57Reader):
    """
    Reader for ASTM E57 files.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        chunk_size: int,
    ) -> None:
        super().__init__(
            path,
            chunk_size=chunk_size,
        )

    def _open(self) -> None:

        if self._reader is not None:
            return

        try:
            self._reader = pye57.E57(str(self.path))

        except Exception as exc:
            raise PointCloudIOError(f"Unable to open E57 file '{self.path}'.") from exc

    def _iterate_scans(
        self,
    ) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """
        Iterate over every scan in the file.

        Notes
        -----
        Found and fixed in PR19: this previously called
        ``read_scan(scan_index)`` with pye57's own bare defaults
        (``intensity=False, colors=False``) -- meaning intensity and
        RGB color were NEVER read, even from E57 files that
        genuinely contain them, despite ``E57Converter`` already
        being fully prepared to extract both. Confirmed directly
        with a real E57 file (written via pye57 itself, not a mock)
        containing intensity and colorRed/colorGreen/colorBlue: the
        resulting Chunk had only X/Y/Z, silently dropping both.
        Fixed by explicitly requesting ``intensity=True,
        colors=True``. Also added ``ignore_missing_fields=True``,
        since not every real E57 file populates every optional
        field (confirmed directly: pye57's own default raises
        ValueError for a missing, genuinely-optional field like
        ``cartesianInvalidState`` rather than simply omitting it) --
        this correctly makes each field's *presence* in the returned
        dict the source of truth (matching how ``E57Converter``
        already checks ``if name in scan``), instead of the reader
        crashing outright on any file missing an optional field.
        """

        assert self._reader is not None

        for scan_index in range(self._reader.scan_count):
            yield (
                scan_index,
                self._reader.read_scan(
                    scan_index,
                    intensity=True,
                    colors=True,
                    ignore_missing_fields=True,
                ),
            )

    def _create_chunk(
        self,
        arrays: dict[str, np.ndarray],
        source_id: int,
    ) -> Chunk:

        return E57Converter.from_scan(
            arrays,
            source_id=source_id,
        )


__all__ = [
    "E57Reader",
]
