"""
Input and output package for TopoCore.

This package provides the infrastructure required to read and write
point cloud formats supported by TopoCore.

Current API
-----------
Abstract base classes:

- PointCloudReader
- PointCloudWriter

Exceptions:

- TopoCoreError
- PointCloudIOError
- FileFormatError
- UnsupportedFormatError
- CorruptedFileError
- InvalidHeaderError
- MissingAttributeError
- CoordinateSystemError
- WriteError

Future versions will expose concrete implementations such as
LASReader, LAZReader, E57Reader and their corresponding writers.
"""

from __future__ import annotations

from .base import PointCloudReader as PointCloudReader
from .base import PointCloudWriter as PointCloudWriter
from .exceptions import CoordinateSystemError as CoordinateSystemError
from .exceptions import CorruptedFileError as CorruptedFileError
from .exceptions import FileFormatError as FileFormatError
from .exceptions import InvalidHeaderError as InvalidHeaderError
from .exceptions import MissingAttributeError as MissingAttributeError
from .exceptions import PointCloudIOError as PointCloudIOError
from .exceptions import UnsupportedFormatError as UnsupportedFormatError
from .exceptions import WriteError as WriteError

__all__ = [
    "PointCloudReader",
    "PointCloudWriter",
    "PointCloudIOError",
    "FileFormatError",
    "UnsupportedFormatError",
    "CorruptedFileError",
    "InvalidHeaderError",
    "MissingAttributeError",
    "CoordinateSystemError",
    "WriteError",
]
