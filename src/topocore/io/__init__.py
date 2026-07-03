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

from .base import PointCloudReader
from .base import PointCloudWriter

from .exceptions import (
    TopoCoreError,
    PointCloudIOError,
    FileFormatError,
    UnsupportedFormatError,
    CorruptedFileError,
    InvalidHeaderError,
    MissingAttributeError,
    CoordinateSystemError,
    WriteError,
)

__all__ = [
    "PointCloudReader",
    "PointCloudWriter",
    "TopoCoreError",
    "PointCloudIOError",
    "FileFormatError",
    "UnsupportedFormatError",
    "CorruptedFileError",
    "InvalidHeaderError",
    "MissingAttributeError",
    "CoordinateSystemError",
    "WriteError",
]