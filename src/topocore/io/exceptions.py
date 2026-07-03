"""
topocore.io.exceptions
======================

Exceptions raised by the TopoCore I/O subsystem.

These exceptions are shared by every point cloud reader and writer,
independently of the underlying file format (LAS, LAZ, E57, XYZ, PLY,
etc.).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.exceptions import TopoCoreError


class PointCloudIOError(TopoCoreError):
    """
    Base class for point cloud input/output errors.
    """


class FileFormatError(PointCloudIOError):
    """
    Raised when a file does not conform to the expected format.
    """


class UnsupportedFormatError(PointCloudIOError):
    """
    Raised when a file format is not supported.
    """


class CorruptedFileError(PointCloudIOError):
    """
    Raised when a file is corrupted or incomplete.
    """


class InvalidHeaderError(PointCloudIOError):
    """
    Raised when a file header is invalid.
    """


class MissingAttributeError(PointCloudIOError):
    """
    Raised when a required point attribute is missing.
    """


class CoordinateSystemError(PointCloudIOError):
    """
    Raised when the coordinate reference system is invalid.
    """


class WriteError(PointCloudIOError):
    """
    Raised when writing a point cloud fails.
    """


__all__ = [
    "PointCloudIOError",
    "FileFormatError",
    "UnsupportedFormatError",
    "CorruptedFileError",
    "InvalidHeaderError",
    "MissingAttributeError",
    "CoordinateSystemError",
    "WriteError",
]