"""
topocore.io.ascii.format
========================

ASCII format configuration.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ASCIIFormat:
    """
    Configuration describing an ASCII point cloud format.

    Parameters
    ----------
    delimiter
        Column separator. If ``None``, it is automatically detected.
    has_header
        Indicates whether the file contains a header row.

        * ``True``  -> first row is always a header.
        * ``False`` -> first row is always data.
        * ``None``  -> automatically detect.
    encoding
        File encoding.
    comment_prefixes
        Prefixes identifying comment lines.
    x_column
        Default X column index when no header exists.
    y_column
        Default Y column index when no header exists.
    z_column
        Default Z column index when no header exists.
    """

    delimiter: str | None = None

    has_header: bool | None = None

    encoding: str = "utf-8"

    comment_prefixes: tuple[str, ...] = (
        "#",
        "//",
    )

    x_column: int = 0

    y_column: int = 1

    z_column: int = 2