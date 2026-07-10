from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from topocore.io.ply.enums import PLYFormat


def build_header_stream(text: str) -> io.BytesIO:
    """Return a binary stream containing a PLY header snippet."""
    return io.BytesIO(text.encode("utf-8"))


class FakeScalarProperty:
    """Minimal scalar property stub for reader unit tests."""

    def __init__(
        self,
        name: str,
        *,
        numpy_dtype: np.dtype | None = None,
    ) -> None:
        self.name = name
        dtype = numpy_dtype or np.dtype("float32")
        self.dtype = type(
            "DTypeWrapper",
            (),
            {"numpy_dtype": dtype},
        )()


class FakeListProperty:
    """Minimal list property stub (no scalar dtype)."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeVertex:
    """Vertex element stub for reader unit tests."""

    def __init__(
        self,
        count: int = 3,
        *,
        include_list_property: bool = False,
    ) -> None:
        self.name = "vertex"
        self.count = count
        self.properties: list[FakeScalarProperty | FakeListProperty] = [
            FakeScalarProperty("x"),
            FakeScalarProperty("y"),
            FakeScalarProperty("z"),
        ]
        if include_list_property:
            self.properties.append(FakeListProperty("indices"))


class FakeHeader:
    """Header stub for reader unit tests."""

    def __init__(
        self,
        fmt: PLYFormat = PLYFormat.ASCII,
        *,
        vertex_count: int = 3,
        include_list_property: bool = False,
    ) -> None:
        self.format = fmt
        self.elements = [
            FakeVertex(
                vertex_count,
                include_list_property=include_list_property,
            )
        ]


def write_ascii_ply(
    path: Path,
    rows: list[tuple[float, float, float]],
) -> None:
    """Write a minimal ASCII PLY file with x/y/z float properties."""
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(rows)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    body = "".join(f"{x} {y} {z}\n" for x, y, z in rows)
    path.write_text(header + body, encoding="utf-8")


def write_binary_ply(
    path: Path,
    rows: list[tuple[float, float, float]],
    *,
    fmt: PLYFormat,
) -> None:
    """Write a minimal binary PLY file (little or big endian)."""
    if fmt == PLYFormat.BINARY_LITTLE_ENDIAN:
        format_line = "binary_little_endian"
        endian = "<"
    elif fmt == PLYFormat.BINARY_BIG_ENDIAN:
        format_line = "binary_big_endian"
        endian = ">"
    else:
        msg = "fmt must be a binary PLY format"
        raise ValueError(msg)

    header = (
        "ply\n"
        f"format {format_line} 1.0\n"
        f"element vertex {len(rows)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    dtype = np.dtype(
        [
            ("x", f"{endian}f4"),
            ("y", f"{endian}f4"),
            ("z", f"{endian}f4"),
        ]
    )
    data = np.array(rows, dtype=dtype)
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        data.tofile(stream)
