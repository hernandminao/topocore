"""
Regression suite for topocore.io.ply.reader.PLYReader,
.header_parser.PLYHeaderParser, and .converter.PLYConverter -- PR19.

Verified the critical endianness handling with REAL binary files
(both little- and big-endian, not mocks) -- a classic, easy-to-get-
wrong pitfall in binary PLY I/O. Also verified ASCII parsing with
small chunk sizes (no data loss/duplication across boundaries),
comments/obj_info handling, header validation (bad magic,
unsupported version, unsupported format string), and combined
color/normal vector attributes. No bugs found in this module.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from topocore.io.ply.exceptions import InvalidPLYError
from topocore.io.ply.reader import PLYReader
from topocore.pointcloud.attributes import PointAttribute


def _write_binary_ply(path: str, endian_char: str, points: list[tuple[float, float, float]]) -> None:
    fmt_name = "binary_little_endian" if endian_char == "<" else "binary_big_endian"
    header = f"ply\nformat {fmt_name} 1.0\nelement vertex {len(points)}\n"
    header += "property float x\nproperty float y\nproperty float z\nend_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.writelines(struct.pack(f"{endian_char}fff", x, y, z) for x, y, z in points)


def _write_ascii_ply(path: str, points: list[tuple[float, float, float]]) -> None:
    header = f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
    header += "property float x\nproperty float y\nproperty float z\nend_header\n"
    with open(path, "w") as f:
        f.write(header)
        f.writelines(f"{x} {y} {z}\n" for x, y, z in points)


# ----------------------------------------------------------------------
# Endianness -- the critical, decisive check.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("endian_char", ["<", ">"])
def test_binary_endianness_parsed_correctly(tmp_path, endian_char: str) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "endian.ply")
    points = [(1.5, 2.5, 3.5), (10.25, -20.5, 30.125), (-1.0, 0.0, 100.5)]
    _write_binary_ply(path, endian_char, points)

    cloud = PLYReader(path).read()
    chunk = next(iter(cloud))

    np.testing.assert_allclose(chunk[PointAttribute.X], [p[0] for p in points])
    np.testing.assert_allclose(chunk[PointAttribute.Y], [p[1] for p in points])
    np.testing.assert_allclose(chunk[PointAttribute.Z], [p[2] for p in points])


# ----------------------------------------------------------------------
# ASCII chunking.
# ----------------------------------------------------------------------


def test_ascii_chunking_no_loss_or_duplication(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "ascii.ply")
    n = 10
    points = [(float(i), float(i * 2), float(i * 3)) for i in range(n)]
    _write_ascii_ply(path, points)

    reader = PLYReader(path, chunk_size=3)
    all_x_list = []
    chunk_count = 0
    for chunk in reader:
        all_x_list.append(chunk[PointAttribute.X])
        chunk_count += 1
    all_x = np.concatenate(all_x_list)

    assert chunk_count == 4  # 3, 3, 3, 1
    assert len(all_x) == n
    np.testing.assert_allclose(np.sort(all_x), np.arange(n, dtype=np.float64))


# ----------------------------------------------------------------------
# Header parsing / validation.
# ----------------------------------------------------------------------


def test_comments_and_obj_info_do_not_break_parsing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "commented.ply"
    path.write_text(
        "ply\nformat ascii 1.0\ncomment test file\nobj_info generated\n"
        "element vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
        "end_header\n1.0 2.0 3.0\n"
    )
    cloud = PLYReader(str(path)).read()
    assert cloud.point_count == 1


def test_rejects_invalid_magic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "bad.ply"
    path.write_text("notply\nformat ascii 1.0\nend_header\n")
    with pytest.raises(InvalidPLYError):
        PLYReader(str(path)).read()


def test_rejects_unsupported_version(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "badversion.ply"
    path.write_text("ply\nformat ascii 2.0\nelement vertex 0\nend_header\n")
    with pytest.raises(InvalidPLYError):
        PLYReader(str(path)).read()


def test_rejects_unsupported_format_string(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "badformat.ply"
    path.write_text("ply\nformat weird_format 1.0\nelement vertex 0\nend_header\n")
    with pytest.raises(InvalidPLYError):
        PLYReader(str(path)).read()


def test_rejects_missing_vertex_element(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "novertex.ply"
    path.write_text("ply\nformat ascii 1.0\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n")
    with pytest.raises(InvalidPLYError):
        PLYReader(str(path)).read()


# ----------------------------------------------------------------------
# Vector attributes: color + normal combined from separate PLY properties.
# ----------------------------------------------------------------------


def test_color_and_normal_combined_correctly(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "full.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "end_header\n"
        "1.0 2.0 3.0 255 128 0 0.0 0.0 1.0\n"
        "4.0 5.0 6.0 0 255 64 1.0 0.0 0.0\n"
    )
    cloud = PLYReader(str(path)).read()
    chunk = next(iter(cloud))

    np.testing.assert_array_equal(chunk[PointAttribute.COLOR], [[255, 128, 0], [0, 255, 64]])
    np.testing.assert_allclose(chunk[PointAttribute.NORMAL], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def test_rejects_nonpositive_chunk_size() -> None:
    with pytest.raises(ValueError):
        PLYReader("dummy.ply", chunk_size=0)
