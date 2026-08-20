"""
Regression suite for topocore.io.ascii CSV/XYZ/PTS readers -- PR19.

Verified end-to-end with real files: header-based column mapping
(including reordered columns), comment-line skipping, chunked
reading with no data loss, and PTS's leading point-count line being
correctly skipped. No bugs found in this part of the module.
"""

from __future__ import annotations

import numpy as np

from topocore.io.ascii.csv.reader import CSVReader
from topocore.io.ascii.pts.reader import PTSReader
from topocore.io.ascii.xyz.reader import XYZReader
from topocore.pointcloud.attributes import PointAttribute


def test_csv_with_header(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "test.csv"
    path.write_text("x,y,z,intensity\n1.0,2.0,3.0,100\n4.0,5.0,6.0,200\n")

    cloud = CSVReader(str(path)).read()
    chunk = next(iter(cloud))

    np.testing.assert_allclose(chunk[PointAttribute.X], [1.0, 4.0])
    np.testing.assert_array_equal(chunk[PointAttribute.INTENSITY], [100, 200])


def test_csv_columns_mapped_by_header_not_position(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "reorder.csv"
    path.write_text("z,intensity,x,y\n30.0,999,10.0,20.0\n")

    cloud = CSVReader(str(path)).read()
    chunk = next(iter(cloud))

    assert chunk[PointAttribute.X][0] == 10.0
    assert chunk[PointAttribute.Y][0] == 20.0
    assert chunk[PointAttribute.Z][0] == 30.0


def test_xyz_without_header(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "test.xyz"
    path.write_text("10.5 20.5 30.5\n11.5 21.5 31.5\n")

    cloud = XYZReader(str(path)).read()
    chunk = next(iter(cloud))

    np.testing.assert_allclose(chunk[PointAttribute.X], [10.5, 11.5])


def test_xyz_skips_comment_lines(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "comments.xyz"
    path.write_text("# comment\n1.0 2.0 3.0\n# another\n4.0 5.0 6.0\n")

    cloud = XYZReader(str(path)).read()
    assert cloud.point_count == 2


def test_xyz_chunking_no_data_loss(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "chunks.xyz"
    n = 25
    with path.open("w") as f:
        for i in range(n):
            f.write(f"{i}.0 {i * 2}.0 {i * 3}.0\n")

    reader = XYZReader(str(path), chunk_size=10)
    all_x_list = []
    chunk_count = 0
    for chunk in reader:
        all_x_list.append(chunk[PointAttribute.X])
        chunk_count += 1
    all_x = np.concatenate(all_x_list)

    assert chunk_count == 3  # 10, 10, 5
    assert len(all_x) == n


def test_pts_skips_leading_point_count_line(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "test.pts"
    path.write_text("2\n100.0 200.0 300.0\n101.0 201.0 301.0\n")

    cloud = PTSReader(str(path)).read()
    chunk = next(iter(cloud))

    np.testing.assert_allclose(chunk[PointAttribute.X], [100.0, 101.0])
    assert cloud.point_count == 2  # the count line itself was NOT read as a point
