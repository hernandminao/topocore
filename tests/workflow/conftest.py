"""Shared fixtures for topocore.workflow tests -- PR19."""

from __future__ import annotations

import pytest


@pytest.fixture
def xyz_file(tmp_path: object) -> str:
    """A small but triangulatable XYZ point cloud (5 points, non-collinear)."""
    path = str(tmp_path) + "/points.xyz"  # type: ignore[operator]
    with open(path, "w") as f:
        f.write("0.0 0.0 1.0\n10.0 0.0 2.0\n0.0 10.0 3.0\n10.0 10.0 4.0\n5.0 5.0 2.5\n")
    return path


@pytest.fixture
def csv_file(tmp_path: object) -> str:
    path = str(tmp_path) + "/points.csv"  # type: ignore[operator]
    with open(path, "w") as f:
        f.write("x,y,z\n0.0,0.0,1.0\n10.0,0.0,2.0\n0.0,10.0,3.0\n10.0,10.0,4.0\n5.0,5.0,2.5\n")
    return path


@pytest.fixture
def las_file(tmp_path: object) -> str:
    import laspy  # type: ignore[import-untyped]
    import numpy as np

    path = str(tmp_path) + "/points.las"  # type: ignore[operator]
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(header)
    las.x = np.array([0.0, 10.0, 0.0, 10.0, 5.0])
    las.y = np.array([0.0, 0.0, 10.0, 10.0, 5.0])
    las.z = np.array([1.0, 2.0, 3.0, 4.0, 2.5])
    las.write(path)
    return path
