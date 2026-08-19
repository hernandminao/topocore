"""
Regression suite for topocore.processing.features.pca.PCAFeatures --
PR19.

Verified against pure, deterministic geometric configurations
(perfectly flat plane, perfectly vertical wall, perfectly straight
line) -- each gives an EXACT known ratio (1.0 or 0.0), not just an
approximate/statistical expectation. No bugs found.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import PointDescriptorError
from topocore.processing.features.pca import PCAFeatures


def _grid_cloud(z_fn) -> PointCloud:  # type: ignore[no-untyped-def]
    xs, ys, zs = [], [], []
    for i in range(5):
        for j in range(5):
            xs.append(float(i))
            ys.append(float(j))
            zs.append(z_fn(i, j))
    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


def test_flat_horizontal_plane_is_maximally_planar() -> None:
    cloud = _grid_cloud(lambda i, j: 0.0)
    result = PCAFeatures(k=9).compute_all(cloud)
    center = 12

    assert result["planarity"][center] == pytest.approx(1.0, abs=1e-9)
    assert result["linearity"][center] == pytest.approx(0.0, abs=1e-9)
    assert result["sphericity"][center] == pytest.approx(0.0, abs=1e-9)
    assert result["verticality"][center] == pytest.approx(0.0, abs=1e-9)  # horizontal -> not vertical


def test_vertical_wall_is_maximally_vertical() -> None:
    # Flat plane in the XZ axes (constant Y) -- a vertical wall.
    xs, ys, zs = [], [], []
    for i in range(5):
        for k in range(5):
            xs.append(float(i))
            ys.append(0.0)
            zs.append(float(k))
    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)

    result = PCAFeatures(k=9).compute_all(cloud)
    center = 12

    assert result["planarity"][center] == pytest.approx(1.0, abs=1e-9)
    assert result["verticality"][center] == pytest.approx(1.0, abs=1e-9)


def test_straight_line_is_maximally_linear() -> None:
    xs = [float(i) for i in range(20)]
    ys = [0.0] * 20
    zs = [0.0] * 20
    cloud = PointCloud()
    chunk = Chunk(size=20, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)

    result = PCAFeatures(k=5).compute_all(cloud)
    mid = 10

    assert result["linearity"][mid] == pytest.approx(1.0, abs=1e-9)
    assert result["planarity"][mid] == pytest.approx(0.0, abs=1e-9)
    assert result["sphericity"][mid] == pytest.approx(0.0, abs=1e-9)


def test_eigenvalues_are_descending() -> None:
    cloud = _grid_cloud(lambda i, j: 0.1 * (i + j))  # slightly bumpy, not perfectly degenerate
    result = PCAFeatures(k=9).compute_all(cloud)
    eigvals = result["eigenvalues"]

    assert np.all(eigvals[:, 0] >= eigvals[:, 1])
    assert np.all(eigvals[:, 1] >= eigvals[:, 2])


def test_eigenvalues_never_negative() -> None:
    """
    Explicitly clamped in the source to avoid propagating tiny
    negative eigenvalues from floating-point error.
    """
    cloud = _grid_cloud(lambda i, j: 0.0)
    result = PCAFeatures(k=9).compute_all(cloud)
    assert np.all(result["eigenvalues"] >= 0.0)


def test_individual_accessor_methods_match_compute_all() -> None:
    cloud = _grid_cloud(lambda i, j: 0.0)
    pca = PCAFeatures(k=9)
    full = pca.compute_all(cloud)

    np.testing.assert_array_equal(pca.planarity(cloud), full["planarity"])
    np.testing.assert_array_equal(pca.linearity(cloud), full["linearity"])
    np.testing.assert_array_equal(pca.sphericity(cloud), full["sphericity"])
    np.testing.assert_array_equal(pca.verticality(cloud), full["verticality"])
    np.testing.assert_array_equal(pca.omnivariance(cloud), full["omnivariance"])
    np.testing.assert_array_equal(pca.anisotropy(cloud), full["anisotropy"])
    np.testing.assert_array_equal(pca.surface_variation(cloud), full["surface_variation"])


def test_rejects_k_below_three() -> None:
    with pytest.raises(PointDescriptorError):
        PCAFeatures(k=2)


def test_rejects_empty_cloud() -> None:
    with pytest.raises(PointDescriptorError):
        PCAFeatures(k=3).compute_all(PointCloud())


def test_rejects_cloud_smaller_than_k() -> None:
    cloud = _grid_cloud(lambda i, j: 0.0)  # 25 points
    with pytest.raises(PointDescriptorError):
        PCAFeatures(k=100).compute_all(cloud)
