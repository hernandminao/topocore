"""
Regression suite for PCA-VIEWPOINT-001 (fixed in this PR).

Bug: PCANormalEstimator/WeightedPCANormalEstimator's `viewpoint`
constructor argument is typed as Vector3D (NDArray[np.float64]), but
was stored as-is with no validation. A plain tuple or list -- a
reasonable mistake given the type hint alone -- did not fail at
construction; it failed later, deep inside _orient_normals(), with a
confusing TypeError instead of this class's own NormalError.

Fix: a shared validate_viewpoint() helper (normals/base.py) now
checks, at construction time, that a non-None viewpoint is a NumPy
array with shape (3,) and a numeric dtype, raising NormalError with
a clear message otherwise. Applied identically to both
PCANormalEstimator and WeightedPCANormalEstimator.

A valid ndarray continues to work exactly as before -- confirmed the
resulting behavior for a valid viewpoint is unchanged (same object,
same orientation results).
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import NormalError
from topocore.processing.normals.pca import PCANormalEstimator
from topocore.processing.normals.weighted_pca import WeightedPCANormalEstimator


@pytest.mark.parametrize("estimator_class", [PCANormalEstimator, WeightedPCANormalEstimator])
@pytest.mark.parametrize(
    ("bad_viewpoint", "expected_message"),
    [
        ((1.0, 2.0, 3.0), "must be a numpy array"),
        ([1.0, 2.0, 3.0], "must be a numpy array"),
        (np.array([1.0, 2.0]), "must have shape"),
        (np.array(["a", "b", "c"]), "must have a numeric dtype"),
    ],
)
def test_invalid_viewpoint_raises_normal_error(
    estimator_class: type[PCANormalEstimator | WeightedPCANormalEstimator],
    bad_viewpoint: object,
    expected_message: str,
) -> None:
    """Previously a tuple/list raised a confusing TypeError deep inside _orient_normals(); now it's a clear NormalError at construction."""
    with pytest.raises(NormalError, match=expected_message):
        estimator_class(k=5, viewpoint=bad_viewpoint)  # type: ignore[arg-type]


@pytest.mark.parametrize("estimator_class", [PCANormalEstimator, WeightedPCANormalEstimator])
def test_valid_viewpoint_still_works_unchanged(
    estimator_class: type[PCANormalEstimator | WeightedPCANormalEstimator],
) -> None:
    rng = np.random.default_rng(0)
    n = 30
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = np.zeros(n)
    cloud.add_chunk(chunk)

    viewpoint = np.array([5.0, 5.0, -100.0])
    normals = estimator_class(k=5, viewpoint=viewpoint).estimate(cloud)

    assert (normals[:, 2] <= 0).all()


@pytest.mark.parametrize("estimator_class", [PCANormalEstimator, WeightedPCANormalEstimator])
def test_none_viewpoint_still_accepted(
    estimator_class: type[PCANormalEstimator | WeightedPCANormalEstimator],
) -> None:
    estimator_class(k=5, viewpoint=None)  # must not raise
