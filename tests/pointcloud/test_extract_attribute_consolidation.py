"""
Regression suite for Phase 4 (Architectural bypass / duplication):
classification/ml.py's two independent reimplementations of
_shared.extract_attribute() now consume the shared abstraction
directly.

Site 1: _GroundRelativeHeightFeatureComputer.compute() previously did
`np.concatenate([chunk[PointAttribute.X] for chunk in cloud])`
inline for X, Y, and Z individually. Now calls extract_attribute()
for each. X/Y/Z are declared float64 in ATTRIBUTE_DTYPES, so the
trailing astype(..., copy=False) remains a no-op safety cast.

Site 2: MachineLearningClassifier._extract_attribute() previously
reimplemented the same concatenation/validation logic itself, with
its own per-chunk `.astype(np.float64)` cast. Now delegates to
extract_attribute() and applies the float64 cast to the result.
Confirmed this cast is genuinely necessary, not redundant: INTENSITY
is declared uint16, and RETURN_NUMBER/NUMBER_OF_RETURNS are declared
uint8 in ATTRIBUTE_DTYPES -- a naive substitution without preserving
the cast would have silently changed MachineLearningClassifier's own
feature-matrix dtype contract.

Both sites are confirmed to produce identical dtypes and values to
their pre-consolidation implementations.
"""

from __future__ import annotations

import numpy as np
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.ml import (
    MachineLearningClassifier,
    _GroundRelativeHeightFeatureComputer,
)


class _FakeModel:
    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.ones((X.shape[0], 2)) * 0.5


def test_ground_relative_height_computer_returns_float64() -> None:
    rng = np.random.default_rng(0)
    n = 30
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 5, n)
    cloud.add_chunk(chunk)

    result = _GroundRelativeHeightFeatureComputer().compute(cloud)

    assert result.shape == (n,)
    assert result.dtype == np.float64


def test_extract_attribute_casts_non_float64_radiometric_dtype_correctly() -> None:
    """INTENSITY is native uint16 -- the feature matrix must still come out float64 with identical values."""
    n = 30
    intensity_values = np.arange(n, dtype=np.uint16)

    cloud = PointCloud()
    chunk = Chunk(
        size=n,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
            PointAttribute.INTENSITY,
        ],
    )
    chunk[PointAttribute.X][:] = np.zeros(n)
    chunk[PointAttribute.Y][:] = np.zeros(n)
    chunk[PointAttribute.Z][:] = np.zeros(n)
    chunk[PointAttribute.INTENSITY][:] = intensity_values
    cloud.add_chunk(chunk)

    clf = MachineLearningClassifier(_FakeModel(), feature_names=["intensity"])
    matrix = clf._build_feature_matrix(cloud)

    assert matrix.dtype == np.float64
    np.testing.assert_array_equal(matrix[:, 0], intensity_values.astype(np.float64))
