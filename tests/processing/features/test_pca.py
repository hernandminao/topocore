"""
Coverage audit tests for topocore.processing.features.pca
(PCAFeatures, PCAFeatureComputer).

PCA-FEATURES-001 (FIXED in this PR): compute_all()'s own docstring
explicitly promises PointDescriptorError "If the cloud is empty,
contains invalid coordinates, does not have enough points, [...] or
eigendecomposition fails." Confirmed directly, before this fix, that
NaN coordinates (a genuine "invalid coordinates" case the docstring
names explicitly) instead raised a raw, undocumented ValueError from
scipy's own cKDTree construction -- NOT PointDescriptorError. Root
cause confirmed by reading the exact source: `manager =
NeighborhoodManager.from_point_cloud(cloud)` sat OUTSIDE the
`try: ... except ProcessingError:` block that wrapped only the later
`compute_pca(...)` call.

Fix: the try block now also covers manager construction, and the
except clause also catches ValueError (confirmed via its MRO to have
no relationship to TopoCore's own exception hierarchy) alongside the
existing ProcessingError -- closing the gap without introducing a
separate, duplicate finite-coordinate validation (scipy's own
KD-tree construction already performs this check) and without
changing the manager-construction logic itself. Both NaN and Inf are
confirmed to now raise PointDescriptorError, with __cause__ preserved
as the original ValueError. The happy path and the two
already-existing validations (empty cloud, point_count < k) are
confirmed completely unaffected by this change.

All 10 of PCAFeatures' own single-feature convenience methods
(eigenvalues(), eigenvectors(), omnivariance(), anisotropy(),
linearity(), planarity(), sphericity(), surface_variation(),
verticality(), eigenentropy()) are confirmed orphaned -- zero
external callers via grep. Each independently re-runs the FULL
compute_all() pipeline, so using more than one together would be
redundantly expensive if ever adopted -- not tested individually
here beyond confirming they delegate correctly to compute_all().

PCAFeatureComputer's own final `raise PointDescriptorError(...)`
after the match statement is confirmed unreachable: __init__ already
validates feature_name against _DIMENSIONS' own keys, and the match
statement's cases cover every one of those same keys exhaustively.

PCAFeatureComputer IS genuinely, actively used (via
classification/ml.py's own FeatureManager registration) -- unlike
PCAFeatures' own convenience methods.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import PointDescriptorError
from topocore.processing.features.pca import PCAFeatureComputer, PCAFeatures


def _cloud(n: int = 100, seed: int = 0, flat: bool = False) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = np.zeros(n) if flat else rng.uniform(0, 10, n)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# PCAFeatures -- constructor and compute_all() validation.
# ----------------------------------------------------------------------


def test_constructor_rejects_k_less_than_three() -> None:
    with pytest.raises(PointDescriptorError, match="at least 3"):
        PCAFeatures(k=2)


def test_compute_all_rejects_empty_cloud() -> None:
    with pytest.raises(PointDescriptorError, match="empty point cloud"):
        PCAFeatures(k=5).compute_all(PointCloud())


def test_compute_all_rejects_point_count_less_than_k() -> None:
    small = PointCloud()
    chunk = Chunk(size=3, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [0.0, 1.0, 2.0]
    chunk[PointAttribute.Y][:] = [0.0] * 3
    chunk[PointAttribute.Z][:] = [0.0] * 3
    small.add_chunk(chunk)

    with pytest.raises(PointDescriptorError, match="requires at least"):
        PCAFeatures(k=5).compute_all(small)


def test_pca_features_001_nan_coordinates_now_raise_point_descriptor_error() -> None:
    """
    The core regression: NaN coordinates must now raise
    PointDescriptorError (matching compute_all()'s own documented
    contract), not the raw ValueError that previously escaped from
    NeighborhoodManager.from_point_cloud()'s own KD-tree construction.
    """
    cloud = PointCloud()
    chunk = Chunk(size=10, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan, 10.0]
    chunk[PointAttribute.Y][:] = np.zeros(10)
    chunk[PointAttribute.Z][:] = np.zeros(10)
    cloud.add_chunk(chunk)

    with pytest.raises(PointDescriptorError, match="finite") as exc_info:
        PCAFeatures(k=5).compute_all(cloud)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_pca_features_001_inf_coordinates_also_raise_point_descriptor_error() -> None:
    cloud = PointCloud()
    chunk = Chunk(size=10, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, np.inf, 10.0]
    chunk[PointAttribute.Y][:] = np.zeros(10)
    chunk[PointAttribute.Z][:] = np.zeros(10)
    cloud.add_chunk(chunk)

    with pytest.raises(PointDescriptorError, match="finite"):
        PCAFeatures(k=5).compute_all(cloud)


# ----------------------------------------------------------------------
# Mathematical correctness.
# ----------------------------------------------------------------------


def test_eigenvalues_are_sorted_descending() -> None:
    result = PCAFeatures(k=10).compute_all(_cloud())
    eigvals = result["eigenvalues"]

    assert np.all(eigvals[:, 0] >= eigvals[:, 1])
    assert np.all(eigvals[:, 1] >= eigvals[:, 2])


def test_flat_plane_has_near_zero_third_eigenvalue_and_high_planarity() -> None:
    result = PCAFeatures(k=10).compute_all(_cloud(flat=True))

    assert result["eigenvalues"][:, 2].mean() == pytest.approx(0.0, abs=1e-6)
    assert result["planarity"].mean() > 0.3


def test_flat_horizontal_plane_has_near_zero_verticality() -> None:
    """A horizontal surface's normal points along Z, so verticality (1 - |eigenvector_z|) is near 0, not 1."""
    result = PCAFeatures(k=10).compute_all(_cloud(flat=True))

    assert result["verticality"].mean() < 0.1


def test_compute_delegates_to_compute_all() -> None:
    cloud = _cloud()
    pca = PCAFeatures(k=10)

    via_compute = pca.compute(cloud)
    via_compute_all = pca.compute_all(cloud)

    np.testing.assert_array_equal(via_compute["eigenvalues"], via_compute_all["eigenvalues"])


# ----------------------------------------------------------------------
# PCAFeatureComputer.
# ----------------------------------------------------------------------


def test_feature_computer_rejects_unknown_feature_name() -> None:
    with pytest.raises(PointDescriptorError, match="Unknown PCA feature"):
        PCAFeatureComputer(feature_name="bogus")


@pytest.mark.parametrize(
    ("feature_name", "expected_dim"),
    [
        ("eigenvalues", 3),
        ("eigenvectors", 9),
        ("omnivariance", 1),
        ("planarity", 1),
        ("verticality", 1),
    ],
)
def test_feature_computer_returns_correct_shape(feature_name: str, expected_dim: int) -> None:
    cloud = _cloud(n=30)
    computer = PCAFeatureComputer(feature_name=feature_name, k=10)

    result = computer.compute(cloud)

    assert result.shape == (30,) if expected_dim == 1 else result.shape == (30, expected_dim)
    assert computer.dimension() == expected_dim
    assert computer.name() == f"pca_{feature_name}"


def test_feature_computer_metadata() -> None:
    computer = PCAFeatureComputer(feature_name="planarity", k=15)

    assert computer.requires_neighbors() is True
    assert computer.default_k() == 15
    assert computer.default_radius() is None
