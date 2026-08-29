"""
Coverage audit tests for topocore.processing._shared.compute_pca(),
targeting only the branches confirmed reachable via legitimate public
API usage or genuine invalid input -- not a mechanical "hit every
line" pass.

Audit findings (documented here, not force-tested):

Confirmed via direct execution that KDTreeNeighborSearch (and
therefore any valid NeighborhoodManager) already rejects malformed
point shape and NaN/Inf coordinates AT CONSTRUCTION TIME -- scipy's
own cKDTree raises "data must be finite" for non-finite input, and
KDTreeNeighborSearch's own __init__ raises NeighborError for wrong
shape. This means compute_pca()'s own internal re-checks of
points.ndim/shape and np.isfinite(points) can never actually fire
given any manager that exists at all -- they are defensive
invariants against a future change to NeighborhoodManager's own
contract, not reachable via today's public API. NOT tested here by
design (per explicit review): forcing a test would require
constructing an invalid internal state that the public API cannot
produce, which tests an artificial scenario rather than real
behavior.

Similarly confirmed unreachable given already-tested contracts
elsewhere in the codebase: indices/distances shape mismatches
(guaranteed correct by knn_many()'s own tested contract),
neighbor_points shape/finiteness (a deterministic consequence of
NumPy fancy indexing on already-validated, already-finite arrays),
covariances shape (a deterministic consequence of np.einsum's own
output shape given valid input), LinAlgError from eigh (the
covariance matrix is provably symmetric by construction --
einsum("nki,nkj->nij", centered, centered) is symmetric in i/j by
definition -- and eigh is specifically designed for symmetric
matrices, essentially never failing for genuinely symmetric real
input), and eigenvalue/eigenvector shape (eigh's own deterministic
API contract for a (N,3,3) batch). None of these are tested here;
they remain in the source as safety invariants, not coverage debt.

What IS tested: the two genuinely reachable validation branches
(k < 3, point_count < k), the knn_many() exception-wrapping branch
(reachable only via k > 1,000,000 -- tested here via mock rather
than constructing over a million real points), and the happy path,
whose expected shapes/dtypes/mathematical invariants (orthonormal
eigenvectors, non-negative eigenvalues, neighbor_points ==
points[indices]) were confirmed by direct execution before writing
this test, not assumed from field names.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from topocore.processing._shared import PCAComputation, compute_pca
from topocore.processing.exceptions import ProcessingError
from topocore.processing.neighbors.manager import NeighborhoodManager


@pytest.fixture
def manager() -> NeighborhoodManager:
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 100, (50, 3))
    return NeighborhoodManager.from_array(points)


# ----------------------------------------------------------------------
# Happy path -- validated against the real, executed contract, not assumed.
# ----------------------------------------------------------------------


def test_happy_path_returns_pca_computation_with_correct_shapes(
    manager: NeighborhoodManager,
) -> None:
    k = 5
    result = compute_pca(manager, k=k)

    assert isinstance(result, PCAComputation)
    assert result.points.shape == (50, 3)
    assert result.indices.shape == (50, k)
    assert result.distances.shape == (50, k)
    assert result.neighbor_points.shape == (50, k, 3)
    assert result.eigenvalues.shape == (50, 3)
    assert result.eigenvectors.shape == (50, 3, 3)


def test_happy_path_neighbor_points_matches_points_indexed_by_indices(
    manager: NeighborhoodManager,
) -> None:
    result = compute_pca(manager, k=5)

    np.testing.assert_array_equal(result.neighbor_points, result.points[result.indices])


def test_happy_path_eigenvectors_are_orthonormal(manager: NeighborhoodManager) -> None:
    result = compute_pca(manager, k=5)

    for point_index in range(result.eigenvectors.shape[0]):
        vectors = result.eigenvectors[point_index]
        np.testing.assert_allclose(vectors @ vectors.T, np.eye(3), atol=1e-8)


def test_happy_path_eigenvalues_are_non_negative(manager: NeighborhoodManager) -> None:
    """Covariance matrices are positive semi-definite by construction -- eigenvalues must never be negative."""
    result = compute_pca(manager, k=5)

    assert (result.eigenvalues >= -1e-10).all()


def test_happy_path_result_is_immutable(manager: NeighborhoodManager) -> None:
    from dataclasses import FrozenInstanceError

    result = compute_pca(manager, k=5)

    with pytest.raises(FrozenInstanceError):
        result.points = np.zeros((1, 3))  # type: ignore[misc]


# ----------------------------------------------------------------------
# k < 3 -- genuinely reachable, no other validation precedes it.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 1, 2, -1])
def test_k_less_than_three_raises_processing_error(manager: NeighborhoodManager, k: int) -> None:
    with pytest.raises(ProcessingError, match="k must be at least 3"):
        compute_pca(manager, k=k)


# ----------------------------------------------------------------------
# point_count < k -- genuinely reachable via a small cloud + large k.
# ----------------------------------------------------------------------


def test_point_count_less_than_k_raises_processing_error() -> None:
    small_manager = NeighborhoodManager.from_array(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

    with pytest.raises(ProcessingError, match="requires at least"):
        compute_pca(small_manager, k=5)


# ----------------------------------------------------------------------
# knn_many() failure -- reachable only via k > 1,000,000 in practice;
# tested via mock rather than constructing over a million real points.
# ----------------------------------------------------------------------


def test_knn_many_failure_is_wrapped_in_processing_error_with_cause(
    manager: NeighborhoodManager,
) -> None:
    original_error = RuntimeError("simulated backend failure")

    with (
        patch.object(NeighborhoodManager, "knn_many", side_effect=original_error),
        pytest.raises(ProcessingError, match="Failed to compute PCA neighbourhood search") as exc_info,
    ):
        compute_pca(manager, k=5)

    assert exc_info.value.__cause__ is original_error
