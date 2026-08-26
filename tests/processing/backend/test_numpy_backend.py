"""
Tests for topocore.processing.backend.numpy_backend.

Validates the public behavior of the NumPy backend against NumPy's
reference implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.processing.backend.numpy_backend import (
    NUMPY_BACKEND,
    NumPyBackend,
)


@pytest.fixture
def backend() -> NumPyBackend:
    return NumPyBackend()


# ============================================================================
# Construction
# ============================================================================


def test_numpy_backend_can_be_instantiated() -> None:
    backend = NumPyBackend()

    assert isinstance(backend, NumPyBackend)


def test_numpy_backend_singleton_is_correct_type() -> None:
    assert isinstance(NUMPY_BACKEND, NumPyBackend)


# ============================================================================
# Array creation
# ============================================================================


def test_array() -> None:
    result = NumPyBackend().array([1, 2, 3])

    np.testing.assert_array_equal(
        result,
        np.array([1, 2, 3]),
    )


def test_array_with_dtype() -> None:
    result = NumPyBackend().array(
        [1, 2, 3],
        dtype=np.float64,
    )

    assert result.dtype == np.float64


def test_zeros() -> None:
    result = NumPyBackend().zeros((2, 3))

    np.testing.assert_array_equal(
        result,
        np.zeros((2, 3)),
    )


def test_ones() -> None:
    result = NumPyBackend().ones((2, 3))

    np.testing.assert_array_equal(
        result,
        np.ones((2, 3)),
    )


def test_full() -> None:
    result = NumPyBackend().full((2, 2), 7.0)

    np.testing.assert_array_equal(
        result,
        np.full((2, 2), 7.0),
    )


def test_empty() -> None:
    result = NumPyBackend().empty((2, 3))

    assert result.shape == (2, 3)


def test_arange() -> None:
    result = NumPyBackend().arange(0, 5)

    np.testing.assert_array_equal(
        result,
        np.arange(0, 5),
    )


def test_linspace() -> None:
    result = NumPyBackend().linspace(0, 1, 5)

    np.testing.assert_allclose(
        result,
        np.linspace(0, 1, 5),
    )


def test_meshgrid() -> None:
    backend = NumPyBackend()

    x, y = backend.meshgrid(
        np.array([1, 2]),
        np.array([10, 20, 30]),
    )

    expected_x, expected_y = np.meshgrid(
        np.array([1, 2]),
        np.array([10, 20, 30]),
    )

    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(y, expected_y)


# ============================================================================
# Array properties
# ============================================================================


def test_array_properties() -> None:
    backend = NumPyBackend()
    arr = np.zeros((2, 3), dtype=np.float64)

    assert backend.shape(arr) == (2, 3)
    assert backend.dtype(arr) == np.dtype(np.float64)
    assert backend.size(arr) == 6
    assert backend.ndim(arr) == 2


# ============================================================================
# Type conversion
# ============================================================================


def test_astype() -> None:
    backend = NumPyBackend()
    arr = np.array([1, 2, 3], dtype=np.int32)

    result = backend.astype(
        arr,
        np.dtype(np.float64),
    )

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, arr)


def test_to_numpy_returns_ndarray_unchanged() -> None:
    backend = NumPyBackend()
    arr = np.array([1, 2, 3])

    result = backend.to_numpy(arr)

    assert result is arr


def test_to_numpy_converts_sequence() -> None:
    backend = NumPyBackend()

    result = backend.to_numpy([1, 2, 3])

    np.testing.assert_array_equal(
        result,
        np.array([1, 2, 3]),
    )


# ============================================================================
# Arithmetic
# ============================================================================


def test_arithmetic_operations() -> None:
    backend = NumPyBackend()

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    np.testing.assert_array_equal(
        backend.add(a, b),
        np.array([5.0, 7.0, 9.0]),
    )

    np.testing.assert_array_equal(
        backend.subtract(b, a),
        np.array([3.0, 3.0, 3.0]),
    )

    np.testing.assert_array_equal(
        backend.multiply(a, b),
        np.array([4.0, 10.0, 18.0]),
    )

    np.testing.assert_allclose(
        backend.divide(b, a),
        np.array([4.0, 2.5, 2.0]),
    )


def test_matmul() -> None:
    backend = NumPyBackend()

    a = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
    )

    b = np.array(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ],
    )

    np.testing.assert_array_equal(
        backend.matmul(a, b),
        np.matmul(a, b),
    )


def test_transpose() -> None:
    backend = NumPyBackend()

    arr = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
    )

    np.testing.assert_array_equal(
        backend.transpose(arr),
        arr.T,
    )


def test_reshape() -> None:
    backend = NumPyBackend()
    arr = np.arange(6)

    np.testing.assert_array_equal(
        backend.reshape(arr, (2, 3)),
        arr.reshape(2, 3),
    )


# ============================================================================
# Reductions
# ============================================================================


def test_reductions() -> None:
    backend = NumPyBackend()
    arr = np.array([1.0, 2.0, 3.0, 4.0])

    assert backend.sum(arr) == pytest.approx(10.0)
    assert backend.mean(arr) == pytest.approx(2.5)
    assert backend.var(arr) == pytest.approx(np.var(arr))
    assert backend.std(arr) == pytest.approx(np.std(arr))
    assert backend.min(arr) == pytest.approx(1.0)
    assert backend.max(arr) == pytest.approx(4.0)


def test_reductions_with_axis_and_keepdims() -> None:
    backend = NumPyBackend()

    arr = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
    )

    result = backend.sum(
        arr,
        axis=1,
        keepdims=True,
    )

    expected = np.sum(
        arr,
        axis=1,
        keepdims=True,
    )

    np.testing.assert_array_equal(result, expected)


def test_elementwise_math_operations() -> None:
    backend = NumPyBackend()

    arr = np.array([1.0, 4.0, 9.0])

    np.testing.assert_array_equal(
        backend.abs(-arr),
        np.abs(-arr),
    )

    np.testing.assert_allclose(
        backend.sqrt(arr),
        np.sqrt(arr),
    )

    np.testing.assert_array_equal(
        backend.square(arr),
        np.square(arr),
    )


# ============================================================================
# Linear algebra
# ============================================================================


def test_eigvalsh() -> None:
    backend = NumPyBackend()

    matrix = np.array(
        [
            [2.0, 0.0],
            [0.0, 3.0],
        ],
    )

    np.testing.assert_allclose(
        backend.eigvalsh(matrix),
        np.linalg.eigvalsh(matrix),
    )


def test_eigh() -> None:
    backend = NumPyBackend()

    matrix = np.array(
        [
            [2.0, 1.0],
            [1.0, 2.0],
        ],
    )

    eigenvalues, eigenvectors = backend.eigh(matrix)
    expected_values, expected_vectors = np.linalg.eigh(matrix)

    np.testing.assert_allclose(
        eigenvalues,
        expected_values,
    )

    # Eigenvectors are defined up to sign.
    np.testing.assert_allclose(
        np.abs(eigenvectors),
        np.abs(expected_vectors),
    )


def test_svd() -> None:
    backend = NumPyBackend()

    matrix = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
    )

    u, s, vh = backend.svd(matrix)

    expected_u, expected_s, expected_vh = np.linalg.svd(
        matrix,
        full_matrices=False,
    )

    np.testing.assert_allclose(
        np.abs(u),
        np.abs(expected_u),
    )

    np.testing.assert_allclose(s, expected_s)

    np.testing.assert_allclose(
        np.abs(vh),
        np.abs(expected_vh),
    )


def test_qr() -> None:
    backend = NumPyBackend()

    matrix = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
    )

    q, r = backend.qr(matrix)
    expected_q, expected_r = np.linalg.qr(matrix)

    np.testing.assert_allclose(
        np.abs(q),
        np.abs(expected_q),
    )

    np.testing.assert_allclose(
        np.abs(r),
        np.abs(expected_r),
    )


def test_solve() -> None:
    backend = NumPyBackend()

    a = np.array(
        [
            [3.0, 1.0],
            [1.0, 2.0],
        ],
    )

    b = np.array([9.0, 8.0])

    np.testing.assert_allclose(
        backend.solve(a, b),
        np.linalg.solve(a, b),
    )


# ============================================================================
# Searching / sorting
# ============================================================================


def test_sort_and_argsort() -> None:
    backend = NumPyBackend()
    arr = np.array([3, 1, 2])

    np.testing.assert_array_equal(
        backend.sort(arr),
        np.array([1, 2, 3]),
    )

    np.testing.assert_array_equal(
        backend.argsort(arr),
        np.argsort(arr),
    )


def test_where() -> None:
    backend = NumPyBackend()

    condition = np.array([True, False, True])

    result = backend.where(
        condition,
        np.array([1, 1, 1]),
        np.array([0, 0, 0]),
    )

    np.testing.assert_array_equal(
        result,
        np.array([1, 0, 1]),
    )


def test_unique_without_counts() -> None:
    backend = NumPyBackend()

    arr = np.array([3, 1, 3, 2, 1])

    np.testing.assert_array_equal(
        backend.unique(arr),
        np.array([1, 2, 3]),
    )


def test_unique_with_counts() -> None:
    backend = NumPyBackend()

    arr = np.array([3, 1, 3, 2, 1])

    values, counts = backend.unique(
        arr,
        return_counts=True,
    )

    expected_values, expected_counts = np.unique(
        arr,
        return_counts=True,
    )

    np.testing.assert_array_equal(
        values,
        expected_values,
    )

    np.testing.assert_array_equal(
        counts,
        expected_counts,
    )


# ============================================================================
# Broadcasting
# ============================================================================


def test_broadcast_to() -> None:
    backend = NumPyBackend()

    result = backend.broadcast_to(
        np.array([1, 2, 3]),
        (2, 3),
    )

    expected = np.broadcast_to(
        np.array([1, 2, 3]),
        (2, 3),
    )

    np.testing.assert_array_equal(result, expected)


def test_concatenate() -> None:
    backend = NumPyBackend()

    result = backend.concatenate(
        [
            np.array([1, 2]),
            np.array([3, 4]),
        ],
    )

    np.testing.assert_array_equal(
        result,
        np.array([1, 2, 3, 4]),
    )


def test_stack() -> None:
    backend = NumPyBackend()

    result = backend.stack(
        [
            np.array([1, 2]),
            np.array([3, 4]),
        ],
    )

    expected = np.array(
        [
            [1, 2],
            [3, 4],
        ],
    )

    np.testing.assert_array_equal(result, expected)


# ============================================================================
# Random
# ============================================================================


def test_random_uniform_is_reproducible_with_seed() -> None:
    backend = NumPyBackend()

    first = backend.random_uniform(
        0.0,
        1.0,
        10,
        seed=42,
    )

    second = backend.random_uniform(
        0.0,
        1.0,
        10,
        seed=42,
    )

    np.testing.assert_array_equal(first, second)


def test_random_choice_is_reproducible_with_seed() -> None:
    backend = NumPyBackend()

    values = np.array([10, 20, 30, 40])

    first = backend.random_choice(
        values,
        8,
        seed=42,
    )

    second = backend.random_choice(
        values,
        8,
        seed=42,
    )

    np.testing.assert_array_equal(first, second)


# ============================================================================
# Helper
# ============================================================================


def test_is_array() -> None:
    backend = NumPyBackend()

    assert backend.is_array(np.array([1, 2, 3]))
    assert not backend.is_array([1, 2, 3])
    assert not backend.is_array((1, 2, 3))
    assert not backend.is_array(10)
