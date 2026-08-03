"""
topocore.processing.backend.numpy_backend
=========================================

NumPy implementation of the Backend interface.

This module provides the default backend implementation using NumPy.
All operations are delegated to NumPy functions, ensuring optimal
performance for CPU-based processing.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Any, cast, final

import numpy as np
from numpy.typing import NDArray

from .base import Backend


@final
class NumPyBackend(Backend):
    """
    NumPy implementation of the Backend interface.

    This is the default backend for TopoCore, providing a complete
    implementation of all abstract methods using NumPy.
    """

    __slots__ = ()

    # ========================================================================
    # Array Creation
    # ========================================================================

    def array(
        self,
        data: Any,
        dtype: np.dtype[Any] | None = None,
    ) -> NDArray[Any]:
        return np.asarray(data, dtype=dtype)

    def zeros(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any] | None = None,
    ) -> NDArray[Any]:
        return np.zeros(shape, dtype=dtype)

    def ones(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any] | None = None,
    ) -> NDArray[Any]:
        return np.ones(shape, dtype=dtype)

    def full(
        self,
        shape: tuple[int, ...],
        fill_value: float,
        dtype: np.dtype[Any] | None = None,
    ) -> NDArray[Any]:
        return np.full(shape, fill_value, dtype=dtype)

    def empty(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any] | None = None,
    ) -> NDArray[Any]:
        return np.empty(shape, dtype=dtype)

    def arange(
        self,
        start: float,
        stop: float,
        step: float = 1.0,
        dtype: np.dtype[Any] | None = None,
    ) -> NDArray[Any]:
        return np.arange(start, stop, step, dtype=dtype)

    def linspace(
        self,
        start: float,
        stop: float,
        num: int,
        dtype: np.dtype[Any] | None = None,
    ) -> NDArray[Any]:
        return np.linspace(start, stop, num, dtype=dtype)

    def meshgrid(
        self,
        x: Any,
        y: Any,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        return np.meshgrid(x, y)

    # ========================================================================
    # Array Properties
    # ========================================================================

    def shape(
        self,
        arr: NDArray[Any],
    ) -> tuple[int, ...]:
        return arr.shape

    def dtype(
        self,
        arr: NDArray[Any],
    ) -> np.dtype[Any]:
        return arr.dtype

    def size(
        self,
        arr: NDArray[Any],
    ) -> int:
        return arr.size

    def ndim(
        self,
        arr: NDArray[Any],
    ) -> int:
        return arr.ndim

    # ========================================================================
    # Type Conversion
    # ========================================================================

    def astype(self, arr: NDArray[Any], dtype: np.dtype[Any]) -> NDArray[Any]:
        return arr.astype(dtype)

    def to_numpy(
        self,
        arr: Any,
    ) -> NDArray[Any]:
        if isinstance(arr, np.ndarray):
            return arr
        # For compatibility with other backends that might pass through
        return np.asarray(arr)

    # ========================================================================
    # Operations
    # ========================================================================

    def add(
        self,
        a: NDArray[Any],
        b: NDArray[Any],
    ) -> NDArray[Any]:
        return cast(NDArray[Any], np.add(a, b))

    def subtract(
        self,
        a: NDArray[Any],
        b: NDArray[Any],
    ) -> NDArray[Any]:
        return cast(NDArray[Any], np.subtract(a, b))

    def multiply(
        self,
        a: NDArray[Any],
        b: NDArray[Any],
    ) -> NDArray[Any]:
        return cast(NDArray[Any], np.multiply(a, b))

    def divide(
        self,
        a: NDArray[Any],
        b: NDArray[Any],
    ) -> NDArray[Any]:
        return cast(NDArray[Any], np.divide(a, b))

    def matmul(
        self,
        a: NDArray[Any],
        b: NDArray[Any],
    ) -> NDArray[Any]:
        return cast(NDArray[Any], np.matmul(a, b))

    def transpose(
        self,
        arr: NDArray[Any],
    ) -> NDArray[Any]:
        return arr.T

    def reshape(
        self,
        arr: NDArray[Any],
        shape: tuple[int, ...],
    ) -> NDArray[Any]:
        return arr.reshape(shape)

    # ========================================================================
    # Reduction Operations
    # ========================================================================

    def sum(
        self,
        arr: NDArray[Any],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any:
        np_sum = cast(Any, np.sum)

        return np_sum(
            arr,
            axis=axis,
            keepdims=keepdims,
        )

    def mean(
        self,
        arr: NDArray[Any],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any:
        np_mean = cast(Any, np.mean)

        return np_mean(
            arr,
            axis=axis,
            keepdims=keepdims,
        )

    def var(
        self,
        arr: NDArray[Any],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any:
        np_var = cast(Any, np.var)

        return np_var(
            arr,
            axis=axis,
            keepdims=keepdims,
        )

    def std(
        self,
        arr: NDArray[Any],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any:
        np_std = cast(Any, np.std)

        return np_std(
            arr,
            axis=axis,
            keepdims=keepdims,
        )

    def min(
        self,
        arr: NDArray[Any],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any:
        np_min = cast(Any, np.min)

        return np_min(
            arr,
            axis=axis,
            keepdims=keepdims,
        )

    def max(
        self,
        arr: NDArray[Any],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any:
        np_max = cast(Any, np.max)

        return np_max(
            arr,
            axis=axis,
            keepdims=keepdims,
        )

    def abs(
        self,
        arr: NDArray[Any],
    ) -> NDArray[Any]:
        return np.abs(arr)

    def sqrt(
        self,
        arr: NDArray[Any],
    ) -> NDArray[Any]:
        return np.sqrt(arr)

    def square(
        self,
        arr: NDArray[Any],
    ) -> NDArray[Any]:
        return np.square(arr)

    # ========================================================================
    # Linear Algebra
    # ========================================================================

    def eigvalsh(
        self,
        arr: Any,
    ) -> NDArray[Any]:
        return np.linalg.eigvalsh(arr)

    def eigh(
        self,
        arr: NDArray[Any],
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        eigenvalues, eigenvectors = np.linalg.eigh(arr)
        return eigenvalues, eigenvectors

    def svd(
        self,
        arr: NDArray[Any],
        full_matrices: bool = False,
    ) -> tuple[
        NDArray[Any],
        NDArray[Any],
        NDArray[Any],
    ]:
        u, s, vh = np.linalg.svd(
            arr,
            full_matrices=full_matrices,
        )
        return u, s, vh

    def qr(
        self,
        arr: NDArray[Any],
    ) -> tuple[
        NDArray[Any],
        NDArray[Any],
    ]:
        q, r = np.linalg.qr(arr)
        return q, r

    def solve(
        self,
        a: Any,
        b: Any,
    ) -> NDArray[Any]:
        return np.linalg.solve(a, b)

    # ========================================================================
    # Searching and Sorting
    # ========================================================================

    def sort(
        self,
        arr: Any,
        axis: int = -1,
    ) -> NDArray[Any]:
        return np.sort(arr, axis=axis)

    def argsort(
        self,
        arr: Any,
        axis: int = -1,
    ) -> NDArray[Any]:
        return np.argsort(arr, axis=axis)

    def where(
        self,
        condition: Any,
        x: Any | None = None,
        y: Any | None = None,
    ) -> Any:
        return np.where(condition, x, y)

    def unique(
        self,
        arr: NDArray[Any],
        return_counts: bool = False,
    ) -> Any:
        np_unique = cast(Any, np.unique)

        return np_unique(arr, return_counts=return_counts)

    # ========================================================================
    # Broadcasting and Indexing
    # ========================================================================

    def broadcast_to(
        self,
        arr: Any,
        shape: tuple[int, ...],
    ) -> NDArray[Any]:
        return np.broadcast_to(arr, shape)

    def concatenate(
        self,
        arrays: list[Any],
        axis: int = 0,
    ) -> NDArray[Any]:
        return np.concatenate(arrays, axis=axis)

    def stack(
        self,
        arrays: list[Any],
        axis: int = 0,
    ) -> NDArray[Any]:
        return np.stack(arrays, axis=axis)

    # ========================================================================
    # Random
    # ========================================================================

    def random_uniform(
        self,
        low: float,
        high: float,
        size: tuple[int, ...] | int,
        seed: int | None = None,
    ) -> NDArray[Any]:
        rng = np.random.default_rng(seed)
        return rng.uniform(low, high, size)

    def random_choice(
        self,
        a: Any,
        size: int,
        replace: bool = True,
        p: Any | None = None,
        seed: int | None = None,
    ) -> NDArray[Any]:
        rng = np.random.default_rng(seed)
        return rng.choice(a, size=size, replace=replace, p=p)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def is_array(
        self,
        obj: Any,
    ) -> bool:
        return isinstance(obj, np.ndarray)


# Singleton instance for convenience.
NUMPY_BACKEND: NumPyBackend = NumPyBackend()

__all__ = [
    "NumPyBackend",
    "NUMPY_BACKEND",
]
