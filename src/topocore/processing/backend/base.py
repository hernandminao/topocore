"""
topocore.processing.backend.base
================================

Abstract base class for mathematical backends.

This module defines the interface that all backends (NumPy, CuPy, Torch,
JAX, etc.) must implement. By programming against this interface, the
entire processing subsystem can switch backends without modifying
algorithm implementations.

The backend is responsible for providing:
- Array creation and manipulation
- Linear algebra operations (eigenvalues, SVD, matrix multiplication)
- Sorting and searching
- Reduction operations (sum, mean, min, max)
- Element-wise operations (sin, cos, sqrt, etc.)

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", bound="Backend")


class Backend(ABC):
    """
    Abstract interface for mathematical backends.

    All backend implementations must inherit from this class and implement
    all abstract methods. This ensures that the processing subsystem can
    operate uniformly regardless of the underlying array library.

    Examples
    --------
    >>> backend = NumPyBackend()
    >>> arr = backend.array([1, 2, 3])
    >>> mean = backend.mean(arr)
    """

    # ========================================================================
    # Array Creation
    # ========================================================================

    @abstractmethod
    def array(
        self,
        data: Any,
        dtype: np.dtype | None = None,
    ) -> Any:
        """
        Create an array from the given data.

        Parameters
        ----------
        data
            Input data (list, tuple, array, etc.).
        dtype
            Desired data type.

        Returns
        -------
        Any
            An array in the backend's native format.
        """
        ...

    @abstractmethod
    def zeros(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype | None = None,
    ) -> Any:
        """
        Create an array of zeros.

        Parameters
        ----------
        shape
            Array shape.
        dtype
            Desired data type.

        Returns
        -------
        Any
            An array of zeros.
        """
        ...

    @abstractmethod
    def ones(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype | None = None,
    ) -> Any:
        """
        Create an array of ones.

        Parameters
        ----------
        shape
            Array shape.
        dtype
            Desired data type.

        Returns
        -------
        Any
            An array of ones.
        """
        ...

    @abstractmethod
    def full(
        self,
        shape: tuple[int, ...],
        fill_value: float,
        dtype: np.dtype | None = None,
    ) -> Any:
        """
        Create an array filled with a constant value.

        Parameters
        ----------
        shape
            Array shape.
        fill_value
            Value to fill.
        dtype
            Desired data type.

        Returns
        -------
        Any
            An array filled with the given value.
        """
        ...

    @abstractmethod
    def empty(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype | None = None,
    ) -> Any:
        """
        Create an uninitialized array.

        Parameters
        ----------
        shape
            Array shape.
        dtype
            Desired data type.

        Returns
        -------
        Any
            An uninitialized array.
        """
        ...

    @abstractmethod
    def arange(
        self,
        start: float,
        stop: float,
        step: float = 1.0,
        dtype: np.dtype | None = None,
    ) -> Any:
        """
        Create an array with evenly spaced values.

        Parameters
        ----------
        start
            Start value.
        stop
            Stop value (exclusive).
        step
            Step size.
        dtype
            Desired data type.

        Returns
        -------
        Any
            An array with evenly spaced values.
        """
        ...

    @abstractmethod
    def linspace(
        self,
        start: float,
        stop: float,
        num: int,
        dtype: np.dtype | None = None,
    ) -> Any:
        """
        Create an array with linearly spaced values.

        Parameters
        ----------
        start
            Start value.
        stop
            Stop value (inclusive).
        num
            Number of points.
        dtype
            Desired data type.

        Returns
        -------
        Any
            An array with linearly spaced values.
        """
        ...

    @abstractmethod
    def meshgrid(
        self,
        x: Any,
        y: Any,
    ) -> tuple[Any, Any]:
        """
        Create coordinate matrices from coordinate vectors.

        Parameters
        ----------
        x
            X coordinates.
        y
            Y coordinates.

        Returns
        -------
        tuple
            (X_grid, Y_grid) arrays.
        """
        ...

    # ========================================================================
    # Array Properties
    # ========================================================================

    @abstractmethod
    def shape(
        self,
        arr: Any,
    ) -> tuple[int, ...]:
        """
        Get the shape of an array.

        Parameters
        ----------
        arr
            Input array.

        Returns
        -------
        tuple[int, ...]
            Array shape.
        """
        ...

    @abstractmethod
    def dtype(
        self,
        arr: Any,
    ) -> np.dtype:
        """
        Get the data type of an array.

        Parameters
        ----------
        arr
            Input array.

        Returns
        -------
        np.dtype
            Array data type.
        """
        ...

    @abstractmethod
    def size(
        self,
        arr: Any,
    ) -> int:
        """
        Get the number of elements in an array.

        Parameters
        ----------
        arr
            Input array.

        Returns
        -------
        int
            Number of elements.
        """
        ...

    @abstractmethod
    def ndim(
        self,
        arr: Any,
    ) -> int:
        """
        Get the number of dimensions of an array.

        Parameters
        ----------
        arr
            Input array.

        Returns
        -------
        int
            Number of dimensions.
        """
        ...

    # ========================================================================
    # Type Conversion
    # ========================================================================

    @abstractmethod
    def astype(
        self,
        arr: Any,
        dtype: np.dtype,
    ) -> Any:
        """
        Cast an array to a different data type.

        Parameters
        ----------
        arr
            Input array.
        dtype
            Desired data type.

        Returns
        -------
        Any
            Cast array.
        """
        ...

    @abstractmethod
    def to_numpy(
        self,
        arr: Any,
    ) -> NDArray[Any]:
        """
        Convert a backend array to a NumPy array.

        This is the canonical method for extracting data from the backend
        for use in non-backend-aware code.

        Parameters
        ----------
        arr
            Backend array.

        Returns
        -------
        NDArray
            NumPy array.
        """
        ...

    # ========================================================================
    # Operations
    # ========================================================================

    @abstractmethod
    def add(
        self,
        a: Any,
        b: Any,
    ) -> Any:
        """Element-wise addition."""
        ...

    @abstractmethod
    def subtract(
        self,
        a: Any,
        b: Any,
    ) -> Any:
        """Element-wise subtraction."""
        ...

    @abstractmethod
    def multiply(
        self,
        a: Any,
        b: Any,
    ) -> Any:
        """Element-wise multiplication."""
        ...

    @abstractmethod
    def divide(
        self,
        a: Any,
        b: Any,
    ) -> Any:
        """Element-wise division."""
        ...

    @abstractmethod
    def matmul(
        self,
        a: Any,
        b: Any,
    ) -> Any:
        """Matrix multiplication."""
        ...

    @abstractmethod
    def transpose(
        self,
        arr: Any,
    ) -> Any:
        """Transpose an array."""
        ...

    @abstractmethod
    def reshape(
        self,
        arr: Any,
        shape: tuple[int, ...],
    ) -> Any:
        """Reshape an array."""
        ...

    # ========================================================================
    # Reduction Operations
    # ========================================================================

    @abstractmethod
    def sum(
        self,
        arr: Any,
        axis: int | None = None,
        keepdims: bool = False,
    ) -> Any:
        """Compute the sum of array elements."""
        ...

    @abstractmethod
    def mean(
        self,
        arr: Any,
        axis: int | None = None,
        keepdims: bool = False,
    ) -> Any:
        """Compute the mean of array elements."""
        ...

    @abstractmethod
    def var(
        self,
        arr: Any,
        axis: int | None = None,
        keepdims: bool = False,
    ) -> Any:
        """Compute the variance of array elements."""
        ...

    @abstractmethod
    def std(
        self,
        arr: Any,
        axis: int | None = None,
        keepdims: bool = False,
    ) -> Any:
        """Compute the standard deviation of array elements."""
        ...

    @abstractmethod
    def min(
        self,
        arr: Any,
        axis: int | None = None,
        keepdims: bool = False,
    ) -> Any:
        """Compute the minimum of array elements."""
        ...

    @abstractmethod
    def max(
        self,
        arr: Any,
        axis: int | None = None,
        keepdims: bool = False,
    ) -> Any:
        """Compute the maximum of array elements."""
        ...

    @abstractmethod
    def abs(
        self,
        arr: Any,
    ) -> Any:
        """Compute the absolute value element-wise."""
        ...

    @abstractmethod
    def sqrt(
        self,
        arr: Any,
    ) -> Any:
        """Compute the square root element-wise."""
        ...

    @abstractmethod
    def square(
        self,
        arr: Any,
    ) -> Any:
        """Compute the square element-wise."""
        ...

    # ========================================================================
    # Linear Algebra
    # ========================================================================

    @abstractmethod
    def eigvalsh(
        self,
        arr: Any,
    ) -> Any:
        """
        Compute the eigenvalues of a symmetric/Hermitian matrix.

        Parameters
        ----------
        arr
            Symmetric matrix.

        Returns
        -------
        Any
            Eigenvalues (ascending order).
        """
        ...

    @abstractmethod
    def eigh(
        self,
        arr: Any,
    ) -> tuple[Any, Any]:
        """
        Compute the eigenvalues and eigenvectors of a symmetric/Hermitian matrix.

        Parameters
        ----------
        arr
            Symmetric matrix.

        Returns
        -------
        tuple
            (eigenvalues, eigenvectors) where eigenvalues are in
            ascending order and eigenvectors are columns.
        """
        ...

    @abstractmethod
    def svd(
        self,
        arr: Any,
        full_matrices: bool = False,
    ) -> tuple[Any, Any, Any]:
        """
        Compute the Singular Value Decomposition (SVD).

        Parameters
        ----------
        arr
            Input matrix.
        full_matrices
            Whether to compute full or reduced SVD.

        Returns
        -------
        tuple
            (U, S, Vh) where S is a 1D array of singular values.
        """
        ...

    @abstractmethod
    def qr(
        self,
        arr: Any,
    ) -> tuple[Any, Any]:
        """
        Compute the QR decomposition.

        Parameters
        ----------
        arr
            Input matrix.

        Returns
        -------
        tuple
            (Q, R) matrices.
        """
        ...

    @abstractmethod
    def solve(
        self,
        a: Any,
        b: Any,
    ) -> Any:
        """
        Solve a linear system A * x = b.

        Parameters
        ----------
        a
            Coefficient matrix.
        b
            Right-hand side vector or matrix.

        Returns
        -------
        Any
            Solution x.
        """
        ...

    # ========================================================================
    # Searching and Sorting
    # ========================================================================

    @abstractmethod
    def sort(
        self,
        arr: Any,
        axis: int = -1,
    ) -> Any:
        """Sort an array along an axis."""
        ...

    @abstractmethod
    def argsort(
        self,
        arr: Any,
        axis: int = -1,
    ) -> Any:
        """Return indices that would sort the array."""
        ...

    @abstractmethod
    def where(
        self,
        condition: Any,
        x: Any | None = None,
        y: Any | None = None,
    ) -> Any:
        """Return elements chosen from x or y depending on condition."""
        ...

    @abstractmethod
    def unique(
        self,
        arr: Any,
        return_counts: bool = False,
    ) -> Any:
        """
        Return unique values from an array.

        Parameters
        ----------
        arr
            Input array.
        return_counts
            Whether to return counts.

        Returns
        -------
        Any
            Unique values (and optionally counts).
        """
        ...

    # ========================================================================
    # Broadcasting and Indexing
    # ========================================================================

    @abstractmethod
    def broadcast_to(
        self,
        arr: Any,
        shape: tuple[int, ...],
    ) -> Any:
        """Broadcast an array to a new shape."""
        ...

    @abstractmethod
    def concatenate(
        self,
        arrays: list[Any],
        axis: int = 0,
    ) -> Any:
        """Concatenate arrays along an axis."""
        ...

    @abstractmethod
    def stack(
        self,
        arrays: list[Any],
        axis: int = 0,
    ) -> Any:
        """Stack arrays along a new axis."""
        ...

    # ========================================================================
    # Random
    # ========================================================================

    @abstractmethod
    def random_uniform(
        self,
        low: float,
        high: float,
        size: tuple[int, ...] | int,
        seed: int | None = None,
    ) -> Any:
        """
        Draw random samples from a uniform distribution.

        Parameters
        ----------
        low
            Lower bound.
        high
            Upper bound.
        size
            Output shape.
        seed
            Random seed.

        Returns
        -------
        Any
            Random samples.
        """
        ...

    @abstractmethod
    def random_choice(
        self,
        a: Any,
        size: int,
        replace: bool = True,
        p: Any | None = None,
        seed: int | None = None,
    ) -> Any:
        """
        Generate random samples from a given array.

        Parameters
        ----------
        a
            1D array-like.
        size
            Number of samples.
        replace
            Whether to sample with replacement.
        p
            Probabilities for each element.
        seed
            Random seed.

        Returns
        -------
        Any
            Random samples.
        """
        ...

    # ========================================================================
    # Helper Methods
    # ========================================================================

    @abstractmethod
    def is_array(
        self,
        obj: Any,
    ) -> bool:
        """
        Check if an object is a backend array.

        Parameters
        ----------
        obj
            Object to check.

        Returns
        -------
        bool
            True if the object is a backend array.
        """
        ...


__all__ = [
    "Backend",
]
