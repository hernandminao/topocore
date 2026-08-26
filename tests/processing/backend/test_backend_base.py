"""Tests for topocore.processing.backend.base."""

from __future__ import annotations

import inspect
from typing import ForwardRef

import pytest

from topocore.processing.backend.base import Backend, T

EXPECTED_ABSTRACT_METHODS = {
    "array",
    "zeros",
    "ones",
    "full",
    "empty",
    "arange",
    "linspace",
    "meshgrid",
    "shape",
    "dtype",
    "size",
    "ndim",
    "astype",
    "to_numpy",
    "add",
    "subtract",
    "multiply",
    "divide",
    "matmul",
    "transpose",
    "reshape",
    "sum",
    "mean",
    "var",
    "std",
    "min",
    "max",
    "abs",
    "sqrt",
    "square",
    "eigvalsh",
    "eigh",
    "svd",
    "qr",
    "solve",
    "sort",
    "argsort",
    "where",
    "unique",
    "broadcast_to",
    "concatenate",
    "stack",
    "random_uniform",
    "random_choice",
    "is_array",
}


def _abstract_methods() -> set[str]:
    """Return abstract Backend methods without using ABC internals."""
    return {name for name, member in inspect.getmembers(Backend) if getattr(member, "__isabstractmethod__", False)}


def test_backend_is_abstract_strategy_interface() -> None:
    """Backend is an abstract class and cannot be instantiated."""
    assert inspect.isabstract(Backend)

    with pytest.raises(TypeError):
        Backend()


def test_backend_declares_all_required_operations() -> None:
    """Backend exposes every required abstract operation."""
    assert _abstract_methods() == EXPECTED_ABSTRACT_METHODS


def test_backend_methods_are_abstract() -> None:
    """Every required backend operation is marked abstract."""
    for name in EXPECTED_ABSTRACT_METHODS:
        attribute = getattr(Backend, name)

        assert getattr(attribute, "__isabstractmethod__", False) is True


def test_backend_public_export() -> None:
    """Only Backend is exported from the base module."""
    import topocore.processing.backend.base as module

    assert module.__all__ == ["Backend"]


def test_backend_typevar_is_bound_to_backend() -> None:
    """T is bound to Backend, including postponed annotations."""
    bound = T.__bound__

    assert bound is not None

    if bound is Backend:
        return

    if isinstance(bound, ForwardRef):
        assert bound.__forward_arg__ == "Backend"
        return

    if isinstance(bound, str):
        assert bound == "Backend"
        return

    raise AssertionError(f"Unexpected TypeVar bound: {bound!r}")


def test_backend_method_annotations_exist() -> None:
    """Abstract backend methods retain type annotations."""
    for name in EXPECTED_ABSTRACT_METHODS:
        method = getattr(Backend, name)

        assert getattr(method, "__annotations__", None) is not None
