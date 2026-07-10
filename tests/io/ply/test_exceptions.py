"""
Tests for topocore.io.ply.exceptions.
"""

from __future__ import annotations

import pytest

from topocore.io.exceptions import InvalidHeaderError
from topocore.io.ply.exceptions import (
    InvalidPLYError,
    PLYError,
)


def test_ply_error_inherits_invalid_header_error() -> None:
    """
    PLYError should inherit from InvalidHeaderError.
    """
    assert issubclass(
        PLYError,
        InvalidHeaderError,
    )


def test_invalid_ply_error_inherits_ply_error() -> None:
    """
    InvalidPLYError should inherit from PLYError.
    """
    assert issubclass(
        InvalidPLYError,
        PLYError,
    )


def test_raise_ply_error() -> None:
    """
    PLYError can be raised.
    """
    with pytest.raises(PLYError):
        raise PLYError(
            "failure",
        )


def test_raise_invalid_ply_error() -> None:
    """
    InvalidPLYError can be raised.
    """
    with pytest.raises(
        InvalidPLYError,
    ):
        raise InvalidPLYError(
            "failure",
        )
