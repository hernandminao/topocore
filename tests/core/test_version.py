# mypy: ignore-errors
"""
Unit tests for version information.
"""

from __future__ import annotations

from topocore import __version__


def test_version_exists() -> None:
    """Test that __version__ is defined."""
    assert __version__ is not None
    assert isinstance(__version__, str)


def test_version_format() -> None:
    """Test that version follows semantic versioning."""
    assert len(__version__.split(".")) >= 2
