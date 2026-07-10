"""
Tests for topocore.geometry.base.
"""

from __future__ import annotations

from typing import final

from topocore.geometry.base import Geometry


@final
class DummyGeometry(Geometry):
    """
    Concrete implementation used for testing.
    """

    def to_dict(self) -> dict[str, object]:
        return {
            "x": 10,
            "y": 20,
        }


def test_to_dict() -> None:
    geometry = DummyGeometry()

    assert geometry.to_dict() == {
        "x": 10,
        "y": 20,
    }


def test_repr() -> None:
    geometry = DummyGeometry()

    assert repr(geometry) == "DummyGeometry(x=10, y=20)"
