"""Regression tests for topocore.terrain.cell."""

from __future__ import annotations

import pytest

from topocore.terrain.cell import Cell


def test_cell_valid_data() -> None:
    cell = Cell(
        row=2,
        column=3,
        x=10.0,
        y=20.0,
        z=100.0,
    )

    assert cell.row == 2
    assert cell.column == 3
    assert cell.x == 10.0
    assert cell.y == 20.0
    assert cell.z == 100.0

    assert cell.has_data is True
    assert cell.is_nodata is False


@pytest.mark.parametrize(
    "z",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_cell_non_finite_elevation_is_nodata(
    z: float,
) -> None:
    cell = Cell(
        row=0,
        column=0,
        x=0.0,
        y=0.0,
        z=z,
    )

    assert cell.has_data is False
    assert cell.is_nodata is True


def test_cell_xy() -> None:
    cell = Cell(
        row=1,
        column=2,
        x=12.5,
        y=37.5,
        z=100.0,
    )

    assert cell.xy == (12.5, 37.5)


def test_cell_xyz() -> None:
    cell = Cell(
        row=1,
        column=2,
        x=12.5,
        y=37.5,
        z=100.0,
    )

    assert cell.xyz == (12.5, 37.5, 100.0)


def test_cell_distance_is_planimetric() -> None:
    first = Cell(
        row=0,
        column=0,
        x=0.0,
        y=0.0,
        z=100.0,
    )

    second = Cell(
        row=1,
        column=1,
        x=3.0,
        y=4.0,
        z=999.0,
    )

    assert first.distance_to(second) == pytest.approx(5.0)
    assert second.distance_to(first) == pytest.approx(5.0)


def test_cell_distance_to_same_coordinates_is_zero() -> None:
    first = Cell(
        row=0,
        column=0,
        x=10.0,
        y=20.0,
        z=100.0,
    )

    second = Cell(
        row=5,
        column=8,
        x=10.0,
        y=20.0,
        z=-500.0,
    )

    assert first.distance_to(second) == 0.0


def test_cell_translated_updates_selected_fields() -> None:
    original = Cell(
        row=1,
        column=2,
        x=10.0,
        y=20.0,
        z=30.0,
    )

    translated = original.translated(
        row=5,
        column=6,
        x=50.0,
        y=60.0,
        z=70.0,
    )

    assert translated == Cell(
        row=5,
        column=6,
        x=50.0,
        y=60.0,
        z=70.0,
    )


def test_cell_translated_preserves_unspecified_fields() -> None:
    original = Cell(
        row=1,
        column=2,
        x=10.0,
        y=20.0,
        z=30.0,
    )

    translated = original.translated(
        z=99.0,
    )

    assert translated == Cell(
        row=1,
        column=2,
        x=10.0,
        y=20.0,
        z=99.0,
    )


def test_cell_translated_accepts_explicit_none_as_preserve() -> None:
    original = Cell(
        row=1,
        column=2,
        x=10.0,
        y=20.0,
        z=30.0,
    )

    translated = original.translated(
        row=None,
        column=None,
        x=None,
        y=None,
        z=None,
    )

    assert translated == original


def test_cell_is_immutable() -> None:
    cell = Cell(
        row=0,
        column=0,
        x=0.0,
        y=0.0,
        z=1.0,
    )

    with pytest.raises(AttributeError):
        cell.z = 2.0  # type: ignore[misc]


def test_cell_uses_slots() -> None:
    assert hasattr(Cell, "__slots__")
    assert "row" in Cell.__slots__
    assert "column" in Cell.__slots__
