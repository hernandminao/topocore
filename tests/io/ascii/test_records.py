"""Regression tests for topocore.io.ascii.records."""

from __future__ import annotations

import numpy as np

from topocore.io.ascii.records import ASCIIRecordBatch


def test_empty_batch() -> None:
    batch = ASCIIRecordBatch()

    assert batch.size == 0
    assert len(batch) == 0
    assert list(batch) == []
    assert list(batch.keys()) == []
    assert list(batch.values()) == []
    assert list(batch.items()) == []


def test_batch_size_is_number_of_records() -> None:
    batch = ASCIIRecordBatch(
        columns={
            "x": np.array([1.0, 2.0, 3.0]),
            "z": np.array([10.0, 20.0, 30.0]),
        }
    )

    assert batch.size == 3
    assert len(batch) == 3


def test_batch_contains_and_getitem() -> None:
    x = np.array([1.0, 2.0])

    batch = ASCIIRecordBatch(
        columns={"x": x},
    )

    assert "x" in batch
    assert "z" not in batch

    result = batch["x"]

    assert result is x
    np.testing.assert_array_equal(result, [1.0, 2.0])


def test_batch_get_existing_value() -> None:
    x = np.array([1.0, 2.0])

    batch = ASCIIRecordBatch(
        columns={"x": x},
    )

    assert batch.get("x") is x


def test_batch_get_missing_returns_none() -> None:
    batch = ASCIIRecordBatch(
        columns={
            "x": np.array([1.0]),
        }
    )

    assert batch.get("missing") is None


def test_batch_get_missing_returns_default() -> None:
    batch = ASCIIRecordBatch(
        columns={
            "x": np.array([1.0]),
        }
    )

    default = np.array([-1.0])

    assert batch.get("missing", default) is default


def test_batch_mapping_views() -> None:
    x = np.array([1.0, 2.0])
    z = np.array([3.0, 4.0])

    batch = ASCIIRecordBatch(
        columns={
            "x": x,
            "z": z,
        }
    )

    assert list(batch.keys()) == ["x", "z"]

    values = list(batch.values())

    assert values[0] is x
    assert values[1] is z

    items = list(batch.items())

    assert items[0][0] == "x"
    assert items[0][1] is x

    assert items[1][0] == "z"
    assert items[1][1] is z


def test_batch_iteration_returns_column_names() -> None:
    batch = ASCIIRecordBatch(
        columns={
            "x": np.array([1.0]),
            "y": np.array([2.0]),
            "z": np.array([3.0]),
        }
    )

    assert list(batch) == ["x", "y", "z"]


def test_batch_preserves_numpy_dtypes() -> None:
    classification = np.array(
        [1, 2, 5],
        dtype=np.uint8,
    )

    batch = ASCIIRecordBatch(
        columns={
            "classification": classification,
        }
    )

    assert batch["classification"].dtype == np.dtype(np.uint8)


def test_batch_uses_declared_columns_mapping() -> None:
    columns = {
        "x": np.array([1.0]),
        "classification": np.array([2], dtype=np.uint8),
    }

    batch = ASCIIRecordBatch(columns=columns)

    assert batch.columns is columns
