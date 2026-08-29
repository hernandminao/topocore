"""
Coverage audit tests for topocore.io.ascii.mapper.ColumnMapper.

COLUMN-MAPPER-001 (FIXED in this PR): when two source columns
normalized to the same target key (e.g. "X" -> "x" via identity, and
"Easting" -> "x" via DEFAULT_MAPPING), normalize()'s
`mapped[target] = values` previously silently OVERWROTE the earlier
entry with the later one (in dict insertion order) -- no warning,
error, or indication that data was discarded. Confirmed directly
with real code before this fix.

Decision made explicitly (matching TopoCore's own "fail fast, no
silent data loss" philosophy, applied consistently with
SAMPLING-MANAGER-001's own resolution): rather than picking a winner,
merging, or auto-disambiguating (any of which would make a semantic
decision ColumnMapper has no basis for), an ambiguous file is now
explicitly rejected. A new `ColumnMappingError` (matching this
module's existing `ASCIIError` subclass naming convention:
`InvalidASCIIRecordError`, `MissingColumnError`,
`UnsupportedDelimiterError`) is raised, naming every colliding source
column and its shared target.

Collisions are detected in a first pass BEFORE any part of the
result is built -- confirmed directly that the original input batch
is left completely unmodified when the error is raised (no partial
result is ever returned).

Other cases (empty batch, whitespace/underscore/hyphen
normalization, unrecognized columns falling through to their own
normalized name) remain correct and unaffected by this change.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.io.ascii.exceptions import ColumnMappingError
from topocore.io.ascii.mapper import ColumnMapper
from topocore.io.ascii.records import ASCIIRecordBatch


def test_normalize_empty_batch_returns_empty_batch() -> None:
    result = ColumnMapper.normalize(ASCIIRecordBatch({}))
    assert list(result.keys()) == []
    assert result.size == 0


def test_normalize_maps_known_aliases_to_canonical_names() -> None:
    batch = ASCIIRecordBatch(
        {
            "Easting": np.array([1.0, 2.0]),
            "Northing": np.array([3.0, 4.0]),
            "Elevation": np.array([5.0, 6.0]),
        }
    )
    result = ColumnMapper.normalize(batch)

    assert set(result.keys()) == {"x", "y", "z"}
    np.testing.assert_array_equal(result["x"], [1.0, 2.0])
    np.testing.assert_array_equal(result["y"], [3.0, 4.0])
    np.testing.assert_array_equal(result["z"], [5.0, 6.0])


def test_normalize_strips_spaces_underscores_and_hyphens_case_insensitively() -> None:
    for column_name in ["Point ID", "POINT_ID", "point-id", "PointId"]:
        batch = ASCIIRecordBatch({column_name: np.array([1, 2])})
        result = ColumnMapper.normalize(batch)
        assert list(result.keys()) == ["id"], f"failed for column name {column_name!r}"


def test_normalize_preserves_unrecognized_columns_under_their_normalized_name() -> None:
    batch = ASCIIRecordBatch({"Custom Field": np.array([9, 9])})
    result = ColumnMapper.normalize(batch)
    assert list(result.keys()) == ["customfield"]


def test_normalize_colliding_columns_raises_column_mapping_error() -> None:
    """
    The core regression: "X" and "Easting" both normalize to "x" --
    this must now be explicitly rejected, not silently resolved by
    picking a winner.
    """
    batch = ASCIIRecordBatch(
        {
            "X": np.array([1.0, 2.0, 3.0]),
            "Easting": np.array([100.0, 200.0, 300.0]),
        }
    )

    with pytest.raises(ColumnMappingError, match="collision"):
        ColumnMapper.normalize(batch)


def test_normalize_collision_error_names_the_conflicting_columns() -> None:
    batch = ASCIIRecordBatch(
        {
            "X": np.array([1.0]),
            "Easting": np.array([2.0]),
        }
    )

    with pytest.raises(ColumnMappingError, match=r"'X'.*'Easting'.*'x'|'Easting'.*'X'.*'x'"):
        ColumnMapper.normalize(batch)


def test_normalize_reports_multiple_simultaneous_collisions() -> None:
    batch = ASCIIRecordBatch(
        {
            "X": np.array([1]),
            "Easting": np.array([2]),
            "Y": np.array([3]),
            "Northing": np.array([4]),
        }
    )

    with pytest.raises(ColumnMappingError) as exc_info:
        ColumnMapper.normalize(batch)

    message = str(exc_info.value)
    assert "'x'" in message
    assert "'y'" in message


def test_normalize_does_not_partially_build_result_before_raising() -> None:
    """The original input batch must be left completely untouched when a collision is detected."""
    original_columns = {
        "X": np.array([1.0, 2.0, 3.0]),
        "Easting": np.array([100.0, 200.0, 300.0]),
    }
    batch = ASCIIRecordBatch(dict(original_columns))

    with pytest.raises(ColumnMappingError):
        ColumnMapper.normalize(batch)

    assert list(batch.columns.keys()) == ["X", "Easting"]
    np.testing.assert_array_equal(batch.columns["X"], original_columns["X"])
    np.testing.assert_array_equal(batch.columns["Easting"], original_columns["Easting"])
