"""
Regression tests for topocore.analysis.profile.writer -- a CSV
export helper for the already-existing ProfileResult/ProfilePoint
types, with no new domain model introduced.
"""
from __future__ import annotations

import csv

import pytest
from topocore.analysis.profile.writer import ProfileWriteError, write_profile_csv
from topocore.analysis.types import ProfilePoint, ProfileResult, ProfileType


def _profile(points: list[ProfilePoint]) -> ProfileResult:
    return ProfileResult(points=points, profile_type=ProfileType.TRANSVERSAL)


def test_writes_the_exact_agreed_column_contract(tmp_path) -> None:
    profile = _profile([
        ProfilePoint(station=0.0, x=0.0, y=2.0, z=10.1, offset=-8.0),
        ProfilePoint(station=0.0, x=0.0, y=10.0, z=10.5, offset=0.0),
    ])

    output = tmp_path / "sections.csv"
    write_profile_csv([profile], output)

    with open(output, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["station", "offset", "x", "y", "elevation"]
    assert rows[1] == ["0.0", "-8.0", "0.0", "2.0", "10.1"]
    assert rows[2] == ["0.0", "0.0", "0.0", "10.0", "10.5"]


def test_profilepoint_z_maps_to_the_elevation_column_without_renaming_the_model() -> None:
    """
    Confirmed design decision: ProfilePoint keeps its own field name
    (z) unchanged -- only the CSV writer's own output column is
    named "elevation", which is clearer for a file a person opens
    directly than the internal API name.
    """
    point = ProfilePoint(station=1.0, x=2.0, y=3.0, z=99.5, offset=0.0)
    assert point.z == 99.5
    assert not hasattr(point, "elevation")


def test_multiple_profiles_are_concatenated_in_the_order_given(tmp_path) -> None:
    first = _profile([ProfilePoint(station=0.0, x=0.0, y=0.0, z=1.0)])
    second = _profile([ProfilePoint(station=20.0, x=20.0, y=0.0, z=2.0)])

    output = tmp_path / "sections.csv"
    write_profile_csv([first, second], output)

    with open(output, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[1][0] == "0.0"
    assert rows[2][0] == "20.0"


def test_an_empty_profile_list_still_writes_a_valid_header_only_csv(tmp_path) -> None:
    output = tmp_path / "empty.csv"
    result_path = write_profile_csv([], output)

    assert result_path == output
    with open(output, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows == [["station", "offset", "x", "y", "elevation"]]


def test_a_profile_with_no_points_contributes_no_rows(tmp_path) -> None:
    empty_profile = _profile([])
    non_empty_profile = _profile([ProfilePoint(station=0.0, x=0.0, y=0.0, z=1.0)])

    output = tmp_path / "sections.csv"
    write_profile_csv([empty_profile, non_empty_profile], output)

    with open(output, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2  # header + the single real point


def test_an_unwritable_path_raises_a_specific_error() -> None:
    profile = _profile([ProfilePoint(station=0.0, x=0.0, y=0.0, z=1.0)])

    with pytest.raises(ProfileWriteError):
        write_profile_csv([profile], "/this/path/cannot/possibly/exist/sections.csv")


def test_returns_the_path_it_was_given(tmp_path) -> None:
    output = tmp_path / "sections.csv"
    result = write_profile_csv([], output)
    assert result == output
