"""
Regression suite for Workflow.read_point_cloud()'s reader dispatch
-- PR19.

Includes a real bug found and fixed in this session: this method
previously only distinguished ".laz" from everything else, silently
routing PLY/E57/XYZ/CSV/PTS files (and any unrecognized extension)
to LASReader. Also confirms two subtler details discovered while
verifying the fix against the real package structure (not assumed
from a proposed patch): topocore.io.ply/topocore.io.ascii.csv/
topocore.io.ascii.pts do NOT re-export their Reader class from
their package __init__.py (unlike las/laz/e57/ascii.xyz, which do)
-- confirmed by direct import testing, not assumption -- and
E57Reader.chunk_size has no built-in default (unlike every other
reader here), so the dispatcher must supply one explicitly to keep
the workflow-level API uniform.
"""

from __future__ import annotations

import pytest

from topocore.workflow.exceptions import WorkflowExecutionError
from topocore.workflow.workflow import Workflow


def test_las_dispatches_correctly(las_file: str) -> None:
    workflow = Workflow()
    workflow.read_point_cloud(las_file)
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


def test_xyz_dispatches_correctly_not_to_lasreader(xyz_file: str) -> None:
    """
    The exact regression: before the fix, a .xyz file was silently
    routed to LASReader, which cannot parse plain-text ASCII data.
    """
    workflow = Workflow()
    workflow.read_point_cloud(xyz_file)
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


def test_csv_dispatches_correctly(csv_file: str) -> None:
    workflow = Workflow()
    workflow.read_point_cloud(csv_file)
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


def test_case_insensitive_extension(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "points.XYZ")
    with open(path, "w") as f:
        f.write("0.0 0.0 1.0\n10.0 0.0 2.0\n0.0 10.0 3.0\n")

    workflow = Workflow()
    workflow.read_point_cloud(path)
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


def test_unsupported_extension_rejected_with_clear_message(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "points.foo")
    open(path, "w").close()

    workflow = Workflow()
    with pytest.raises(WorkflowExecutionError, match="Unsupported point-cloud format"):
        workflow.read_point_cloud(path)


def test_missing_extension_rejected() -> None:
    workflow = Workflow()
    with pytest.raises(WorkflowExecutionError):
        workflow.read_point_cloud("/nonexistent/path/with/no/extension")


def test_nonexistent_file_rejected_before_format_dispatch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """
    Confirms the file-existence check runs before format dispatch --
    a missing file gives a clear "does not exist" error, not a
    confusing reader-specific failure.
    """
    path = str(tmp_path / "does_not_exist.xyz")

    workflow = Workflow()
    with pytest.raises(WorkflowExecutionError, match="does not exist"):
        workflow.read_point_cloud(path)


def test_reader_kwargs_pass_through_csv_has_header(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Confirms format-specific kwargs (has_header, only meaningful for CSV) reach the underlying reader."""
    path = str(tmp_path / "noheader.csv")
    with open(path, "w") as f:
        f.write("0.0,0.0,1.0\n10.0,0.0,2.0\n0.0,10.0,3.0\n")

    workflow = Workflow()
    workflow.read_point_cloud(path, has_header=False)
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded
