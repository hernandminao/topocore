from __future__ import annotations

import numpy as np

from topocore.io.common.records import PointRecordBatch
from topocore.io.ply.converter import PLYConverter
from topocore.pointcloud.attributes import PointAttribute


class FakeChunk(dict):
    def __init__(self, attributes):
        super().__init__(attributes)

    def has_attribute(self, attribute):
        return attribute in self


def rgb_batch():
    return PointRecordBatch(
        arrays={
            "red": np.array([10, 20], dtype=np.uint8),
            "green": np.array([30, 40], dtype=np.uint8),
            "blue": np.array([50, 60], dtype=np.uint8),
        }
    )


def normal_batch():
    return PointRecordBatch(
        arrays={
            "nx": np.array([1.0, 2.0], dtype=np.float32),
            "ny": np.array([3.0, 4.0], dtype=np.float32),
            "nz": np.array([5.0, 6.0], dtype=np.float32),
        }
    )


def test_attribute_mapping():

    converter = PLYConverter()

    mapping = converter.attribute_mapping

    assert mapping["red"] is PointAttribute.COLOR
    assert mapping["green"] is PointAttribute.COLOR
    assert mapping["blue"] is PointAttribute.COLOR

    assert mapping["nx"] is PointAttribute.NORMAL
    assert mapping["ny"] is PointAttribute.NORMAL
    assert mapping["nz"] is PointAttribute.NORMAL


def test_populate_color():

    converter = PLYConverter()

    chunk = FakeChunk(
        {
            PointAttribute.COLOR: np.zeros(
                (2, 3),
                dtype=np.uint8,
            )
        }
    )

    converter._populate_color(
        chunk,
        rgb_batch(),
    )

    np.testing.assert_array_equal(
        chunk[PointAttribute.COLOR],
        np.array(
            [
                [10, 30, 50],
                [20, 40, 60],
            ],
            dtype=np.uint8,
        ),
    )


def test_populate_color_missing_component():

    converter = PLYConverter()

    chunk = FakeChunk(
        {
            PointAttribute.COLOR: np.zeros(
                (2, 3),
                dtype=np.uint8,
            )
        }
    )

    batch = PointRecordBatch(
        arrays={
            "red": np.array([1, 2]),
            "green": np.array([3, 4]),
        }
    )

    before = chunk[PointAttribute.COLOR].copy()

    converter._populate_color(
        chunk,
        batch,
    )

    np.testing.assert_array_equal(
        chunk[PointAttribute.COLOR],
        before,
    )


def test_populate_color_without_attribute():

    converter = PLYConverter()

    chunk = FakeChunk({})

    converter._populate_color(
        chunk,
        rgb_batch(),
    )

    assert PointAttribute.COLOR not in chunk


def test_populate_normals():

    converter = PLYConverter()

    chunk = FakeChunk(
        {
            PointAttribute.NORMAL: np.zeros(
                (2, 3),
                dtype=np.float32,
            )
        }
    )

    converter._populate_normals(
        chunk,
        normal_batch(),
    )

    np.testing.assert_array_equal(
        chunk[PointAttribute.NORMAL],
        np.array(
            [
                [1, 3, 5],
                [2, 4, 6],
            ],
            dtype=np.float32,
        ),
    )


def test_populate_normals_missing_component():

    converter = PLYConverter()

    chunk = FakeChunk(
        {
            PointAttribute.NORMAL: np.zeros(
                (2, 3),
                dtype=np.float32,
            )
        }
    )

    batch = PointRecordBatch(
        arrays={
            "nx": np.array([1, 2]),
            "ny": np.array([3, 4]),
        }
    )

    before = chunk[PointAttribute.NORMAL].copy()

    converter._populate_normals(
        chunk,
        batch,
    )

    np.testing.assert_array_equal(
        chunk[PointAttribute.NORMAL],
        before,
    )


def test_populate_normals_without_attribute():

    converter = PLYConverter()

    chunk = FakeChunk({})

    converter._populate_normals(
        chunk,
        normal_batch(),
    )

    assert PointAttribute.NORMAL not in chunk


def test_populate_special_attributes(monkeypatch):

    converter = PLYConverter()

    called = {
        "color": False,
        "normal": False,
    }

    def fake_color(chunk, batch):
        called["color"] = True

    def fake_normal(chunk, batch):
        called["normal"] = True

    monkeypatch.setattr(
        converter,
        "_populate_color",
        fake_color,
    )

    monkeypatch.setattr(
        converter,
        "_populate_normals",
        fake_normal,
    )

    converter._populate_special_attributes(
        FakeChunk({}),
        PointRecordBatch(arrays={}),
    )

    assert called["color"]
    assert called["normal"]