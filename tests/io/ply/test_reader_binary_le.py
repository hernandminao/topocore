from __future__ import annotations

from unittest.mock import patch

import numpy as np

from tests.io.ply.helpers import FakeHeader, write_binary_ply
from topocore.io.ply.enums import PLYFormat
from topocore.io.ply.reader import PLYReader


class TestPLYReaderBinaryBigEndian:
    @patch("numpy.fromfile")
    def test_binary_big_endian_iteration(
        self,
        mock_fromfile,
        identity_converter,
    ) -> None:
        mock_fromfile.return_value = np.array(
            [(1.0, 2.0, 3.0)],
            dtype=[("x", ">f4"), ("y", ">f4"), ("z", ">f4")],
        )

        header = FakeHeader(
            PLYFormat.BINARY_BIG_ENDIAN,
            vertex_count=1,
        )
        reader = PLYReader("dummy.ply")
        reader._header = header
        identity_converter(reader)

        chunks = list(reader._iter_binary(header.elements[0]))

        assert len(chunks) == 1

    def test_iter_routes_to_binary_big_endian(
        self,
        tmp_path,
        identity_converter,
    ) -> None:
        ply_path = tmp_path / "binary_be.ply"
        write_binary_ply(
            ply_path,
            [
                (1.0, 2.0, 3.0),
                (4.0, 5.0, 6.0),
            ],
            fmt=PLYFormat.BINARY_BIG_ENDIAN,
        )

        reader = PLYReader(ply_path, chunk_size=1)
        identity_converter(reader)

        chunks = list(reader)

        assert len(chunks) == 2

    def test_big_endian_dtype_prefix(
        self,
        identity_converter,
    ) -> None:
        header = FakeHeader(
            PLYFormat.BINARY_BIG_ENDIAN,
            vertex_count=1,
        )
        reader = PLYReader("dummy.ply")
        reader._header = header

        captured: dict[str, object] = {}

        def fake_fromfile(stream, dtype, count):  # noqa: ANN001
            captured["dtype"] = dtype
            return np.array(
                [(1.0, 2.0, 3.0)],
                dtype=dtype,
            )

        with patch(
            "numpy.fromfile",
            side_effect=fake_fromfile,
        ):
            identity_converter(reader)
            list(reader._iter_binary(header.elements[0]))

        dtype = captured["dtype"]
        assert isinstance(dtype, np.dtype)
        assert dtype.fields is not None
        assert dtype["x"].str.startswith(">")
