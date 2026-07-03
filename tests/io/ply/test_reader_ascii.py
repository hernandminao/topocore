from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from tests.io.ply.helpers import FakeHeader, write_ascii_ply
from topocore.io.ply.enums import PLYFormat
from topocore.io.ply.exceptions import InvalidPLYError
from topocore.io.ply.header import PLYElement
from topocore.io.ply.reader import PLYReader


class TestPLYReaderAsciiValidation:
    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            PLYReader("dummy.ply", chunk_size=0)

    def test_vertex_requires_header(self) -> None:
        reader = PLYReader("dummy.ply")
        reader._stream = io.BytesIO(b"")

        with pytest.raises(
            InvalidPLYError,
            match="header not loaded",
        ):
            reader._vertex_element()

    def test_missing_vertex_element(self) -> None:
        reader = PLYReader("dummy.ply")
        reader._header = PLYHeaderStubWithoutVertex()

        with pytest.raises(
            InvalidPLYError,
            match="no vertex element",
        ):
            reader._vertex_element()

    def test_empty_elements_raises(self) -> None:
        reader = PLYReader("dummy.ply")
        reader._header = type(
            "Header",
            (),
            {"elements": []},
        )()

        with pytest.raises(InvalidPLYError):
            reader._vertex_element()


class PLYHeaderStubWithoutVertex:
    elements = [
        PLYElement(name="face", count=1, properties=[]),
    ]


class TestPLYReaderAsciiIteration:
    def test_ascii_iteration_with_mock_header(
        self,
        identity_converter,
    ) -> None:
        header = FakeHeader(PLYFormat.ASCII)
        reader = PLYReader("dummy.ply")
        reader._stream = io.BytesIO(b"1 2 3\n4 5 6\n7 8 9\n")
        reader._header = header
        identity_converter(reader)

        chunks = list(reader._iter_ascii(header.elements[0]))

        assert len(chunks) >= 1

    def test_ascii_skips_empty_lines(
        self,
        identity_converter,
    ) -> None:
        header = FakeHeader(PLYFormat.ASCII)
        reader = PLYReader("dummy.ply")
        reader._header = header
        reader._stream = io.BytesIO(b"\n\n1 2 3\n")
        identity_converter(reader)

        chunks = list(reader._iter_ascii(header.elements[0]))

        assert len(chunks) == 1

    def test_ascii_utf8_decode_error(self) -> None:
        header = FakeHeader(PLYFormat.ASCII)
        reader = PLYReader("dummy.ply")
        reader._header = header
        reader._stream = io.BytesIO(b"\xff\xff\xff\n")

        with pytest.raises(
            InvalidPLYError,
            match="not valid UTF-8",
        ):
            next(reader._iter_ascii(header.elements[0]))

    def test_ascii_chunk_flush_on_boundary(
        self,
        identity_converter,
    ) -> None:
        header = FakeHeader(PLYFormat.ASCII, vertex_count=4)
        reader = PLYReader("dummy.ply", chunk_size=2)
        reader._header = header
        reader._stream = io.BytesIO(
            b"1 2 3\n4 5 6\n7 8 9\n10 11 12\n",
        )
        identity_converter(reader)

        chunks = list(reader._iter_ascii(header.elements[0]))

        assert len(chunks) == 2

    def test_flush_ascii_skips_list_properties(
        self,
        identity_converter,
    ) -> None:
        header = FakeHeader(
            PLYFormat.ASCII,
            include_list_property=True,
        )
        reader = PLYReader("dummy.ply")
        reader._header = header
        reader._stream = io.BytesIO(b"1 2 3\n")
        identity_converter(reader)

        chunks = list(reader._iter_ascii(header.elements[0]))

        assert len(chunks) == 1


class TestPLYReaderAsciiIntegration:
    @patch("topocore.io.ply.reader.PLYHeaderParser.parse")
    def test_iter_closes_stream(
        self,
        mock_parse,
        identity_converter,
    ) -> None:
        header = FakeHeader(PLYFormat.ASCII)
        mock_parse.return_value = header

        reader = PLYReader("dummy.ply")
        reader._stream = io.BytesIO(b"1 2 3\n")
        reader._header = header
        identity_converter(reader)

        list(reader)

        assert reader._stream is None

    def test_read_from_file_opens_stream(
        self,
        tmp_path,
        identity_converter,
    ) -> None:
        ply_path = tmp_path / "points.ply"
        write_ascii_ply(
            ply_path,
            [
                (1.0, 2.0, 3.0),
                (4.0, 5.0, 6.0),
            ],
        )

        reader = PLYReader(ply_path, chunk_size=1)
        identity_converter(reader)

        chunks = list(reader)

        assert len(chunks) == 2
        assert reader._stream is None

    def test_open_is_idempotent(self, tmp_path) -> None:
        ply_path = tmp_path / "points.ply"
        write_ascii_ply(ply_path, [(0.0, 0.0, 0.0)])

        reader = PLYReader(ply_path)
        reader._open()
        stream = reader._stream
        header = reader._header

        reader._open()

        assert reader._stream is stream
        assert reader._header is header

        reader.close()