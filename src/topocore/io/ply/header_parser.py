"""
topocore.io.ply.header_parser
=============================

PLY header parser.

This module parses the header of ASCII and binary PLY files and builds
the corresponding domain model.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BufferedReader

from .enums import PLYFormat, PLYScalarType
from .exceptions import InvalidPLYError
from .header import (
    PLYElement,
    PLYHeader,
    PLYListProperty,
    PLYProperty,
)

_MAGIC = "ply"

_HEADER_END = "end_header"

_SUPPORTED_VERSIONS = frozenset(
    {
        "1.0",
    }
)


@dataclass(slots=True)
class _HeaderParseState:
    """
    Mutable parser state.
    """

    fmt: PLYFormat | None = None

    version: str | None = None

    comments: list[str] = field(
        default_factory=list,
    )

    obj_info: list[str] = field(
        default_factory=list,
    )

    elements: list[PLYElement] = field(
        default_factory=list,
    )

    current: PLYElement | None = None


class PLYHeaderParser:
    """
    Parses a PLY header.
    """

    @classmethod
    def parse(
        cls,
        stream: BufferedReader,
    ) -> PLYHeader:
        """
        Parse a PLY header.
        """

        lines: list[str] = []

        header_size = 0

        while True:
            raw = stream.readline()

            if raw == b"":
                raise InvalidPLYError("Unexpected end of file while reading PLY header.")

            header_size += len(raw)

            try:
                line = raw.decode("utf-8").strip()

            except UnicodeDecodeError as exc:
                raise InvalidPLYError("PLY header is not valid UTF-8.") from exc

            lines.append(
                line,
            )

            if line == _HEADER_END:
                break

        return cls._build_header(
            lines=lines,
            header_size=header_size,
        )

    @classmethod
    def _build_header(
        cls,
        lines: list[str],
        header_size: int,
    ) -> PLYHeader:
        """
        Build a header model.
        """

        cls._validate_magic(
            lines,
        )

        state = _HeaderParseState()

        for line in lines[1:]:
            if not line:
                continue

            tokens = line.split()

            keyword = tokens[0]

            if keyword == _HEADER_END:
                break

            cls._process_line(
                keyword=keyword,
                tokens=tokens,
                line=line,
                state=state,
            )

        cls._validate_elements(
            state,
        )

        if state.fmt is None:
            raise InvalidPLYError("PLY header does not declare a format.")

        if state.version is None:
            raise InvalidPLYError("PLY header does not declare a version.")

        return PLYHeader(
            format=state.fmt,
            version=state.version,
            elements=state.elements,
            comments=state.comments,
            obj_info=state.obj_info,
            header_size=header_size,
        )

    @classmethod
    def _process_line(
        cls,
        keyword: str,
        tokens: list[str],
        line: str,
        state: _HeaderParseState,
    ) -> None:
        """
        Process a single header line.
        """

        match keyword:
            case "format":
                cls._handle_format(
                    tokens=tokens,
                    state=state,
                )

            case "comment":
                state.comments.append(
                    line[8:].strip(),
                )

            case "obj_info":
                state.obj_info.append(
                    line[9:].strip(),
                )

            case "element":
                cls._handle_element(
                    tokens=tokens,
                    state=state,
                )

            case "property":
                cls._handle_property(
                    tokens=tokens,
                    state=state,
                )

            case _:
                raise InvalidPLYError(f"Unknown PLY header keyword '{keyword}'.")

    @classmethod
    def _handle_format(
        cls,
        tokens: list[str],
        state: _HeaderParseState,
    ) -> None:
        """
        Handle a format declaration.

        Expected format::

            format <encoding> <version>
        """

        if len(tokens) != 3:
            raise InvalidPLYError("Malformed format declaration.")

        if state.fmt is not None:
            raise InvalidPLYError("Duplicate format declaration.")

        try:
            fmt = PLYFormat(tokens[1])

        except ValueError as exc:
            raise InvalidPLYError(f"Unsupported PLY format '{tokens[1]}'.") from exc

        version = tokens[2]

        if version not in _SUPPORTED_VERSIONS:
            raise InvalidPLYError(f"Unsupported PLY version '{version}'.")

        state.fmt = fmt
        state.version = version

    @classmethod
    def _handle_element(
        cls,
        tokens: list[str],
        state: _HeaderParseState,
    ) -> None:
        """
        Handle an element declaration.

        Expected format::

            element <name> <count>
        """

        if len(tokens) != 3:
            raise InvalidPLYError("Malformed element declaration.")

        name = tokens[1]

        try:
            count = int(tokens[2])

        except ValueError as exc:
            raise InvalidPLYError(f"Invalid element count '{tokens[2]}'.") from exc

        if count < 0:
            raise InvalidPLYError("Element count cannot be negative.")

        if any(element.name == name for element in state.elements):
            raise InvalidPLYError(f"Duplicated element '{name}'.")

        element = PLYElement(
            name=name,
            count=count,
            properties=[],
        )

        state.current = element
        state.elements.append(element)

    @classmethod
    def _handle_property(
        cls,
        tokens: list[str],
        state: _HeaderParseState,
    ) -> None:
        """
        Handle a property declaration.
        """

        if state.current is None:
            raise InvalidPLYError("Property declared before an element.")

        property_ = cls._parse_property(
            tokens,
        )

        if any(existing.name == property_.name for existing in state.current.properties):
            raise InvalidPLYError(f"Duplicated property '{property_.name}' in element '{state.current.name}'.")

        state.current.properties.append(
            property_,
        )

    @classmethod
    def _parse_property(
        cls,
        tokens: list[str],
    ) -> PLYProperty | PLYListProperty:

        if len(tokens) < 2:
            raise InvalidPLYError("Malformed property declaration.")

        if tokens[1] == "list":
            return cls._parse_list_property(
                tokens,
            )

        return cls._parse_scalar_property(
            tokens,
        )

    @staticmethod
    def _parse_scalar_property(
        tokens: list[str],
    ) -> PLYProperty:

        if len(tokens) != 3:
            raise InvalidPLYError("Malformed scalar property declaration.")

        try:
            dtype = PLYScalarType(
                tokens[1],
            )

        except ValueError as exc:
            raise InvalidPLYError(f"Unsupported scalar type '{tokens[1]}'.") from exc

        return PLYProperty(
            name=tokens[2],
            dtype=dtype,
        )

    @staticmethod
    def _parse_list_property(
        tokens: list[str],
    ) -> PLYListProperty:

        if len(tokens) != 5:
            raise InvalidPLYError("Malformed list property declaration.")

        try:
            count_type = PLYScalarType(
                tokens[2],
            )

            value_type = PLYScalarType(
                tokens[3],
            )

        except ValueError as exc:
            raise InvalidPLYError("Unsupported list property type.") from exc

        return PLYListProperty(
            name=tokens[4],
            count_type=count_type,
            value_type=value_type,
        )

    @staticmethod
    def _validate_magic(
        lines: list[str],
    ) -> None:
        """
        Validate the PLY magic string.
        """

        if not lines:
            raise InvalidPLYError("Empty PLY file.")

        if lines[0].strip() != _MAGIC:
            raise InvalidPLYError("Invalid PLY signature.")

    @staticmethod
    def _validate_elements(
        state: _HeaderParseState,
    ) -> None:
        """
        Validate parsed elements.
        """

        if not state.elements:
            raise InvalidPLYError("PLY header contains no elements.")
