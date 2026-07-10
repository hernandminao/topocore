"""
topocore.io.ply.header
======================

PLY header domain model.

This module contains immutable models describing the structure of a
PLY file. Parsing logic is intentionally implemented elsewhere.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import PLYFormat, PLYScalarType


@dataclass(slots=True, frozen=True)
class PLYProperty:
    """
    Scalar property definition.

    Example
    -------
    property float x
    """

    name: str
    dtype: PLYScalarType


@dataclass(slots=True, frozen=True)
class PLYListProperty:
    """
    List property definition.

    Example
    -------
    property list uchar int vertex_indices
    """

    name: str
    count_type: PLYScalarType
    value_type: PLYScalarType


PLYPropertyDefinition = PLYProperty | PLYListProperty


@dataclass(slots=True)
class PLYElement:
    """
    Element definition.

    Example
    -------
    element vertex 1500000
    """

    name: str

    count: int

    properties: list[PLYPropertyDefinition] = field(default_factory=list)

    @property
    def property_names(
        self,
    ) -> tuple[str, ...]:
        """
        Return property names.
        """
        return tuple(property.name for property in self.properties)

    def has_property(
        self,
        name: str,
    ) -> bool:
        """
        Return whether the element contains the property.
        """
        return any(property.name == name for property in self.properties)

    def get_property(
        self,
        name: str,
    ) -> PLYPropertyDefinition | None:
        """
        Return a property by name.
        """
        for property in self.properties:
            if property.name == name:
                return property

        return None


@dataclass(slots=True)
class PLYHeader:
    """
    Parsed PLY header.
    """

    format: PLYFormat

    version: str

    elements: list[PLYElement]

    comments: list[str] = field(default_factory=list)

    obj_info: list[str] = field(default_factory=list)

    header_size: int = 0

    @property
    def vertex_element(
        self,
    ) -> PLYElement | None:
        """
        Return the vertex element.
        """
        for element in self.elements:
            if element.name == "vertex":
                return element

        return None

    @property
    def face_element(
        self,
    ) -> PLYElement | None:
        """
        Return the face element.
        """
        for element in self.elements:
            if element.name == "face":
                return element

        return None

    @property
    def vertex_count(
        self,
    ) -> int:
        """
        Number of vertices.
        """
        vertex = self.vertex_element

        if vertex is None:
            return 0

        return vertex.count

    def has_element(
        self,
        name: str,
    ) -> bool:
        """
        Return whether an element exists.
        """
        return any(element.name == name for element in self.elements)

    def get_element(
        self,
        name: str,
    ) -> PLYElement | None:
        """
        Return an element by name.
        """
        for element in self.elements:
            if element.name == name:
                return element

        return None


__all__ = [
    "PLYHeader",
    "PLYElement",
    "PLYProperty",
    "PLYListProperty",
    "PLYPropertyDefinition",
]
