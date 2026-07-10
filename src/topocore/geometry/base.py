"""
topocore.geometry.base
======================

Defines the abstract base class for all TopoCore geometry objects.

This module provides the minimal interface shared by all geometry
primitives. It intentionally contains no geometric algorithms or
operations such as distance, area, centroid or intersection.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Geometry(ABC):
    """
    Abstract base class for all geometry objects.

    Notes
    -----
    This class only defines the common infrastructure shared by
    concrete geometry classes.

    Subclasses must implement :meth:`to_dict`.
    """

    __slots__ = ()

    @abstractmethod
    def to_dict(self) -> dict[str, object]:
        """
        Return a serializable representation of the geometry.

        Returns
        -------
        dict[str, object]
            Dictionary containing the geometry attributes.
        """

    def __repr__(self) -> str:
        """
        Return a developer-friendly representation.

        Returns
        -------
        str
            String representation including the class name and
            attribute values.
        """
        attributes = ", ".join(f"{key}={value!r}" for key, value in self.to_dict().items())

        return f"{self.__class__.__name__}({attributes})"


__all__ = [
    "Geometry",
]
