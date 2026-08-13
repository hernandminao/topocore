"""
topocore.io.landxml._xml_utils
================================

Small private helpers shared by ``reader.py``, ``writer.py`` and
``validation.py``. Not part of the public API.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from xml.etree.ElementTree import Element


def local_tag(element: Element) -> str:
    """
    Strip the XML namespace from a tag, e.g.
    ``{http://www.landxml.org/schema/LandXML-1.2}Surface`` -> ``Surface``.
    """
    tag = element.tag
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def children(element: Element, tag: str) -> list[Element]:
    """
    Direct children of ``element`` whose local (namespace-stripped)
    tag equals ``tag``.
    """
    return [child for child in element if local_tag(child) == tag]


__all__ = [
    "children",
    "local_tag",
]
