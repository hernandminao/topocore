"""
topocore.io.landxml.models
===========================

Domain models for the TopoCore LandXML IO subsystem.

Deliberately thin: ``TIN`` and ``SurveyPointSet`` (both frozen
modules -- ``topocore.terrain``, ``topocore.survey``) already carry
all the geometry. Neither has a ``name`` field, and per the project's
architectural rules neither should gain one just for this format --
so ``NamedSurface``/``NamedPointGroup`` exist purely to carry the
``name``/``desc`` metadata a LandXML ``<Surface>``/``<CgPoints>``
collection has, without duplicating or wrapping the domain objects'
own behavior.

Per rule G-001, coordinate reference system information is exposed
only at the document level (``LandXMLDocument.crs``), never injected
into ``TIN`` or ``SurveyPointSet`` -- consistent with how
``topocore.dxf.ExportContext.crs`` is kept separate from
``FeatureCollection``.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from topocore.alignment.models import Alignment, DesignProfile
from topocore.survey.models import SurveyPointSet
from topocore.terrain.tin import TIN


class LinearUnit(StrEnum):
    """
    LandXML ``<Units>`` linear unit.

    LandXML expresses this via a child element name
    (``<Metric linearUnit="meter">`` vs
    ``<Imperial linearUnit="foot">``), not a single attribute value
    -- ``reader.py``/``writer.py`` handle that mapping; this enum is
    just the resolved unit.
    """

    METER = "meter"
    FOOT = "foot"


@dataclass(frozen=True, slots=True)
class NamedSurface:
    """
    A LandXML ``<Surface>`` (TIN definition), with its name/desc.

    Parameters
    ----------
    name
        The ``<Surface name="...">`` attribute. Must be unique
        within a document's surface collection (validated by
        ``LandXMLValidator``).
    tin
        The surface geometry, built via ``TIN.from_mesh()`` so the
        exact ``<Faces>`` connectivity from the file is preserved.
    desc
        The ``<Surface desc="...">`` attribute, if present.
    """

    name: str
    tin: TIN
    desc: str | None = None


@dataclass(frozen=True, slots=True)
class NamedPointGroup:
    """
    A LandXML ``<CgPoints>`` collection, with its name/desc.

    Parameters
    ----------
    name
        The ``<CgPoints name="...">`` attribute. Must be unique
        within a document's point-group collection (validated by
        ``LandXMLValidator``).
    points
        The contained points, in file order (``SurveyPointSet``
        order is meaningful -- see ``topocore.survey.models``).
    desc
        The ``<CgPoints desc="...">`` attribute, if present.
    """

    name: str
    points: SurveyPointSet
    desc: str | None = None


@dataclass(frozen=True, slots=True)
class NamedAlignment:
    """
    A LandXML ``<Alignment>``, with its name/desc and optional
    vertical design profile.

    Parameters
    ----------
    name
        The ``<Alignment name="...">`` attribute. Must be unique
        within a document's alignment collection (validated by
        ``LandXMLValidator``).
    alignment
        The horizontal geometry, built from ``<CoordGeom>``'s
        ``<Line>``/``<Curve>``/``<Spiral>`` sequence.
    profile
        The vertical design profile, if the ``<Alignment>`` has a
        ``<Profile>``/``<ProfAlign>``. ``None`` if horizontal-only.
    desc
        The ``<Alignment desc="...">`` attribute, if present.
    """

    name: str
    alignment: Alignment
    profile: DesignProfile | None = None
    desc: str | None = None


@dataclass(frozen=True, slots=True)
class LandXMLDocument:
    """
    Root container for a LandXML file's supported content.

    ``<Surfaces>``, ``<CgPoints>``, and ``<Alignments>`` (horizontal
    ``<CoordGeom>`` plus vertical ``<Profile>``) are represented.
    ``<Feature>`` embedded in ``<CgPoint>`` remains out of scope --
    TopoCore has no domain model for it yet, and this subsystem does
    not invent one just to round-trip LandXML.
    """

    surfaces: tuple[NamedSurface, ...] = ()
    point_groups: tuple[NamedPointGroup, ...] = ()
    alignments: tuple[NamedAlignment, ...] = ()
    crs: str | None = None
    linear_unit: LinearUnit = LinearUnit.METER


__all__ = [
    "LandXMLDocument",
    "LinearUnit",
    "NamedAlignment",
    "NamedPointGroup",
    "NamedSurface",
]
