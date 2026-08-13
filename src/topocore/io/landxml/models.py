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
class LandXMLDocument:
    """
    Root container for a LandXML file's supported content.

    Only ``<Surfaces>`` and ``<CgPoints>`` are represented -- see the
    PR18B contract. ``<Alignments>``, ``<Profile>`` and ``<Feature>``
    (embedded in ``<CgPoint>``) are out of scope: TopoCore has no
    domain model for them yet, and this subsystem does not invent
    one just to round-trip LandXML.
    """

    surfaces: tuple[NamedSurface, ...] = ()
    point_groups: tuple[NamedPointGroup, ...] = ()
    crs: str | None = None
    linear_unit: LinearUnit = LinearUnit.METER


__all__ = [
    "LandXMLDocument",
    "LinearUnit",
    "NamedPointGroup",
    "NamedSurface",
]
