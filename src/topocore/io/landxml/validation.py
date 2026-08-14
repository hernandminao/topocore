"""
topocore.io.landxml.validation
================================

Semantic validation for the TopoCore LandXML IO subsystem.

No external XSD is used (see PR18B contract) -- these are the same
kind of hand-written semantic checks already used by
``topocore.dxf.validation``/``topocore.gpkg.validation``, scoped to
exactly the LandXML 1.2 subset this package reads and writes
(``<Surfaces>``/TIN, ``<CgPoints>``).

``LandXMLValidator`` is usable standalone, independent of
``LandXMLReader``/``LandXMLWriter``:

- ``validate_xml`` runs on the raw parsed tree, before any domain
  object is built -- structural completeness and id-reference
  integrity (a ``<F>`` face referencing a point id absent from its
  ``<Pnts>``), things that can only be checked while ids are still
  strings.
- ``validate_document`` runs on an assembled ``LandXMLDocument`` --
  invariants that only make sense once the domain objects exist
  (unique names, finite coordinates). ``LandXMLWriter`` calls this
  before serializing.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections import Counter
from math import isfinite
from xml.etree.ElementTree import Element

from topocore.io.landxml._xml_utils import children as _children
from topocore.io.landxml._xml_utils import local_tag as _local_tag
from topocore.io.landxml.constants import SURF_TYPE_TIN
from topocore.io.landxml.exceptions import LandXMLValidationError
from topocore.io.landxml.models import LandXMLDocument


class LandXMLValidator:
    """
    Semantic validator for the LandXML 1.2 subset TopoCore supports.
    """

    __slots__ = ()

    def validate_xml(self, root: Element) -> None:
        """
        Validate a parsed LandXML tree before it is converted into
        domain objects.

        Parameters
        ----------
        root
            The ``<LandXML>`` root element.

        Raises
        ------
        LandXMLValidationError
            If a required attribute is missing, point ids are
            duplicated within a ``<Pnts>``/``<CgPoints>`` collection,
            or a ``<F>`` face references a point id absent from its
            surface's ``<Pnts>``.
        """
        if _local_tag(root) != "LandXML":
            raise LandXMLValidationError(f"Expected a <LandXML> root element, got <{_local_tag(root)}>.")

        for surfaces in _children(root, "Surfaces"):
            self._validate_surfaces(surfaces)

        for point_groups in _children(root, "CgPoints"):
            self._validate_cgpoints(point_groups)

        for alignments in _children(root, "Alignments"):
            self._validate_alignments(alignments)

    def _validate_surfaces(self, surfaces: Element) -> None:
        surface_names: list[str] = []

        for surface in _children(surfaces, "Surface"):
            name = surface.get("name")

            if not name:
                raise LandXMLValidationError("<Surface> is missing a required 'name' attribute.")

            surface_names.append(name)

            for definition in _children(surface, "Definition"):
                if definition.get("surfType", SURF_TYPE_TIN) != SURF_TYPE_TIN:
                    # Unsupported surface type (GRID/ROAD): the reader
                    # skips these with a warning, not an error here --
                    # see LandXMLReader.
                    continue

                self._validate_definition(definition, surface_name=name)

        _raise_on_duplicates(surface_names, "<Surface> name")

    def _validate_definition(self, definition: Element, *, surface_name: str) -> None:
        pnts_list = _children(definition, "Pnts")

        if not pnts_list:
            raise LandXMLValidationError(f"Surface '{surface_name}': <Definition> has no <Pnts>.")

        point_ids: set[str] = set()
        seen_ids: list[str] = []

        for pnts in pnts_list:
            for p in _children(pnts, "P"):
                point_id = p.get("id")

                if not point_id:
                    raise LandXMLValidationError(f"Surface '{surface_name}': <P> is missing a required 'id'.")

                seen_ids.append(point_id)
                point_ids.add(point_id)

        _raise_on_duplicates(seen_ids, f"Surface '{surface_name}': <P> id")

        for faces in _children(definition, "Faces"):
            for f in _children(faces, "F"):
                text = (f.text or "").split()

                if len(text) != 3:
                    raise LandXMLValidationError(
                        f"Surface '{surface_name}': <F> must reference exactly 3 point ids, got {len(text)}."
                    )

                for referenced_id in text:
                    if referenced_id not in point_ids:
                        raise LandXMLValidationError(
                            f"Surface '{surface_name}': <F> references point id "
                            f"'{referenced_id}', which does not exist in <Pnts>."
                        )

    def _validate_cgpoints(self, point_groups: Element) -> None:
        """
        ``<CgPoints name="...">`` on the *group* is optional per the
        LandXML schema -- confirmed against a genuine Autodesk
        Civil 3D 2007 export that omits it (see the PR18C session
        notes). Only ``<CgPoint name="...">``, the individual point,
        is required -- UNLESS it uses ``pntRef`` (point-by-reference,
        another real, valid LandXML feature confirmed in the same
        export): the point's identity then comes from the
        referenced point, not its own ``name``. ``LandXMLReader``
        skips ``pntRef`` points individually with an explicit
        warning (not supported -- no literal coordinates to parse),
        rather than this validator rejecting the whole group over a
        missing ``name`` that was never required in that case.
        """
        name = point_groups.get("name") or "<CgPoints> (unnamed)"

        point_names: list[str] = []

        for cgpoint in _children(point_groups, "CgPoint"):
            point_name = cgpoint.get("name")
            point_ref = cgpoint.get("pntRef")

            if not point_name:
                if point_ref:
                    continue

                raise LandXMLValidationError(f"CgPoints '{name}': <CgPoint> is missing a required 'name'.")

            point_names.append(point_name)

        _raise_on_duplicates(point_names, f"CgPoints '{name}': <CgPoint> name")

    def _validate_alignments(self, alignments: Element) -> None:
        alignment_names: list[str] = []

        for alignment in _children(alignments, "Alignment"):
            name = alignment.get("name")

            if not name:
                raise LandXMLValidationError("<Alignment> is missing a required 'name' attribute.")

            alignment_names.append(name)

            for coordgeom in _children(alignment, "CoordGeom"):
                self._validate_coordgeom(coordgeom, alignment_name=name)

        _raise_on_duplicates(alignment_names, "<Alignment> name")

    def _validate_coordgeom(self, coordgeom: Element, *, alignment_name: str) -> None:
        for child in coordgeom:
            tag = _local_tag(child)

            if tag == "Line":
                self._require_children(
                    child,
                    ("Start", "End"),
                    context=f"Alignment '{alignment_name}': <Line>",
                )
            elif tag == "Curve":
                # <Center> may legitimately be absent for chord-defined
                # curves -- the reader skips those with a warning
                # (crvType handling), not a hard error, so it is not
                # required here.
                self._require_children(
                    child,
                    ("Start", "End"),
                    context=f"Alignment '{alignment_name}': <Curve>",
                )
            elif tag == "Spiral":
                self._require_children(
                    child,
                    ("Start", "End", "PI"),
                    context=f"Alignment '{alignment_name}': <Spiral>",
                )

    def _require_children(self, element: Element, tags: tuple[str, ...], *, context: str) -> None:
        for tag in tags:
            if not _children(element, tag):
                raise LandXMLValidationError(f"{context} is missing a required <{tag}>.")

    def validate_document(self, document: LandXMLDocument) -> None:
        """
        Validate an assembled ``LandXMLDocument``.

        Called by ``LandXMLReader`` after parsing (defense in depth)
        and by ``LandXMLWriter`` before serializing (required --
        the writer must never emit a document that violates these
        invariants).

        Raises
        ------
        LandXMLValidationError
            If surface/point-group names are duplicated or blank, or
            any coordinate is non-finite.
        """
        surface_names = [surface.name for surface in document.surfaces]
        _raise_on_duplicates(surface_names, "Surface name")
        _raise_on_blank(surface_names, "Surface name")

        group_names = [group.name for group in document.point_groups]
        _raise_on_duplicates(group_names, "Point group name")
        _raise_on_blank(group_names, "Point group name")

        alignment_names = [alignment.name for alignment in document.alignments]
        _raise_on_duplicates(alignment_names, "Alignment name")
        _raise_on_blank(alignment_names, "Alignment name")

        for surface in document.surfaces:
            for vertex in surface.tin.vertices:
                if not (isfinite(vertex.x) and isfinite(vertex.y) and isfinite(vertex.z)):
                    raise LandXMLValidationError(f"Surface '{surface.name}' contains a non-finite coordinate.")

        for group in document.point_groups:
            for point in group.points:
                if not (isfinite(point.x) and isfinite(point.y) and isfinite(point.z)):
                    raise LandXMLValidationError(
                        f"Point group '{group.name}': point '{point.id}' has a non-finite coordinate."
                    )


def _raise_on_duplicates(values: list[str], label: str) -> None:
    counts = Counter(values)
    duplicates = [value for value, count in counts.items() if count > 1]

    if duplicates:
        raise LandXMLValidationError(f"Duplicated {label}(s): {sorted(duplicates)}.")


def _raise_on_blank(values: list[str], label: str) -> None:
    if any(not value.strip() for value in values):
        raise LandXMLValidationError(f"Blank {label} is not allowed.")


__all__ = [
    "LandXMLValidator",
]
