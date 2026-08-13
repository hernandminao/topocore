"""
topocore.io.landxml.reader
============================

LandXML reader.

Parses ``<Surfaces>``/TIN and ``<CgPoints>`` into a
``LandXMLDocument``, using ``TIN.from_mesh()`` (not
``TIN.from_points()``) so the exact ``<Faces>`` connectivity from
the file is preserved rather than silently replaced by a fresh
Delaunay triangulation of the same points.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import Element, ParseError, parse

import numpy as np
from numpy.typing import NDArray

from topocore.geometry.point3d import Point3D
from topocore.io.landxml._xml_utils import children
from topocore.io.landxml.coordinates import parse_point_text
from topocore.io.landxml.exceptions import LandXMLParseError
from topocore.io.landxml.models import (
    LandXMLDocument,
    LinearUnit,
    NamedPointGroup,
    NamedSurface,
)
from topocore.io.landxml.report import LandXMLReadReport, _ReadReportBuilder
from topocore.io.landxml.validation import LandXMLValidator
from topocore.survey.models import SurveyPoint, SurveyPointSet
from topocore.terrain.exceptions import TerrainError
from topocore.terrain.tin import TIN

_SURF_TYPE_TIN = "TIN"


class LandXMLReader:
    """
    Reads a LandXML file into a ``LandXMLDocument``.

    Only the ``<Surfaces>``/TIN and ``<CgPoints>`` elements are
    interpreted -- ``<Alignments>``, ``<Profile>`` and ``<Feature>``
    (embedded in ``<CgPoint>``) are left unread (see the PR18B
    contract). Their presence in the file does not raise; it is
    simply not represented in the returned document.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._validator = LandXMLValidator()

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> LandXMLDocument:
        """
        Parse the file.

        Returns
        -------
        LandXMLDocument

        Raises
        ------
        LandXMLParseError
            If the file is not well-formed XML, or a required
            element/attribute in the supported subset is malformed.
        LandXMLValidationError
            If the file violates a semantic invariant (duplicated
            ids/names, a ``<F>`` referencing an unknown point id).
        """
        document, _ = self.read_with_report()
        return document

    def read_with_report(self) -> tuple[LandXMLDocument, LandXMLReadReport]:
        """
        Parse the file, returning both the document and a report of
        what was read (counts, and warnings for skipped content such
        as a non-TIN ``<Surface surfType="...">``).
        """
        root = self._parse_xml()

        self._validator.validate_xml(root)

        report = _ReadReportBuilder(input_path=self._path)

        surfaces = self._read_surfaces(root, report)
        point_groups = self._read_point_groups(root, report)
        linear_unit, crs = self._read_units_and_crs(root)

        document = LandXMLDocument(
            surfaces=surfaces,
            point_groups=point_groups,
            crs=crs,
            linear_unit=linear_unit,
        )

        self._validator.validate_document(document)

        return document, report.build()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_xml(self) -> Element:
        try:
            tree = parse(self._path)
        except ParseError as exc:
            raise LandXMLParseError(f"'{self._path}' is not well-formed XML: {exc}") from exc
        except OSError as exc:
            raise LandXMLParseError(f"Could not read '{self._path}': {exc}") from exc

        return tree.getroot()

    def _read_surfaces(
        self,
        root: Element,
        report: _ReadReportBuilder,
    ) -> tuple[NamedSurface, ...]:
        surfaces: list[NamedSurface] = []

        for surfaces_el in children(root, "Surfaces"):
            for surface_el in children(surfaces_el, "Surface"):
                name = surface_el.get("name", "")
                desc = surface_el.get("desc") or None

                definitions = children(surface_el, "Definition")

                if not definitions:
                    report.warnings.append(f"Surface '{name}': has no <Definition>; skipped.")
                    continue

                definition = definitions[0]
                surf_type = definition.get("surfType", _SURF_TYPE_TIN)

                if surf_type != _SURF_TYPE_TIN:
                    report.warnings.append(
                        f"Surface '{name}': surfType='{surf_type}' is not supported (TIN only); skipped."
                    )
                    continue

                tin = self._read_tin_definition(definition, surface_name=name)

                surfaces.append(NamedSurface(name=name, tin=tin, desc=desc))
                report.surface_count += 1
                report.triangle_count += tin.triangle_count
                report.point_count += tin.vertex_count

        return tuple(surfaces)

    def _read_tin_definition(self, definition: Element, *, surface_name: str) -> TIN:
        id_to_index: dict[str, int] = {}
        vertices: list[Point3D] = []

        for pnts_el in children(definition, "Pnts"):
            for p_el in children(pnts_el, "P"):
                point_id = p_el.get("id", "")
                point = parse_point_text(p_el.text or "")

                id_to_index[point_id] = len(vertices)
                vertices.append(point)

        faces: list[tuple[int, int, int]] = []

        for faces_el in children(definition, "Faces"):
            for f_el in children(faces_el, "F"):
                ids = (f_el.text or "").split()
                # LandXMLValidator.validate_xml already guarantees
                # every id here exists in id_to_index and that there
                # are exactly 3 of them.
                faces.append(
                    (
                        id_to_index[ids[0]],
                        id_to_index[ids[1]],
                        id_to_index[ids[2]],
                    )
                )

        simplices: NDArray[np.int32] = np.asarray(faces, dtype=np.int32)

        try:
            return TIN.from_mesh(tuple(vertices), simplices)
        except TerrainError as exc:
            raise LandXMLParseError(f"Surface '{surface_name}': invalid TIN definition: {exc}") from exc

    def _read_point_groups(
        self,
        root: Element,
        report: _ReadReportBuilder,
    ) -> tuple[NamedPointGroup, ...]:
        groups: list[NamedPointGroup] = []

        for cgpoints_el in children(root, "CgPoints"):
            name = cgpoints_el.get("name", "")
            desc = cgpoints_el.get("desc") or None

            points: list[SurveyPoint] = []

            for cgpoint_el in children(cgpoints_el, "CgPoint"):
                point_id = cgpoint_el.get("name", "")
                code = cgpoint_el.get("code") or None
                position = parse_point_text(cgpoint_el.text or "")

                points.append(
                    SurveyPoint(
                        id=point_id,
                        x=position.x,
                        y=position.y,
                        z=position.z,
                        code=code,
                    )
                )

            groups.append(NamedPointGroup(name=name, points=SurveyPointSet(points=tuple(points)), desc=desc))
            report.point_group_count += 1
            report.point_count += len(points)

        return tuple(groups)

    def _read_units_and_crs(self, root: Element) -> tuple[LinearUnit, str | None]:
        """
        Resolve the document's linear unit and (raw) CRS string.

        Only the first ``<Units>`` element is consulted; within it,
        ``<Imperial>`` and ``<Metric>`` are mutually exclusive per
        the LandXML schema, so whichever is present decides the
        result deterministically -- never "last one wins" between
        the two.
        """
        units_elements = children(root, "Units")
        linear_unit = LinearUnit.METER

        if units_elements:
            units_el = units_elements[0]
            imperial_elements = children(units_el, "Imperial")
            metric_elements = children(units_el, "Metric")

            if imperial_elements or (metric_elements and metric_elements[0].get("linearUnit", "meter") == "foot"):
                linear_unit = LinearUnit.FOOT

        crs_elements = children(root, "CoordinateSystem")
        crs = None

        if crs_elements:
            crs_el = crs_elements[0]
            crs = crs_el.get("name") or crs_el.get("epsgCode") or crs_el.get("desc")

        return linear_unit, crs


__all__ = [
    "LandXMLReader",
]
