"""
topocore.io.landxml.writer
============================

LandXML writer.

Serializes a ``LandXMLDocument`` back to ``<Surfaces>``/TIN and
``<CgPoints>`` elements, producing a LandXML 1.2 file. Validates the
document (``LandXMLValidator.validate_document``) before writing
anything to disk -- the writer must never emit a document that
violates the invariants ``LandXMLReader`` itself enforces on read.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

from topocore.geometry.point3d import Point3D
from topocore.io.landxml.constants import (
    DEFAULT_COORDINATE_PRECISION,
    LANDXML_1_2_NAMESPACE,
    LANDXML_VERSION,
    SURF_TYPE_TIN,
)
from topocore.io.landxml.coordinates import format_point_text
from topocore.io.landxml.exceptions import LandXMLWriteError
from topocore.io.landxml.models import LandXMLDocument, LinearUnit
from topocore.io.landxml.report import LandXMLWriteReport, _WriteReportBuilder
from topocore.io.landxml.validation import LandXMLValidator


class LandXMLWriter:
    """
    Writes a ``LandXMLDocument`` to a LandXML 1.2 file.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        coordinate_precision: int = DEFAULT_COORDINATE_PRECISION,
    ) -> None:
        self._path = Path(path)
        self._precision = coordinate_precision
        self._validator = LandXMLValidator()

    @property
    def path(self) -> Path:
        return self._path

    def write(self, document: LandXMLDocument) -> LandXMLWriteReport:
        """
        Validate and serialize ``document``.

        Returns
        -------
        LandXMLWriteReport

        Raises
        ------
        LandXMLValidationError
            If ``document`` violates a semantic invariant (duplicated
            or blank surface/point-group names, non-finite
            coordinates).
        LandXMLWriteError
            If writing the file to disk fails.
        """
        self._validator.validate_document(document)

        report = _WriteReportBuilder(output_path=self._path)

        root = Element("LandXML")
        root.set("xmlns", LANDXML_1_2_NAMESPACE)
        root.set("version", LANDXML_VERSION)

        self._write_units(root, document.linear_unit)

        if document.crs is not None:
            crs_el = SubElement(root, "CoordinateSystem")
            crs_el.set("name", document.crs)

        if document.surfaces:
            self._write_surfaces(root, document, report)

        if document.point_groups:
            self._write_point_groups(root, document, report)

        self._save(root)

        return report.build()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _write_units(self, root: Element, linear_unit: LinearUnit) -> None:
        units_el = SubElement(root, "Units")

        if linear_unit == LinearUnit.FOOT:
            SubElement(units_el, "Imperial", {"linearUnit": "foot"})
        else:
            SubElement(units_el, "Metric", {"linearUnit": "meter"})

    def _write_surfaces(
        self,
        root: Element,
        document: LandXMLDocument,
        report: _WriteReportBuilder,
    ) -> None:
        surfaces_el = SubElement(root, "Surfaces")

        for surface in document.surfaces:
            surface_el = SubElement(surfaces_el, "Surface", {"name": surface.name})

            if surface.desc:
                surface_el.set("desc", surface.desc)

            definition_el = SubElement(surface_el, "Definition", {"surfType": SURF_TYPE_TIN})

            pnts_el = SubElement(definition_el, "Pnts")

            for index, vertex in enumerate(surface.tin.vertices, start=1):
                p_el = SubElement(pnts_el, "P", {"id": str(index)})
                p_el.text = format_point_text(vertex, precision=self._precision)

            faces_el = SubElement(definition_el, "Faces")

            for simplex in surface.tin.simplices:
                f_el = SubElement(faces_el, "F")
                # <P id> values written above are 1-based, matching
                # the 0-based simplex indices offset by one.
                f_el.text = f"{simplex[0] + 1} {simplex[1] + 1} {simplex[2] + 1}"

            report.surface_count += 1
            report.triangle_count += surface.tin.triangle_count
            report.point_count += surface.tin.vertex_count

    def _write_point_groups(
        self,
        root: Element,
        document: LandXMLDocument,
        report: _WriteReportBuilder,
    ) -> None:
        for group in document.point_groups:
            group_el = SubElement(root, "CgPoints", {"name": group.name})

            if group.desc:
                group_el.set("desc", group.desc)

            for point in group.points:
                point_el = SubElement(group_el, "CgPoint", {"name": point.id})

                if point.code:
                    point_el.set("code", point.code)

                point_el.text = format_point_text(
                    Point3D(point.x, point.y, point.z),
                    precision=self._precision,
                )

            report.point_group_count += 1
            report.point_count += len(group.points)

    def _save(self, root: Element) -> None:
        indent(root, space="  ")
        tree = ElementTree(root)

        try:
            tree.write(self._path, encoding="utf-8", xml_declaration=True)
        except OSError as exc:
            raise LandXMLWriteError(f"Could not write '{self._path}': {exc}") from exc


__all__ = [
    "LandXMLWriter",
]
