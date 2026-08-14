"""
topocore.io.landxml.writer
============================

LandXML writer.

Serializes a ``LandXMLDocument`` back to ``<Surfaces>``/TIN,
``<CgPoints>``, and ``<Alignments>`` elements, producing a LandXML
1.2 file. Validates the document (``LandXMLValidator.validate_document``)
before writing anything to disk -- the writer must never emit a
document that violates the invariants ``LandXMLReader`` itself
enforces on read.

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

from topocore.alignment.elements import (
    ArcElement,
    HorizontalElement,
    LineElement,
    SpiralElement,
)
from topocore.alignment.models import DesignProfile
from topocore.alignment.vertical_elements import VerticalCurve
from topocore.geometry.point2d import Point2D
from topocore.geometry.point3d import Point3D
from topocore.io.landxml.codecs import (
    format_radius,
    format_rotation,
    format_station_elevation,
)
from topocore.io.landxml.constants import (
    CRV_TYPE_ARC,
    DEFAULT_COORDINATE_PRECISION,
    LANDXML_1_2_NAMESPACE,
    LANDXML_VERSION,
    SPI_TYPE_CLOTHOID,
    SURF_TYPE_TIN,
)
from topocore.io.landxml.coordinates import format_point_text
from topocore.io.landxml.exceptions import LandXMLWriteError
from topocore.io.landxml.models import LandXMLDocument, LinearUnit
from topocore.io.landxml.report import LandXMLWriteReport, _WriteReportBuilder
from topocore.io.landxml.validation import LandXMLValidator


def _format_point2d(point: Point2D, *, precision: int) -> str:
    """
    Same "north east" convention as ``coordinates.format_point_text``,
    for the 2D points used throughout ``<CoordGeom>``
    (``<Start>``/``<End>``/``<Center>``/``<PI>``).
    """
    return f"{point.y:.{precision}f} {point.x:.{precision}f}"


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

        if document.alignments:
            self._write_alignments(root, document, report)

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

    def _write_alignments(
        self,
        root: Element,
        document: LandXMLDocument,
        report: _WriteReportBuilder,
    ) -> None:
        alignments_el = SubElement(root, "Alignments")

        for named_alignment in document.alignments:
            alignment = named_alignment.alignment

            alignment_el = SubElement(
                alignments_el,
                "Alignment",
                {
                    "name": named_alignment.name,
                    "length": f"{alignment.length:.{self._precision}f}",
                    "staStart": f"{alignment.start_station:.{self._precision}f}",
                },
            )

            if named_alignment.desc:
                alignment_el.set("desc", named_alignment.desc)

            coordgeom_el = SubElement(alignment_el, "CoordGeom")

            for element in alignment.elements:
                self._write_horizontal_element(coordgeom_el, element)

            if named_alignment.profile is not None:
                self._write_profile(alignment_el, named_alignment.profile)

            report.alignment_count += 1

    def _write_horizontal_element(self, coordgeom_el: Element, element: HorizontalElement) -> None:
        precision = self._precision

        if isinstance(element, LineElement):
            length = element.start.distance_to(element.end)
            line_el = SubElement(coordgeom_el, "Line", {"length": f"{length:.{precision}f}"})
            SubElement(line_el, "Start").text = _format_point2d(element.start, precision=precision)
            SubElement(line_el, "End").text = _format_point2d(element.end, precision=precision)

        elif isinstance(element, ArcElement):
            curve_el = SubElement(
                coordgeom_el,
                "Curve",
                {
                    "crvType": CRV_TYPE_ARC,
                    "rot": format_rotation(element.clockwise),
                    "radius": f"{element.radius:.{precision}f}",
                },
            )
            SubElement(curve_el, "Start").text = _format_point2d(element.start, precision=precision)
            SubElement(curve_el, "Center").text = _format_point2d(element.center, precision=precision)
            SubElement(curve_el, "End").text = _format_point2d(element.end, precision=precision)

        elif isinstance(element, SpiralElement):
            spiral_el = SubElement(
                coordgeom_el,
                "Spiral",
                {
                    "spiType": SPI_TYPE_CLOTHOID,
                    "rot": format_rotation(element.clockwise),
                    "length": f"{element.length:.{precision}f}",
                    "radiusStart": format_radius(element.radius_start, precision=precision),
                    "radiusEnd": format_radius(element.radius_end, precision=precision),
                },
            )
            SubElement(spiral_el, "Start").text = _format_point2d(element.start, precision=precision)
            SubElement(spiral_el, "PI").text = _format_point2d(element.pi, precision=precision)
            SubElement(spiral_el, "End").text = _format_point2d(element.end, precision=precision)

    def _write_profile(self, alignment_el: Element, profile: DesignProfile) -> None:
        profile_el = SubElement(alignment_el, "Profile")
        profalign_el = SubElement(profile_el, "ProfAlign")

        elements = profile.elements

        first = elements[0]
        SubElement(profalign_el, "PVI").text = format_station_elevation(
            first.start_station,
            first.elevation_at(first.start_station),
            precision=self._precision,
        )

        for element in elements:
            if isinstance(element, VerticalCurve):
                if element.is_symmetric:
                    curve_el = SubElement(
                        profalign_el,
                        "ParaCurve",
                        {"length": f"{element.length:.{self._precision}f}"},
                    )
                else:
                    curve_el = SubElement(
                        profalign_el,
                        "ParaCurve",
                        {
                            "lengthIn": f"{element.length_in:.{self._precision}f}",
                            "lengthOut": f"{element.length_out:.{self._precision}f}",
                        },
                    )
                curve_el.text = format_station_elevation(
                    element.pvi_station,
                    element.pvi_elevation,
                    precision=self._precision,
                )
            else:
                SubElement(profalign_el, "PVI").text = format_station_elevation(
                    element.end_station,
                    element.end_elevation,
                    precision=self._precision,
                )

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
