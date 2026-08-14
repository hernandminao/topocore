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

import math
from collections.abc import Callable
from pathlib import Path
from xml.etree.ElementTree import Element, ParseError, parse

import numpy as np
from numpy.typing import NDArray

from topocore.alignment.elements import (
    ArcElement,
    HorizontalElement,
    LineElement,
    SpiralElement,
)
from topocore.alignment.exceptions import AlignmentError, AlignmentGeometryError
from topocore.alignment.models import Alignment, DesignProfile
from topocore.alignment.vertical_elements import (
    GradeSegment,
    VerticalCurve,
    VerticalElement,
)
from topocore.geometry.point2d import Point2D
from topocore.geometry.point3d import Point3D
from topocore.io.landxml._xml_utils import children
from topocore.io.landxml.codecs import (
    parse_radius,
    parse_rotation,
    parse_station_elevation,
)
from topocore.io.landxml.constants import (
    CRV_TYPE_ARC,
    LANDXML_GEOMETRY_TOLERANCE,
    SPI_TYPE_CLOTHOID,
)
from topocore.io.landxml.coordinates import parse_point_text
from topocore.io.landxml.exceptions import LandXMLParseError
from topocore.io.landxml.models import (
    LandXMLDocument,
    LinearUnit,
    NamedAlignment,
    NamedPointGroup,
    NamedSurface,
)
from topocore.io.landxml.report import LandXMLReadReport, _ReadReportBuilder
from topocore.io.landxml.validation import LandXMLValidator
from topocore.math.tolerance import compare
from topocore.survey.models import SurveyPoint, SurveyPointSet
from topocore.terrain.exceptions import TerrainError
from topocore.terrain.tin import TIN

_SURF_TYPE_TIN = "TIN"


def _point2d_from_text(text: str) -> Point2D:
    point = parse_point_text(text)
    return Point2D(point.x, point.y)


class LandXMLReader:
    """
    Reads a LandXML file into a ``LandXMLDocument``.

    ``<Surfaces>``/TIN, ``<CgPoints>``, and ``<Alignments>``
    (``<CoordGeom>`` + ``<Profile>``/``<ProfAlign>``) are
    interpreted. ``<Feature>`` (embedded in ``<CgPoint>``) is left
    unread. Unsupported curve/spiral variants within an ``<Alignment>``
    (chord-defined curves, non-clothoid spirals, curve-to-curve
    compound spirals) are skipped with a warning, not a hard error.
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
        alignments = self._read_alignments(root, report)
        linear_unit, crs = self._read_units_and_crs(root)

        document = LandXMLDocument(
            surfaces=surfaces,
            point_groups=point_groups,
            alignments=alignments,
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
        """
        ``<CgPoints name="...">`` is common but the LandXML schema
        does not require ``name`` on the group itself (unlike
        ``<CgPoint name="...">``, the individual point, which is
        required and always validated). An unnamed ``<CgPoints>`` is
        real, valid data -- confirmed against a genuine Autodesk
        Civil 3D 2007 export, not merely a schema technicality (see
        the PR18C session notes).

        ``NamedPointGroup.name`` still needs a non-blank string
        (it's how TopoCore identifies the group) -- this is a
        TopoCore/IO adaptation, not a tightening of the LandXML
        standard: an unnamed group gets a deterministic generated
        name, ``Unnamed_CgPoints_{n}`` (n counts only the unnamed
        groups, in document order), so it's never silently dropped
        or given a colliding/blank name.
        """
        groups: list[NamedPointGroup] = []
        unnamed_count = 0

        for cgpoints_el in children(root, "CgPoints"):
            name = cgpoints_el.get("name")

            if not name:
                unnamed_count += 1
                name = f"Unnamed_CgPoints_{unnamed_count}"

            desc = cgpoints_el.get("desc") or None

            points: list[SurveyPoint] = []

            for cgpoint_el in children(cgpoints_el, "CgPoint"):
                point_ref = cgpoint_el.get("pntRef")

                if point_ref:
                    report.warnings.append(
                        f"Point group '{name}': <CgPoint pntRef='{point_ref}'> (point-by-reference) "
                        "is not supported (no literal coordinates to read); skipped."
                    )
                    continue

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

    def _read_alignments(
        self,
        root: Element,
        report: _ReadReportBuilder,
    ) -> tuple[NamedAlignment, ...]:
        alignments: list[NamedAlignment] = []

        for alignments_el in children(root, "Alignments"):
            for alignment_el in children(alignments_el, "Alignment"):
                name = alignment_el.get("name", "")
                desc = alignment_el.get("desc") or None
                start_station = float(alignment_el.get("staStart", "0.0"))

                coordgeom_elements = children(alignment_el, "CoordGeom")

                if not coordgeom_elements:
                    report.warnings.append(f"Alignment '{name}': has no <CoordGeom>; skipped.")
                    continue

                horizontal_elements = self._read_coordgeom(coordgeom_elements[0], alignment_name=name, report=report)

                if not horizontal_elements:
                    report.warnings.append(f"Alignment '{name}': <CoordGeom> produced no supported elements; skipped.")
                    continue

                try:
                    alignment = Alignment(
                        name=name,
                        elements=tuple(horizontal_elements),
                        start_station=start_station,
                        desc=desc,
                    )
                except AlignmentError as exc:
                    raise LandXMLParseError(f"Alignment '{name}': invalid horizontal geometry: {exc}") from exc

                profile = self._read_profile(alignment_el, alignment_name=name, report=report)

                alignments.append(NamedAlignment(name=name, alignment=alignment, profile=profile, desc=desc))
                report.alignment_count += 1

        return tuple(alignments)

    def _read_coordgeom(
        self,
        coordgeom_el: Element,
        *,
        alignment_name: str,
        report: _ReadReportBuilder,
    ) -> list[HorizontalElement]:
        elements: list[HorizontalElement] = []

        for child in coordgeom_el:
            tag = child.tag.rsplit("}", 1)[-1] if "}" in child.tag else child.tag

            if tag == "Line":
                elements.append(self._read_line(child))
            elif tag == "Curve":
                curve = self._read_curve(child, alignment_name=alignment_name, report=report)
                if curve is not None:
                    elements.append(curve)
            elif tag == "Spiral":
                spiral = self._read_spiral(child, alignment_name=alignment_name, report=report)
                if spiral is not None:
                    elements.append(spiral)

        return elements

    def _construct_with_landxml_tolerance(
        self,
        factory: Callable[[float | None], HorizontalElement],
        *,
        alignment_name: str,
        element_description: str,
        report: _ReadReportBuilder,
    ) -> HorizontalElement:
        """
        Try building an ``ArcElement``/``SpiralElement`` with the
        domain's strict default tolerance first. If that fails,
        retry with ``LANDXML_GEOMETRY_TOLERANCE`` -- an
        interoperability tolerance for serialized engineering data,
        confirmed empirically against a real Civil 3D export (see
        ``constants.LANDXML_GEOMETRY_TOLERANCE``'s docstring). No
        coordinate or attribute value is ever changed here; a wider
        tolerance only widens the acceptance window for the values
        exactly as parsed. If even that fails, the geometry is
        genuinely inconsistent and is reported as a parse error, not
        silently skipped.

        Reuses the domain's own exception message in the resulting
        warning -- the exact discrepancy is already in it, so it
        does not need to be recomputed here.
        """
        try:
            return factory(None)
        except AlignmentGeometryError as strict_exc:
            try:
                element = factory(LANDXML_GEOMETRY_TOLERANCE)
            except AlignmentGeometryError as exc:
                raise LandXMLParseError(
                    f"Alignment '{alignment_name}': invalid {element_description} geometry: {exc}"
                ) from exc

            report.warnings.append(
                f"Alignment '{alignment_name}': {element_description} accepted within the "
                f"LandXML import tolerance ({LANDXML_GEOMETRY_TOLERANCE:.0e} m) but exceeds "
                f"TopoCore's domain tolerance; no coordinates were modified. Detail: {strict_exc}"
            )
            return element

    def _read_line(self, line_el: Element) -> LineElement:
        starts = children(line_el, "Start")
        ends = children(line_el, "End")

        try:
            return LineElement(
                start=_point2d_from_text(starts[0].text or ""),
                end=_point2d_from_text(ends[0].text or ""),
            )
        except AlignmentGeometryError as exc:
            raise LandXMLParseError(f"Invalid <Line> geometry: {exc}") from exc

    def _read_curve(
        self,
        curve_el: Element,
        *,
        alignment_name: str,
        report: _ReadReportBuilder,
    ) -> ArcElement | None:
        crv_type = curve_el.get("crvType", CRV_TYPE_ARC)

        if crv_type != CRV_TYPE_ARC:
            report.warnings.append(
                f"Alignment '{alignment_name}': <Curve crvType='{crv_type}'> is not supported (arc only); skipped."
            )
            return None

        centers = children(curve_el, "Center")

        if not centers:
            report.warnings.append(
                f"Alignment '{alignment_name}': <Curve> has no <Center> (chord-defined curve); skipped."
            )
            return None

        starts = children(curve_el, "Start")
        ends = children(curve_el, "End")
        radius = float(curve_el.get("radius", "0.0"))
        clockwise = parse_rotation(curve_el.get("rot", "cw"))

        start = _point2d_from_text(starts[0].text or "")
        end = _point2d_from_text(ends[0].text or "")
        center = _point2d_from_text(centers[0].text or "")

        element = self._construct_with_landxml_tolerance(
            lambda tolerance: ArcElement(
                start=start,
                end=end,
                center=center,
                radius=radius,
                clockwise=clockwise,
                tolerance=tolerance,
            ),
            alignment_name=alignment_name,
            element_description="<Curve>",
            report=report,
        )
        assert isinstance(element, ArcElement)
        return element

    def _read_spiral(
        self,
        spiral_el: Element,
        *,
        alignment_name: str,
        report: _ReadReportBuilder,
    ) -> SpiralElement | None:
        spi_type = spiral_el.get("spiType", SPI_TYPE_CLOTHOID)

        if spi_type != SPI_TYPE_CLOTHOID:
            report.warnings.append(
                f"Alignment '{alignment_name}': <Spiral spiType='{spi_type}'> is not supported "
                "(clothoid only); skipped."
            )
            return None

        radius_start = parse_radius(spiral_el.get("radiusStart", "INF"))
        radius_end = parse_radius(spiral_el.get("radiusEnd", "INF"))

        if math.isfinite(radius_start) and math.isfinite(radius_end):
            report.warnings.append(
                f"Alignment '{alignment_name}': <Spiral> has both radiusStart={radius_start} and "
                f"radiusEnd={radius_end} finite (curve-to-curve compound spiral); not supported "
                "in this delivery; skipped."
            )
            return None

        starts = children(spiral_el, "Start")
        ends = children(spiral_el, "End")
        pis = children(spiral_el, "PI")
        length = float(spiral_el.get("length", "0.0"))
        clockwise = parse_rotation(spiral_el.get("rot", "cw"))

        start = _point2d_from_text(starts[0].text or "")
        end = _point2d_from_text(ends[0].text or "")
        pi = _point2d_from_text(pis[0].text or "")

        element = self._construct_with_landxml_tolerance(
            lambda tolerance: SpiralElement(
                start=start,
                end=end,
                pi=pi,
                radius_start=radius_start,
                radius_end=radius_end,
                length=length,
                clockwise=clockwise,
                tolerance=tolerance,
            ),
            alignment_name=alignment_name,
            element_description="<Spiral>",
            report=report,
        )
        assert isinstance(element, SpiralElement)
        return element

    def _read_profile(
        self,
        alignment_el: Element,
        *,
        alignment_name: str,
        report: _ReadReportBuilder,
    ) -> DesignProfile | None:
        profile_elements = children(alignment_el, "Profile")

        if not profile_elements:
            return None

        profalign_elements = children(profile_elements[0], "ProfAlign")

        if not profalign_elements:
            return None

        raw_points = self._read_profalign_points(profalign_elements[0])

        if len(raw_points) < 2:
            report.warnings.append(
                f"Alignment '{alignment_name}': <ProfAlign> has fewer than 2 points; profile skipped."
            )
            return None

        elements = self._build_vertical_elements(raw_points, alignment_name=alignment_name)

        try:
            return DesignProfile(alignment_name=alignment_name, elements=tuple(elements))
        except AlignmentError as exc:
            raise LandXMLParseError(f"Alignment '{alignment_name}': invalid vertical profile: {exc}") from exc

    def _read_profalign_points(
        self,
        profalign_el: Element,
    ) -> list[tuple[float, float, tuple[float, float] | None]]:
        """
        Returns a list of ``(station, elevation, curve_lengths)``,
        where ``curve_lengths`` is ``(length_in, length_out)`` for a
        ``<ParaCurve>`` point, or ``None`` for a plain ``<PVI>``.
        """
        points: list[tuple[float, float, tuple[float, float] | None]] = []

        for child in profalign_el:
            tag = child.tag.rsplit("}", 1)[-1] if "}" in child.tag else child.tag
            text = child.text or ""

            if tag == "PVI":
                station, elevation = parse_station_elevation(text)
                points.append((station, elevation, None))
            elif tag == "ParaCurve":
                station, elevation = parse_station_elevation(text)

                if "lengthIn" in child.attrib and "lengthOut" in child.attrib:
                    length_in = float(child.get("lengthIn", "0.0"))
                    length_out = float(child.get("lengthOut", "0.0"))
                else:
                    length = float(child.get("length", "0.0"))
                    length_in = length_out = length / 2.0

                points.append((station, elevation, (length_in, length_out)))
            # Other PVI variants (CircCurve, etc.) are out of scope
            # and simply not appended -- they are not silently
            # merged into an incorrect straight segment; instead the
            # DesignProfile continuity check would catch any
            # resulting station/elevation mismatch as a validation
            # error, since dropping a point here can only ever
            # remove information, not fabricate a wrong value.

        return points

    def _build_vertical_elements(
        self,
        raw_points: list[tuple[float, float, tuple[float, float] | None]],
        *,
        alignment_name: str,
    ) -> list[VerticalElement]:
        if raw_points[0][2] is not None:
            raise LandXMLParseError(f"Alignment '{alignment_name}': <ProfAlign> cannot start with a <ParaCurve>.")

        if raw_points[-1][2] is not None:
            raise LandXMLParseError(f"Alignment '{alignment_name}': <ProfAlign> cannot end with a <ParaCurve>.")

        elements: list[VerticalElement] = []
        cursor_station, cursor_elevation, _ = raw_points[0]

        index = 1
        while index < len(raw_points):
            station, elevation, curve_lengths = raw_points[index]

            if curve_lengths is not None:
                prev_station, prev_elevation, _ = raw_points[index - 1]
                next_station, next_elevation, _ = raw_points[index + 1]

                incoming_grade = (elevation - prev_elevation) / (station - prev_station)
                outgoing_grade = (next_elevation - elevation) / (next_station - station)
                length_in, length_out = curve_lengths

                curve = VerticalCurve(
                    pvi_station=station,
                    pvi_elevation=elevation,
                    incoming_grade=incoming_grade,
                    outgoing_grade=outgoing_grade,
                    length_in=length_in,
                    length_out=length_out,
                )

                if compare(cursor_station, curve.pvc_station) < 0:
                    elements.append(
                        GradeSegment(
                            start_station=cursor_station,
                            end_station=curve.pvc_station,
                            start_elevation=cursor_elevation,
                            end_elevation=curve.pvc_elevation,
                        )
                    )

                elements.append(curve)
                cursor_station, cursor_elevation = (
                    curve.pvt_station,
                    curve.pvt_elevation,
                )
            elif compare(cursor_station, station) < 0:
                elements.append(
                    GradeSegment(
                        start_station=cursor_station,
                        end_station=station,
                        start_elevation=cursor_elevation,
                        end_elevation=elevation,
                    )
                )
                cursor_station, cursor_elevation = station, elevation

            index += 1

        return elements

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
