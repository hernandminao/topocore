"""
topocore.alignment.vertical_elements
======================================

Vertical alignment (design profile) element types.

``VerticalCurve`` uses the Hickerson (1964) "unequal-tangent" method
-- the standard treatment used by PennDOT/NCDOT/WYDOT and most route
surveying texts, verified against two independently published worked
examples (PDHonline Course L121, "Vertical Curves", instructor Jan
Van Sickle) before implementation:

    Symmetric:  PVC (10+00, 100.00 ft), g1=+2%, g2=-3%, L=600 ft
                -> elevation at 12+50 = 102.40 ft (matched exactly)

    Asymmetric: PVC (44+00, 741.25 ft), PVT (52+72.43, 737.25 ft),
                g1=-4%, g2=+3%, L1=431.00 ft, L2=441.43 ft
                -> CVC elevation at 48+31 = 731.64 ft (matched exactly)

**PVI is the intersection of the two straight tangent lines,
generally OFF the actual curve** -- a design/control point, not a
point the curve passes through. This was the key correction from an
earlier (incorrect) draft of this module, which had wrongly modeled
PVI as lying on the curve (borrowed, incorrectly, from the
horizontal alignment's PI, which behaves differently). Verified
against authoritative sources before re-implementing, not re-derived
from memory a second time.

The curve is built as two component parabolas, PVC-to-CVC and
CVC-to-PVT (CVC = point of Compound Vertical Curvature, at PVI's
station but generally NOT its elevation), sharing a common tangent
(slope) at CVC but not necessarily the same rate of curvature -- an
"abrupt change in curvature" at CVC is a documented, accepted
property of this method (some newer methods use a single
inclined-axis parabola specifically to avoid it; that is a
different, non-standard curve, out of scope here). The two rates
are still exposed (``rate_in``/``rate_out``) so this can be verified
directly rather than assumed.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

from topocore.alignment.exceptions import AlignmentGeometryError
from topocore.math.validation import validate_finite


@dataclass(frozen=True, slots=True)
class GradeSegment:
    """
    A straight tangent grade segment (constant slope).
    """

    start_station: float
    end_station: float
    start_elevation: float
    end_elevation: float

    def __post_init__(self) -> None:
        validate_finite(self.start_station)
        validate_finite(self.end_station)
        validate_finite(self.start_elevation)
        validate_finite(self.end_elevation)

        if self.end_station <= self.start_station:
            raise AlignmentGeometryError("GradeSegment.end_station must be greater than start_station.")

    @property
    def length(self) -> float:
        return self.end_station - self.start_station

    @property
    def grade(self) -> float:
        """
        Constant slope as a fraction (0.03 = 3%).
        """
        return (self.end_elevation - self.start_elevation) / self.length

    @property
    def grade_percent(self) -> float:
        return self.grade * 100.0

    def elevation_at(self, station: float) -> float:
        t = (station - self.start_station) / self.length
        return self.start_elevation + t * (self.end_elevation - self.start_elevation)

    def grade_at(self, _station: float) -> float:
        """
        Constant slope, independent of station -- the parameter
        exists only to match the uniform interface shared with
        ``VerticalCurve.grade_at`` (``DesignProfile`` dispatches on
        it generically, always positionally). The leading
        underscore signals "intentionally unused" to linters
        (ruff/Pylance/SonarQube) without needing a different
        suppression comment for each one.
        """
        return self.grade


@dataclass(frozen=True, slots=True)
class VerticalCurve:
    """
    A parabolic vertical curve (LandXML ``<ParaCurve>``), symmetric
    or asymmetric (``length_in`` == or != ``length_out``). See the
    module docstring for the Hickerson (1964) method this
    implements and the two worked examples it was verified against.
    """

    pvi_station: float
    pvi_elevation: float
    incoming_grade: float
    outgoing_grade: float
    length_in: float
    length_out: float

    def __post_init__(self) -> None:
        validate_finite(self.pvi_station)
        validate_finite(self.pvi_elevation)
        validate_finite(self.incoming_grade)
        validate_finite(self.outgoing_grade)

        if not math.isfinite(self.length_in) or self.length_in <= 0.0:
            raise AlignmentGeometryError("VerticalCurve.length_in must be finite and positive.")

        if not math.isfinite(self.length_out) or self.length_out <= 0.0:
            raise AlignmentGeometryError("VerticalCurve.length_out must be finite and positive.")

    @property
    def is_symmetric(self) -> bool:
        return math.isclose(self.length_in, self.length_out)

    @property
    def length(self) -> float:
        return self.length_in + self.length_out

    @property
    def pvc_station(self) -> float:
        return self.pvi_station - self.length_in

    @property
    def pvt_station(self) -> float:
        return self.pvi_station + self.length_out

    #: Aliases for the uniform interface shared with GradeSegment,
    #: used by DesignProfile's generic station-range dispatch.
    @property
    def start_station(self) -> float:
        return self.pvc_station

    @property
    def end_station(self) -> float:
        return self.pvt_station

    @property
    def pvc_elevation(self) -> float:
        """
        PVC is on both the actual curve AND the straight incoming
        tangent line (that is its definition) -- this straight-line
        formula is exact, in both the symmetric and asymmetric case.
        """
        return self.pvi_elevation - self.incoming_grade * self.length_in

    @property
    def pvt_elevation(self) -> float:
        """
        PVT is on both the actual curve AND the straight outgoing
        tangent line (that is its definition) -- exact, same
        reasoning as ``pvc_elevation``.
        """
        return self.pvi_elevation + self.outgoing_grade * self.length_out

    @property
    def mid_grade(self) -> float:
        """
        The single grade shared by both component parabolas at CVC
        (their common tangent point): ``(g1*L1 + g2*L2) / L``.
        Reduces to ``(g1 + g2) / 2`` when ``length_in == length_out``.
        """
        return (self.incoming_grade * self.length_in + self.outgoing_grade * self.length_out) / self.length

    @property
    def cvc_elevation(self) -> float:
        """
        Elevation of the actual curve at ``pvi_station`` (CVC, point
        of Compound Vertical Curvature). NOT equal to
        ``pvi_elevation`` in general -- PVI is the off-curve tangent
        intersection; the curve passes below (crest) or above (sag)
        it by the classic "tangent offset" amount:

            cvc_elevation = pvi_elevation +
                (length_in * length_out * (outgoing_grade - incoming_grade))
                / (2 * length)

        Verified against two independently published worked examples
        -- see module docstring.
        """
        return self.pvi_elevation + (self.length_in * self.length_out * (self.outgoing_grade - self.incoming_grade)) / (
            2.0 * self.length
        )

    @property
    def rate_in(self) -> float:
        """
        Rate of grade change on the PVC-to-CVC component parabola.
        """
        return (self.mid_grade - self.incoming_grade) / self.length_in

    @property
    def rate_out(self) -> float:
        """
        Rate of grade change on the CVC-to-PVT component parabola.
        Generally different from ``rate_in`` -- an "abrupt change in
        curvature" at CVC is an accepted property of this method,
        not a bug; equal only when ``is_symmetric``.
        """
        return (self.outgoing_grade - self.mid_grade) / self.length_out

    def elevation_at(self, station: float) -> float:
        if station <= self.pvi_station:
            x = station - self.pvc_station
            return self.pvc_elevation + self.incoming_grade * x + 0.5 * self.rate_in * x**2

        x = station - self.pvi_station
        return self.cvc_elevation + self.mid_grade * x + 0.5 * self.rate_out * x**2

    def grade_at(self, station: float) -> float:
        if station <= self.pvi_station:
            x = station - self.pvc_station
            return self.incoming_grade + self.rate_in * x

        x = station - self.pvi_station
        return self.mid_grade + self.rate_out * x


VerticalElement: TypeAlias = GradeSegment | VerticalCurve


__all__ = [
    "GradeSegment",
    "VerticalCurve",
    "VerticalElement",
]
