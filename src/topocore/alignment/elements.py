"""
topocore.alignment.elements
=============================

Horizontal alignment element types.

Deliberately separated from ``models.py`` (which holds ``Alignment``,
the orchestrating model): these are low-level data types that
``topocore.alignment.algorithms`` dispatches on, with no dependency
of their own on ``algorithms`` -- the same layering already used by
``topocore.geometry.point3d.Point3D`` relative to
``topocore.terrain.algorithms``.

Scope of this delivery (Entrega 1, PR18C): ``LineElement`` and
``ArcElement`` only. ``SpiralElement`` (clothoid) is deferred to
Entrega 2.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math
from dataclasses import InitVar, dataclass, field
from typing import TypeAlias

# mypy: disable-error-code=import-untyped
from scipy.special import fresnel

from topocore.alignment.exceptions import AlignmentGeometryError
from topocore.geometry.point2d import Point2D
from topocore.math.config import DEFAULT_MATH_CONFIG
from topocore.math.tolerance import is_close
from topocore.math.validation import validate_positive


def _is_close_with_tolerance(a: float, b: float, tolerance: float | None) -> bool:
    """
    Same comparison as ``topocore.math.tolerance.is_close`` (the
    project's default, ``DEFAULT_MATH_CONFIG.absolute_tolerance``),
    with an optional override for ``abs_tol``.

    Used only by ``ArcElement``/``SpiralElement`` to accept an
    optional ``tolerance`` constructor parameter -- see their
    docstrings. ``DEFAULT_MATH_CONFIG`` itself is never modified;
    every other caller of ``is_close`` throughout TopoCore is
    unaffected.
    """
    if tolerance is None:
        return is_close(a, b)

    return math.isclose(a, b, rel_tol=DEFAULT_MATH_CONFIG.relative_tolerance, abs_tol=tolerance)


def _clothoid_local_uv(
    finite_radius: float,
    length: float,
    clockwise: bool,
    arc_param: float,
) -> tuple[float, float]:
    """
    Canonical local clothoid ``(u, v)`` at ``arc_param`` (``l'``,
    measured from the zero-curvature point, ``0 <= arc_param <= length``),
    for a clothoid of total length ``length`` reaching ``finite_radius``
    at its curved end.

    Shared, low-level helper: used both by ``SpiralElement.__post_init__``
    (to validate that ``start``/``end`` are geometrically consistent
    with ``radius``/``length`` -- see its docstring) and by
    ``topocore.alignment.algorithms.spiral`` (to actually evaluate
    points along the curve). Kept here, in ``elements.py``, rather
    than in ``algorithms/``, specifically so construction-time
    validation does not need to import ``algorithms`` -- see the
    module docstring's note on layering.

    ``scipy.special.fresnel`` returns ``(S, C)``, NOT ``(C, S)`` --
    this function unpacks them in that order deliberately. A
    clockwise-bending spiral mirrors ``v -> -v``.
    """
    a = math.sqrt(finite_radius * length)
    scale = a * math.sqrt(math.pi)
    z = arc_param / scale

    s, c = fresnel(z)  # scipy returns (S, C) -- not (C, S).

    u = float(scale * c)
    v = float(scale * s)

    if clockwise:
        v = -v

    return u, v


@dataclass(frozen=True, slots=True)
class LineElement:
    """
    A straight horizontal alignment element (LandXML ``<Line>``).
    """

    start: Point2D
    end: Point2D

    def __post_init__(self) -> None:
        if self.start.almost_equals(self.end):
            raise AlignmentGeometryError("LineElement start and end coincide (zero-length line).")


@dataclass(frozen=True, slots=True)
class ArcElement:
    """
    A circular arc horizontal alignment element (LandXML ``<Curve>``).

    ``start``/``end`` must both lie at ``radius`` distance from
    ``center`` -- this is validated at construction, not assumed.

    A full-circle arc (``start`` == ``end``) is not supported: it is
    ambiguous which of the two possible sweep directions/lengths is
    meant from geometry alone, and is vanishingly rare in real road
    alignments. This is a deliberate, documented limitation, not an
    oversight.

    ``tolerance`` (construction-time only, not stored on the
    instance): overrides the absolute tolerance used for the
    start/end-vs-radius consistency check, in place of
    ``topocore.math.config.DEFAULT_MATH_CONFIG.absolute_tolerance``.
    Exists specifically for ``topocore.io.landxml.LandXMLReader``:
    real-world LandXML exports (confirmed empirically against a
    genuine Autodesk Civil 3D 2007 file -- see the PR18C session
    notes) commonly have ``Start``/``Center``/``End``/``radius``
    that are mutually consistent only to ~1e-8, not to the domain's
    default ~1e-9 -- a property of serialized engineering data, not
    a TopoCore precision guarantee. Leave as ``None`` (the default)
    for the strict domain tolerance; direct API construction is
    unaffected.
    """

    start: Point2D
    end: Point2D
    center: Point2D
    radius: float
    clockwise: bool
    tolerance: InitVar[float | None] = field(default=None)

    def __post_init__(self, tolerance: float | None) -> None:
        validate_positive(self.radius)

        if self.start.almost_equals(self.end):
            raise AlignmentGeometryError("ArcElement start and end coincide; full-circle arcs are not supported.")

        start_radius = self.center.distance_to(self.start)
        end_radius = self.center.distance_to(self.end)

        if not _is_close_with_tolerance(start_radius, self.radius, tolerance):
            raise AlignmentGeometryError(
                f"ArcElement.start is {start_radius} from center, expected radius {self.radius}."
            )

        if not _is_close_with_tolerance(end_radius, self.radius, tolerance):
            raise AlignmentGeometryError(f"ArcElement.end is {end_radius} from center, expected radius {self.radius}.")


def _is_positive_radius_or_infinite(value: float) -> bool:
    return value == math.inf or (math.isfinite(value) and value > 0.0)


@dataclass(frozen=True, slots=True)
class SpiralElement:
    """
    A clothoid (Euler spiral) transition element (LandXML ``<Spiral>``).

    Exactly one of ``radius_start``/``radius_end`` must be
    ``math.inf`` (the "straight" end, zero curvature); the other
    must be a finite positive radius. A spiral between two distinct
    finite radii (a compound curve-to-curve transition) is out of
    scope for this delivery -- it is a real but comparatively rare
    case in road design and deserves its own verification pass
    rather than being folded silently into this one.

    ``pi`` (the tangent intersection point) is stored but not
    cross-validated against ``radius``/``length`` in this delivery
    -- it is informational, taken as given from the source LandXML.

    ``tolerance`` (construction-time only, not stored on the
    instance): overrides the absolute tolerance used for the chord-
    and ``pi``-consistency checks, in place of
    ``topocore.math.config.DEFAULT_MATH_CONFIG.absolute_tolerance``.
    Same reasoning and intended caller (``topocore.io.landxml.LandXMLReader``)
    as ``ArcElement.tolerance`` -- see its docstring.
    """

    start: Point2D
    end: Point2D
    pi: Point2D
    radius_start: float
    radius_end: float
    length: float
    clockwise: bool
    tolerance: InitVar[float | None] = field(default=None)

    def __post_init__(self, tolerance: float | None) -> None:
        if self.start.almost_equals(self.end):
            raise AlignmentGeometryError("SpiralElement start and end coincide (zero-length spiral).")

        if not math.isfinite(self.length) or self.length <= 0.0:
            raise AlignmentGeometryError("SpiralElement.length must be finite and positive.")

        if not _is_positive_radius_or_infinite(self.radius_start):
            raise AlignmentGeometryError("SpiralElement.radius_start must be a positive radius or math.inf.")

        if not _is_positive_radius_or_infinite(self.radius_end):
            raise AlignmentGeometryError("SpiralElement.radius_end must be a positive radius or math.inf.")

        start_infinite = self.radius_start == math.inf
        end_infinite = self.radius_end == math.inf

        if start_infinite and end_infinite:
            raise AlignmentGeometryError(
                "SpiralElement cannot have both radius_start and radius_end infinite "
                "(that describes a straight line, not a spiral -- use LineElement)."
            )

        if not start_infinite and not end_infinite:
            raise AlignmentGeometryError(
                "SpiralElement requires exactly one infinite radius (a line-to-curve or "
                "curve-to-line transition). Curve-to-curve compound spirals (both radii "
                "finite) are out of scope for this delivery."
            )

        finite_radius = self.radius_end if start_infinite else self.radius_start

        # The canonical local frame's +u axis is the tangent heading
        # at l'=0 (the zero-curvature point) in the direction of
        # INCREASING l'. For an entry spiral (start_infinite), l'
        # increases in the same direction as physical travel
        # (start -> end), so `clockwise` applies directly. For an
        # exit spiral, l'=0 is at `end` and l' increases TOWARD
        # `start` -- i.e. l' increases in the direction OPPOSITE to
        # physical travel -- so the mirroring must be inverted, or
        # the resulting tangent heading (and therefore 'pi') comes
        # out wrong by roughly 180 degrees minus the deflection
        # angle. Confirmed against a genuine PLATEIA 2007 export
        # (Sample_Plateia2007LandXML11.XML, "PREHODNICA 2", an exit
        # spiral): the chord-length check alone did NOT catch this,
        # because hypot(u, v) is insensitive to the sign of v --
        # only the heading/PI comparison exposed it. See the PR18C
        # session notes.
        local_frame_clockwise = self.clockwise if start_infinite else not self.clockwise

        local_u_end, local_v_end = _clothoid_local_uv(finite_radius, self.length, local_frame_clockwise, self.length)
        local_chord_length = math.hypot(local_u_end, local_v_end)

        zero_point = self.start if start_infinite else self.end
        finite_point = self.end if start_infinite else self.start
        global_chord = zero_point.vector_to(finite_point)
        global_chord_length = global_chord.length

        if not _is_close_with_tolerance(local_chord_length, global_chord_length, tolerance):
            raise AlignmentGeometryError(
                "SpiralElement geometry is inconsistent: the chord length implied by "
                f"radius/length ({local_chord_length:.6f}) does not match the actual distance "
                f"between the zero-curvature point and the finite-radius point "
                f"({global_chord_length:.6f})."
            )

        rotation = math.atan2(global_chord.y, global_chord.x) - math.atan2(local_v_end, local_u_end)

        # PI is where the tangent line at the zero-curvature point
        # (local +u axis) intersects the tangent line at the
        # finite-radius end (through (local_u_end, local_v_end), at
        # heading angle theta_s = length / (2*finite_radius), signed
        # to match the same local_frame_clockwise mirroring already
        # applied to local_v_end). Standard "long tangent" identity:
        # PI_local = (local_u_end - local_v_end * cot(theta_s), 0).
        theta_s = self.length / (2.0 * finite_radius)
        signed_theta_s = -theta_s if local_frame_clockwise else theta_s

        if is_close(math.sin(signed_theta_s), 0.0):
            raise AlignmentGeometryError(
                "SpiralElement has a degenerate (near-zero or near-pi) deflection angle; "
                "cannot validate 'pi' against it."
            )

        pi_local_u = local_u_end - local_v_end / math.tan(signed_theta_s)

        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        expected_pi_x = zero_point.x + pi_local_u * cos_r
        expected_pi_y = zero_point.y + pi_local_u * sin_r

        if not (
            _is_close_with_tolerance(expected_pi_x, self.pi.x, tolerance)
            and _is_close_with_tolerance(expected_pi_y, self.pi.y, tolerance)
        ):
            raise AlignmentGeometryError(
                f"SpiralElement.pi ({self.pi.x:.6f}, {self.pi.y:.6f}) is not geometrically "
                "consistent with start/end/radius/length/clockwise; expected approximately "
                f"({expected_pi_x:.6f}, {expected_pi_y:.6f})."
            )


HorizontalElement: TypeAlias = LineElement | ArcElement | SpiralElement


__all__ = [
    "ArcElement",
    "HorizontalElement",
    "LineElement",
    "SpiralElement",
]
