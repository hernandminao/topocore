"""
topocore.alignment.algorithms.spiral
======================================

Clothoid (Euler spiral) geometry evaluation.

The low-level Fresnel-based local geometry (``_clothoid_local_uv``)
lives in ``topocore.alignment.elements``, not here -- it is shared
between ``SpiralElement.__post_init__`` (construction-time chord
consistency validation) and this module (actually evaluating points
along the curve). See ``elements.py`` for the reasoning.

This module places the local curve in global space with a single
rigid transform (rotation + translation), derived once by aligning
the local chord (zero-curvature point -> finite-radius point) onto
the actual global chord between ``start``/``end`` -- never a
per-point patch. Because ``SpiralElement`` already validated that
chord lengths match at construction, ``l'=0``/``l'=length``
reproduce ``start``/``end`` exactly as a consequence of the
transform being exact, not because either endpoint is special-cased.

Entry vs. exit mirroring
--------------------------
The canonical local frame's +u axis is the tangent heading at l'=0
(the zero-curvature point), in the direction of INCREASING l'. For
an entry spiral, l' increases in the same direction as physical
travel (start -> end), so ``element.clockwise`` (the physical
turning direction, as recorded in LandXML's ``rot`` attribute)
applies directly to the local-frame mirroring. For an exit spiral,
l'=0 is at ``end`` and l' increases TOWARD ``start`` -- the OPPOSITE
direction to physical travel -- so the local-frame mirroring must be
inverted (``_local_frame_clockwise``), or the resulting heading/PI
comes out wrong. This does NOT affect ``curvature_at``'s sign
(which describes physical bending sense along actual travel, and
correctly uses ``element.clockwise`` directly, unflipped) -- only
position/heading. Confirmed against a genuine PLATEIA 2007 export
(Sample_Plateia2007LandXML11.XML, "PREHODNICA 2", an exit spiral):
the chord-length check alone did not catch this, since
``hypot(u, v)`` is insensitive to the sign of ``v`` -- only the
heading/PI comparison exposed it. See the PR18C session notes.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

from topocore.alignment.elements import SpiralElement, _clothoid_local_uv
from topocore.geometry.point2d import Point2D


def _is_entry(element: SpiralElement) -> bool:
    """
    True if ``element.start`` is the zero-curvature ("straight") end
    -- i.e. this is an entry (line-to-curve) spiral, as opposed to
    an exit (curve-to-line) spiral.
    """
    return element.radius_start == math.inf


def _finite_radius(element: SpiralElement) -> float:
    return element.radius_end if _is_entry(element) else element.radius_start


def _local_frame_clockwise(element: SpiralElement) -> bool:
    """
    See module docstring's "Entry vs. exit mirroring" note.
    """
    return element.clockwise if _is_entry(element) else not element.clockwise


def _rigid_transform(element: SpiralElement) -> tuple[Point2D, float]:
    """
    The rigid transform (zero-curvature global point, rotation angle
    in radians) mapping the canonical local frame to global space.

    Chord-length consistency was already validated in
    ``SpiralElement.__post_init__`` -- this function does not
    re-validate it, only derives the rotation.
    """
    local_u_end, local_v_end = _clothoid_local_uv(
        _finite_radius(element), element.length, _local_frame_clockwise(element), element.length
    )

    zero_point = element.start if _is_entry(element) else element.end
    finite_point = element.end if _is_entry(element) else element.start

    global_chord = zero_point.vector_to(finite_point)

    rotation = math.atan2(global_chord.y, global_chord.x) - math.atan2(local_v_end, local_u_end)

    return zero_point, rotation


def evaluate_spiral(element: SpiralElement, local_distance: float) -> Point2D:
    """
    Compute the XY position at ``local_distance`` along the spiral,
    measured from ``element.start`` (0 at ``start``, ``element.length``
    at ``end``, regardless of whether this is an entry or exit spiral).
    """
    zero_point, rotation = _rigid_transform(element)

    arc_param = local_distance if _is_entry(element) else element.length - local_distance

    local_u, local_v = _clothoid_local_uv(
        _finite_radius(element), element.length, _local_frame_clockwise(element), arc_param
    )

    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)

    global_dx = local_u * cos_r - local_v * sin_r
    global_dy = local_u * sin_r + local_v * cos_r

    return Point2D(zero_point.x + global_dx, zero_point.y + global_dy)


def curvature_at(element: SpiralElement, local_distance: float) -> float:
    """
    Signed curvature (1/radius) at ``local_distance`` along the
    spiral, measured from ``element.start``.

    Magnitude varies linearly with the distance from the
    zero-curvature point (0 there, ``1/R`` at the finite-radius end)
    regardless of whether this is an entry or exit spiral.

    Sign follows the standard mathematical convention: positive for
    a counterclockwise (left) turn, negative for clockwise (right)
    -- matching the sign a finite-difference estimate from evaluated
    points naturally produces (see the Entrega 2 test suite's
    cross-check against ``evaluate_spiral``). Uses
    ``element.clockwise`` directly (the physical turning sense along
    actual travel), NOT the entry/exit-mirrored local-frame flag --
    unlike position/heading, physical curvature sign does not invert
    between entry and exit spirals.
    """
    a_squared = _finite_radius(element) * element.length
    arc_param = local_distance if _is_entry(element) else element.length - local_distance

    magnitude = arc_param / a_squared

    return -magnitude if element.clockwise else magnitude


__all__ = [
    "curvature_at",
    "evaluate_spiral",
]
