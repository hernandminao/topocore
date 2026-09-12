"""
topocore.features.side.resolver
==================================

Resolves left/right laterality for linear features (pavement edges,
by default) relative to a reference CENTERLINE, as a post-processing
step over an already-built `FeatureCollection`.

Why this is a separate, additive module rather than a change to
`FeatureBuilder`, `features/models.py`, or any catalog: laterality is
a property of the geometric *instance* (where this specific edge sits
relative to a specific centerline), not of the field code itself.
`BORDE` always means `FeatureType.PAVEMENT_EDGE` -- the catalog
correctly stops there. `FeatureType.PAVEMENT_EDGE_LEFT`/`_RIGHT`
would conflate a geometric relationship with a semantic type, and
`FeatureBuilder`'s own per-code-run loop (`_build_legacy`) processes
each run in isolation, with no visibility into other runs -- adding
cross-run reasoning there would mean restructuring already-tested,
PR19-hardened machinery for a concern that doesn't need to live
there. A post-processing resolver over the finished
`FeatureCollection` needs none of that: it reads `FeatureType.
CENTERLINE` and `FeatureType.PAVEMENT_EDGE` (already present because
the catalog already classifies `EJE`/`EJEVIAL`/`CL` and
`BORDE`/`BORDEPAV`/`PAV`/`CALZADA` correctly), and adds
`attributes["side"]` to the result.

Sign convention: reuses, rather than reinvents, the convention
already established in `topocore.analysis.profile.transversal`.
`TransversalProfile`'s own perpendicular offset direction
(`px, py = -uy, ux`, a 90-degree counter-clockwise rotation of the
axis tangent) is mathematically identical to
`cross(tangent, offset) > 0`.
This module's own LEFT/RIGHT split uses that exact same cross-product
sign, confirmed numerically equivalent before writing this module --
so "left" here means the same geometric side as a positive offset in
a transversal profile taken along the same centerline, not a second,
independently-invented convention.

Resolution priority, per feature:

    1. `attributes["side"]` already present (survey-provided, or an
       earlier stage's own explicit assignment) -- used as-is,
       tagged `SideMethod.EXPLICIT`. This is the only real way to
       represent an explicit CSV `SIDE` column today (see this
       module's own package docstring on why no new `SurveyPoint`
       field was added for it) -- `FeatureBuilder`'s own
       `_build_attributes` would need a caller-supplied mapping from
       point/run to a side value to carry it through automatically;
       until that's needed by real data, callers can set
       `attributes["side"]` on the built `Feature`s themselves
       before calling `resolve()`.
    2. Geometric inference against the single nearest CENTERLINE
       candidate, if unambiguous -- tagged `SideMethod.GEOMETRIC`,
       with `attributes["side_reference_feature_id"]` recording
       which CENTERLINE was used.
    3. `Side.UNKNOWN` -- no CENTERLINE within range, or 2+ candidates
       too close together to choose safely (parallel roads,
       intersections, roundabouts), or the edge point lies
       effectively on the centerline's own line. Tagged
       `SideMethod.UNKNOWN` or `SideMethod.AMBIGUOUS` respectively --
       these are NOT errors; guessing wrong here (e.g. at an
       intersection) would silently corrupt real survey data, which
       is worse than an honest "unknown".

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from topocore.features.models import Feature, FeatureCollection, FeatureType
from topocore.features.side.exceptions import SideResolutionError
from topocore.features.side.models import Side, SideMethod

_DEFAULT_MAX_DISTANCE = 30.0
_DEFAULT_AMBIGUITY_MARGIN = 0.10
_DEFAULT_CROSS_TOLERANCE = 1e-6


def _arc_length_midpoint(vertices_2d: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    The point at half the polyline's own total arc length -- used as
    the single representative point of a (possibly curved) edge for
    distance/side computation against a centerline.

    Degenerate case (every vertex coincident, zero total length):
    returns the first vertex -- the polyline has no meaningful
    direction to project against anyway.
    """
    if len(vertices_2d) == 1:
        return np.asarray(vertices_2d[0], dtype=np.float64)

    segment_vectors = np.diff(vertices_2d, axis=0)
    segment_lengths = np.hypot(segment_vectors[:, 0], segment_vectors[:, 1])
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = cumulative[-1]

    if total_length <= 0.0:
        return np.asarray(vertices_2d[0], dtype=np.float64)

    half = total_length / 2.0
    segment_index = int(np.searchsorted(cumulative, half))
    segment_index = max(1, min(segment_index, len(vertices_2d) - 1))

    segment_start_length = cumulative[segment_index - 1]
    segment_end_length = cumulative[segment_index]
    span = segment_end_length - segment_start_length

    t = 0.0 if span <= 0.0 else (half - segment_start_length) / span

    a = vertices_2d[segment_index - 1]
    b = vertices_2d[segment_index]
    return np.asarray(a + t * (b - a), dtype=np.float64)


def _closest_point_and_tangent(
    polyline_2d: NDArray[np.float64],
    query: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]] | None:
    """
    The closest point on `polyline_2d` to `query`, the distance to
    it, and the (unit) tangent direction of the specific segment
    that point lies on.

    Returns `None` if every segment is degenerate (zero length) --
    a polyline with no meaningful direction has no tangent to offer.
    """
    best_distance = math.inf
    best_point: NDArray[np.float64] | None = None
    best_tangent: NDArray[np.float64] | None = None

    for index in range(len(polyline_2d) - 1):
        a = polyline_2d[index]
        b = polyline_2d[index + 1]
        segment = b - a
        length_squared = float(segment @ segment)

        if length_squared <= 1e-18:
            continue

        t = float((query - a) @ segment) / length_squared
        t = max(0.0, min(1.0, t))
        point = a + t * segment
        distance = float(np.hypot(*(query - point)))

        if distance < best_distance:
            best_distance = distance
            best_point = point
            best_tangent = segment / math.sqrt(length_squared)

    if best_point is None or best_tangent is None:
        return None

    return best_point, best_distance, best_tangent


@dataclasses.dataclass(frozen=True, slots=True)
class SideResolver:
    """
    Parameters
    ----------
    target_types
        Which `FeatureType`s get laterality resolution. Defaults to
        `{FeatureType.PAVEMENT_EDGE}` only -- the type this
        capability was designed for; pass a wider set explicitly if
        another linear type (e.g. `FeatureType.CURB`) should also be
        resolved against the same centerlines.
    max_distance
        A CENTERLINE farther than this from a target feature's own
        representative point is not considered a candidate at all.
        Prevents a road on the far side of a project from ever being
        selected just because it's the "least far" of a bad set.
    ambiguity_margin
        If the second-closest candidate's distance is within this
        fraction of the closest one's distance (default 10%), the
        two are considered too close to safely choose between --
        resolves to `Side.UNKNOWN`/`SideMethod.AMBIGUOUS` rather
        than picking arbitrarily.
    cross_tolerance
        If the computed cross product's absolute value is at or
        below this, the point is considered to lie effectively ON
        the centerline's own line (not clearly to either side) --
        resolves to `Side.UNKNOWN`/`SideMethod.AMBIGUOUS`.
    """

    target_types: frozenset[FeatureType] = dataclasses.field(
        default_factory=lambda: frozenset({FeatureType.PAVEMENT_EDGE})
    )
    max_distance: float = _DEFAULT_MAX_DISTANCE
    ambiguity_margin: float = _DEFAULT_AMBIGUITY_MARGIN
    cross_tolerance: float = _DEFAULT_CROSS_TOLERANCE

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_distance) or self.max_distance <= 0:
            raise SideResolutionError(f"max_distance must be positive and finite; got {self.max_distance}.")
        if not (0.0 <= self.ambiguity_margin < 1.0):
            raise SideResolutionError(f"ambiguity_margin must be in [0, 1); got {self.ambiguity_margin}.")
        if not math.isfinite(self.cross_tolerance) or self.cross_tolerance < 0:
            raise SideResolutionError(f"cross_tolerance must be non-negative and finite; got {self.cross_tolerance}.")

    def resolve(self, collection: FeatureCollection) -> FeatureCollection:
        """
        Returns a NEW `FeatureCollection` (input is never mutated --
        `Feature` is frozen and `FeatureCollection.add()` only ever
        appends). Every feature whose `feature_type` is not in
        `target_types` is carried over unchanged, in original order;
        `feature_id`s are preserved exactly as given (this is a
        pure attribute-enrichment pass, not a renumbering one --
        call `.normalize_ids()` yourself afterward if needed).

        `crs` is preserved unchanged from `collection` -- resolving
        laterality only adds `side`/`side_method` attributes to
        existing features; it never moves geometry or changes what
        CRS it's expressed in.
        """
        centerlines = collection.by_type(FeatureType.CENTERLINE)

        resolved = FeatureCollection(crs=collection.crs)

        for feature in collection:
            if feature.feature_type not in self.target_types:
                resolved.add(feature)
                continue

            resolved.add(self._resolve_one(feature, centerlines))

        return resolved

    def _resolve_one(self, feature: Feature, centerlines: Sequence[Feature]) -> Feature:
        explicit = feature.attributes.get("side")

        if explicit is not None:
            return self._with_explicit(feature, explicit)

        if not centerlines:
            return self._with_result(feature, Side.UNKNOWN, SideMethod.UNKNOWN)

        query = _arc_length_midpoint(feature.geometry.vertices[:, :2])

        candidates: list[tuple[float, Feature, NDArray[np.float64], NDArray[np.float64]]] = []

        for centerline in centerlines:
            projection = _closest_point_and_tangent(centerline.geometry.vertices[:, :2], query)

            if projection is None:
                continue

            point, distance, tangent = projection

            if distance <= self.max_distance:
                candidates.append((distance, centerline, point, tangent))

        if not candidates:
            return self._with_result(feature, Side.UNKNOWN, SideMethod.UNKNOWN)

        candidates.sort(key=lambda candidate: candidate[0])

        if len(candidates) >= 2:
            closest_distance = candidates[0][0]
            second_distance = candidates[1][0]

            if (
                closest_distance <= 1e-12
                or (second_distance - closest_distance) / closest_distance < self.ambiguity_margin
            ):
                return self._with_result(feature, Side.UNKNOWN, SideMethod.AMBIGUOUS)

        distance, centerline, closest_point, tangent = candidates[0]
        offset = query - closest_point
        cross = tangent[0] * offset[1] - tangent[1] * offset[0]

        if abs(cross) <= self.cross_tolerance:
            return self._with_result(feature, Side.UNKNOWN, SideMethod.AMBIGUOUS)

        side = Side.LEFT if cross > 0 else Side.RIGHT

        return self._with_result(feature, side, SideMethod.GEOMETRIC, reference_feature_id=centerline.feature_id)

    def _with_explicit(self, feature: Feature, explicit_value: object) -> Feature:
        normalized = str(explicit_value).strip().lower()

        if normalized not in (Side.LEFT.value, Side.RIGHT.value):
            raise SideResolutionError(
                f"Feature {feature.feature_id}: attributes['side'] must be "
                f"'left' or 'right' (case-insensitive); got {explicit_value!r}."
            )

        new_attributes = dict(feature.attributes)
        new_attributes["side"] = normalized
        new_attributes["side_method"] = SideMethod.EXPLICIT.value

        return dataclasses.replace(feature, attributes=new_attributes)

    def _with_result(
        self,
        feature: Feature,
        side: Side,
        method: SideMethod,
        *,
        reference_feature_id: int | None = None,
    ) -> Feature:
        new_attributes = dict(feature.attributes)
        new_attributes["side"] = side.value
        new_attributes["side_method"] = method.value

        if reference_feature_id is not None:
            new_attributes["side_reference_feature_id"] = reference_feature_id

        return dataclasses.replace(feature, attributes=new_attributes)


__all__ = ["SideResolver"]
