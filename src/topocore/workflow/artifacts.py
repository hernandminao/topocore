"""
topocore.workflow.artifacts
===============================

The Artifact Store: holds the most recent version of each artifact
type a Workflow has produced, plus a monotonically increasing
version counter per type.

Deliberately NOT a dependency graph: it never knows which artifact
types derive from which others, never invalidates anything
automatically, and never resolves execution order. It only answers
two questions -- "is X available?" and "what version is X at?" --
so that each Workflow stage method can validate its own
preconditions before running. See the Fase 1 architectural audit
for the full rationale (recorded in the project history, not
reproduced here).

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
from typing import Any


class ArtifactType(StrEnum):
    """
    The 8 reusable artifact types a Workflow can hold -- one current
    version each. Deliberately excludes GROUND_MASK (no real
    consumer; the point counts it would justify are cheaper to
    derive from GROUND_CLOUD.point_count than to store the raw
    boolean array) and DXF/GPKG reports (exports are lateral
    operations, never written to the store).
    """

    SURVEY_POINT_SET = "survey_point_set"
    POINT_CLOUD = "point_cloud"
    GROUND_CLOUD = "ground_cloud"
    CLASSIFICATION_RESULT = "classification_result"
    TIN = "tin"
    DTM = "dtm"
    CONTOURS = "contours"
    FEATURE_COLLECTION = "feature_collection"


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """
    One artifact, tagged with its version.

    Parameters
    ----------
    version
        1-based, monotonically increasing per `ArtifactType` within
        a single `Workflow`. Never global, never meaningful across
        two `snapshot()` branches once they've diverged -- see
        `ArtifactStore.snapshot()`.
    artifact
        The actual object (`PointCloud`, `TIN`, `FeatureCollection`,
        ...). Untyped here deliberately: `Workflow` validates
        presence/type at each stage method's call site, not through
        generics (Python doesn't offer a maintainable way to type a
        fluid, type-changing pipeline -- a documented decision, not
        an oversight).
    """

    version: int
    artifact: Any


class ArtifactStore:
    """
    Holds the current `ArtifactRecord` for each `ArtifactType` a
    Workflow has produced so far -- never more than one record per
    type. The previous version is simply overwritten; the full
    history of every version produced lives in `WorkflowResult`'s
    append-only history, not here.
    """

    __slots__ = ("_records",)

    def __init__(self) -> None:
        self._records: dict[ArtifactType, ArtifactRecord] = {}

    def has(self, artifact_type: ArtifactType) -> bool:
        return artifact_type in self._records

    def get(self, artifact_type: ArtifactType) -> Any:
        """
        Raises
        ------
        KeyError
            If `artifact_type` has never been produced.
        """
        return self._records[artifact_type].artifact

    def get_or_none(self, artifact_type: ArtifactType) -> Any | None:
        record = self._records.get(artifact_type)
        return record.artifact if record is not None else None

    def version_of(self, artifact_type: ArtifactType) -> int | None:
        record = self._records.get(artifact_type)
        return record.version if record is not None else None

    def set(self, artifact_type: ArtifactType, artifact: Any) -> int:
        """
        Store `artifact` as the current version of `artifact_type`,
        incrementing its version counter.

        Only ever called after a stage finishes successfully -- a
        failed stage must never call this (versions only advance on
        SUCCESS, per the project's frozen contract).

        Returns
        -------
        int
            The new version number.
        """
        current = self._records.get(artifact_type)
        new_version = 1 if current is None else current.version + 1
        self._records[artifact_type] = ArtifactRecord(version=new_version, artifact=artifact)
        return new_version

    def snapshot(self) -> ArtifactStore:
        """
        Shallow copy: a new `ArtifactStore` with the same set of
        `ArtifactRecord`s (same version numbers, same artifact
        object references -- never deep-copied, since every
        artifact type here is immutable). O(number of artifact types
        currently held), never O(size of the artifacts themselves).

        After this call, the two stores evolve completely
        independently: calling `.set()` on one never affects the
        other's version counters or records.
        """
        clone = ArtifactStore()
        clone._records = dict(self._records)
        return clone

    def __repr__(self) -> str:
        parts = ", ".join(f"{artifact_type.value}=v{record.version}" for artifact_type, record in self._records.items())
        return f"ArtifactStore({parts})"


__all__ = ["ArtifactRecord", "ArtifactStore", "ArtifactType"]
