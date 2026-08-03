"""
topocore.features.manager
===========================

Feature extraction manager.

Coordinates registered detectors (via `DetectorRegistry`), running
each against a shared `DetectionContext` and merging their results
into one `FeatureCollection` with normalized, collision-free IDs.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import logging

from topocore.features.detector import DetectorRegistry
from topocore.features.exceptions import DetectionError
from topocore.features.models import FeatureCategory, FeatureCollection
from topocore.features.protocols import DetectionContext

logger = logging.getLogger(__name__)


class FeatureExtractionManager:
    """
    Runs registered feature detectors and aggregates their results.

    Parameters
    ----------
    strict
        If ``True`` (default), a detector whose `required_inputs`
        aren't satisfied by the given context raises `DetectionError`.
        If ``False``, that detector is silently skipped — useful for
        `detect_all()` runs where the caller only has, say, a point
        cloud and classification but no TIN, and wants whatever
        subset of detectors can still run.
    """

    __slots__ = ("_strict",)

    def __init__(self, *, strict: bool = True) -> None:
        self._strict = strict

    @property
    def available_detectors(self) -> list[str]:
        return DetectorRegistry.available()

    def detectors_for_category(self, category: FeatureCategory) -> list[str]:
        return DetectorRegistry.for_category(category)

    def detect(
        self,
        detector_name: str,
        context: DetectionContext,
    ) -> FeatureCollection:
        """
        Run a single named detector.

        The returned collection always has normalized ``1..N``
        `feature_id`s (see `FeatureCollection.normalize_ids`).

        Raises
        ------
        FeatureError
            If no detector is registered under `detector_name`.
        DetectionError
            If a required input is missing from `context`.
        """
        detector = DetectorRegistry.get(detector_name)()
        result = detector.detect(context)
        result.normalize_ids()
        return result

    def detect_all(
        self,
        context: DetectionContext,
        categories: list[FeatureCategory] | None = None,
    ) -> FeatureCollection:
        """
        Run every registered detector (optionally filtered by
        category) and merge results into one collection.

        Parameters
        ----------
        context
            Shared inputs passed to every detector.
        categories
            If given, only detectors in these categories run.

        Returns
        -------
        FeatureCollection
            Merged results from every detector that ran, with
            normalized ``1..N`` `feature_id`s across the whole
            collection (each detector's local IDs are discarded).
        """
        result = FeatureCollection()

        if categories is None:
            names = DetectorRegistry.available()
        else:
            names = sorted({name for category in categories for name in DetectorRegistry.for_category(category)})

        for name in names:
            detector = DetectorRegistry.get(name)()

            try:
                partial = detector.detect(context)
            except DetectionError:
                if self._strict:
                    raise
                logger.info("Skipping detector '%s': required inputs not available.", name)
                continue

            result.extend(partial)

        result.normalize_ids()
        return result

    def __call__(
        self,
        context: DetectionContext,
        categories: list[FeatureCategory] | None = None,
    ) -> FeatureCollection:
        return self.detect_all(context, categories)


__all__ = ["FeatureExtractionManager"]
