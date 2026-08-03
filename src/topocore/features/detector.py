"""
topocore.features.detector
============================

Detector registry.

Kept separate from `manager.py` so registration (a load-time, global
concern — "what detectors exist") stays independent of orchestration
(a per-run concern — "which of them do I invoke, and how do I merge
their output"). `FeatureExtractionManager` composes `DetectorRegistry`
rather than owning the registration dict itself.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.features.base import BaseFeatureDetector
from topocore.features.exceptions import FeatureError
from topocore.features.models import FeatureCategory


class DetectorRegistry:
    """
    Global registry of available detector classes.

    A class-level dict, not per-instance state: "what detectors
    exist" is module-global information, populated once at import
    time as each `features/<category>/*.py` module registers itself
    — mirrors `ClassificationManager`'s `_ML_REGISTRY`.
    """

    _detectors: dict[str, type[BaseFeatureDetector]] = {}

    @classmethod
    def register(cls, detector_cls: type[BaseFeatureDetector]) -> type[BaseFeatureDetector]:
        """
        Register a detector class. Intended as a decorator::

            @DetectorRegistry.register
            class BreaklineDetector(BaseFeatureDetector):
                ...

        Raises
        ------
        FeatureError
            If a detector with the same `name()` is already
            registered. Silent overwrite is never allowed — it would
            mean the previous detector's registration disappears
            without any signal, which is exactly the kind of bug
            that's invisible until someone notices missing features
            in production output.
        """
        name = detector_cls().name()

        if name in cls._detectors:
            raise FeatureError(
                f"A detector named '{name}' is already registered "
                f"({cls._detectors[name].__qualname__}); "
                "choose a unique name() or remove the duplicate registration."
            )

        cls._detectors[name] = detector_cls
        return detector_cls

    @classmethod
    def get(cls, name: str) -> type[BaseFeatureDetector]:
        if name not in cls._detectors:
            raise FeatureError(f"Unknown detector: '{name}'. Available: {cls.available()}.")
        return cls._detectors[name]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._detectors.keys())

    @classmethod
    def for_category(cls, category: FeatureCategory) -> list[str]:
        return sorted(name for name, detector_cls in cls._detectors.items() if detector_cls.category == category)

    @classmethod
    def clear(cls) -> None:
        """
        Remove every registered detector.

        For test isolation only (so one test module's dummy
        detectors don't leak into another test's `available()`),
        never called in production code paths.
        """
        cls._detectors.clear()


__all__ = ["DetectorRegistry"]
