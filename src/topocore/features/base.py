"""
topocore.features.base
========================

Abstract base class for feature detectors.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from topocore.features.exceptions import DetectionError
from topocore.features.models import ContextField, FeatureCategory, FeatureCollection, FeatureMetadata
from topocore.features.protocols import DetectionContext


class BaseFeatureDetector(ABC):
    """
    Base class for all feature detectors.

    Subclasses declare `category` and `required_inputs` as class
    attributes, and implement `_detect`. `detect()` (the public
    entry point called by `FeatureExtractionManager`) validates
    `required_inputs` against the given `DetectionContext` before
    delegating to `_detect`, so every detector gets consistent,
    early, readable errors instead of an `AttributeError` deep
    inside its algorithm.
    """

    category: ClassVar[FeatureCategory]
    version: ClassVar[str] = "1.0"
    required_inputs: ClassVar[frozenset[ContextField]] = frozenset()

    def detect(self, context: DetectionContext) -> FeatureCollection:
        """
        Run this detector against the given context.

        Parameters
        ----------
        context
            Bundle of available inputs.

        Returns
        -------
        FeatureCollection
            Detected features (may be empty if none were found).

        Raises
        ------
        DetectionError
            If a required input is missing from `context`.
        """
        missing = [required for required in self.required_inputs if getattr(context, required.value, None) is None]

        if missing:
            names = sorted(m.value for m in missing)
            raise DetectionError(
                f"{self.name()} requires {names}, which {'is' if len(names) == 1 else 'are'} missing from the context."
            )

        return self._detect(context)

    @abstractmethod
    def _detect(self, context: DetectionContext) -> FeatureCollection:
        """Subclasses implement the actual detection algorithm here."""
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """Return a short, unique identifier for this detector."""
        raise NotImplementedError

    def _metadata(
        self,
        inputs_used: frozenset[ContextField] = frozenset(),
        **extra: object,
    ) -> FeatureMetadata:
        """
        Convenience helper for subclasses to build a `FeatureMetadata`
        for a feature they're about to emit, without repeating
        `name()`/`version` at every call site.

        Parameters
        ----------
        inputs_used
            `ContextField`s actually consumed to produce this
            specific feature.
        **extra
            Extra provenance key/value pairs.
        """
        return FeatureMetadata(
            detector=self.name(),
            version=self.version,
            inputs_used=inputs_used,
            extra=extra,
        )

    def __call__(self, context: DetectionContext) -> FeatureCollection:
        return self.detect(context)


__all__ = ["BaseFeatureDetector"]
