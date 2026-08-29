"""
topocore.processing.classification.lightgbm
============================================

LightGBM classifier for point cloud classification.

LightGBM is an optional dependency. The import is attempted once at
module load time; if the package is not installed, the module still
imports cleanly, but instantiating ``LightGBMClassifier`` raises a
clear, actionable ``ClassificationError`` instead of an opaque
``ImportError`` from deep inside a training call.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import importlib
from typing import Any, override

from topocore.processing.exceptions import ClassificationError

from .ml import MachineLearningClassifier

# Se declara como Any para permitir asignarle None sin error [assignment]
_lgb: Any = None

try:
    _lgb = importlib.import_module("lightgbm")
    _LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without lightgbm installed
    _LIGHTGBM_AVAILABLE = False

_NOT_INSTALLED_ERROR = (
    "LightGBM is not installed. Install it with `pip install lightgbm` "
    "(or `uv add lightgbm`) to use LightGBMClassifier."
)


class LightGBMClassifier(MachineLearningClassifier):
    """
    LightGBM classifier for point cloud classification.

    Requires the optional ``lightgbm`` package. Use
    :meth:`is_available` to check installation status without
    triggering an exception.

    Parameters
    ----------
    n_estimators
        Number of boosting rounds.
    max_depth
        Maximum tree depth. ``-1`` means no limit.
    learning_rate
        Shrinkage applied to each tree's contribution.
    num_leaves
        Maximum number of leaves per tree.
    random_state
        Random seed for reproducibility.
    n_jobs
        Number of parallel threads.
    feature_names
        Features used by the classifier.
    k
        Number of neighbors for geometric features.
    radius
        Radius used for neighborhood computations.
    ground_method
        GroundManager method used to geometrically classify ground
        for the "height_above_ground" feature (see
        MachineLearningClassifier's docstring).

    Raises
    ------
    ClassificationError
        If the ``lightgbm`` package is not installed, or if any
        constructor parameter is outside its valid range (PR21
        remediation, ML-VALIDATION-001 -- see RandomForestClassifier's
        own docstring for the full rationale).
    """

    __slots__ = (
        "_learning_rate",
        "_max_depth",
        "_n_estimators",
        "_n_jobs",
        "_num_leaves",
        "_random_state",
    )

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        random_state: int | None = 42,
        n_jobs: int = -1,
        feature_names: list[str] | None = None,
        k: int = 10,
        radius: float = 1.0,
        ground_method: str = "grid",
    ) -> None:
        """
        Initialize LightGBM classifier.
        """
        if not _LIGHTGBM_AVAILABLE:
            raise ClassificationError(_NOT_INSTALLED_ERROR)

        if n_estimators <= 0:
            raise ClassificationError("n_estimators must be greater than zero.")

        if learning_rate <= 0:
            raise ClassificationError("learning_rate must be greater than zero.")

        if num_leaves <= 1:
            raise ClassificationError("num_leaves must be greater than 1.")

        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._num_leaves = num_leaves
        self._random_state = random_state
        self._n_jobs = n_jobs

        assert _lgb is not None  # narrows type for mypy after the availability check above

        model: Any = _lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        )

        super().__init__(
            model=model,
            feature_names=feature_names,
            k=k,
            radius=radius,
            ground_method=ground_method,
        )

    @staticmethod
    def is_available() -> bool:
        """
        Return whether the optional LightGBM dependency is installed.
        """
        return _LIGHTGBM_AVAILABLE

    @override
    def name(self) -> str:
        """
        Return classifier name.
        """
        return "lightgbm"
