"""
topocore.processing.classification.random_forest
================================================

Random Forest classifier for point cloud classification.

This module provides the Random Forest implementation used by
TopoCore. The generic machine learning workflow is implemented by
MachineLearningClassifier; this class only configures the underlying
scikit-learn estimator.

scikit-learn is an optional dependency (extra ``ml``). The import is
attempted once at module load time; if the package is not installed,
the module still imports cleanly, but instantiating
RandomForestClassifier raises a clear ClassificationError instead of
an opaque ImportError deep inside training.

References
----------
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Any, override

from topocore.processing.exceptions import ClassificationError

from .ml import MachineLearningClassifier

try:
    from sklearn.ensemble import RandomForestClassifier as _SklearnRandomForest  # type: ignore

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without scikit-learn installed
    _SklearnRandomForest = None
    _SKLEARN_AVAILABLE = False

_NOT_INSTALLED_ERROR = (
    "scikit-learn is not installed. Install it with `pip install topocore[ml]` "
    "(or `pip install scikit-learn`) to use RandomForestClassifier."
)


class RandomForestClassifier(MachineLearningClassifier):
    """
    Random Forest classifier for point cloud classification.

    Parameters
    ----------
    n_estimators
        Number of trees in the forest.
    max_depth
        Maximum depth of each tree.
    min_samples_split
        Minimum samples required to split an internal node.
    min_samples_leaf
        Minimum samples required at a leaf node.
    random_state
        Random seed for reproducibility.
    n_jobs
        Number of parallel jobs.
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
        If the ``scikit-learn`` package is not installed, or if any
        constructor parameter is outside its valid range.

        PR21 remediation (ML-VALIDATION-001): previously, parameter
        validation raised plain ValueError -- a type unrelated to
        ClassificationError/ProcessingError/TopoCoreError, breaking
        ClassificationManager's own established exception-
        normalization contract (confirmed directly: a ValueError
        here was not caught by `except ClassificationError` around
        the manager). Now consistent with the "backend not
        installed" check below, which already used
        ClassificationError.
    """

    __slots__ = (
        "_max_depth",
        "_max_features",
        "_min_samples_leaf",
        "_min_samples_split",
        "_n_estimators",
        "_n_jobs",
        "_random_state",
    )

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: str | float | None = "sqrt",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int | None = 42,
        n_jobs: int = -1,
        feature_names: list[str] | None = None,
        k: int = 10,
        radius: float = 1.0,
        ground_method: str = "grid",
    ) -> None:
        """
        Initialize Random Forest classifier.
        """
        if not _SKLEARN_AVAILABLE:
            raise ClassificationError(_NOT_INSTALLED_ERROR)

        if n_estimators <= 0:
            raise ClassificationError("n_estimators must be greater than zero.")

        if min_samples_split < 2:
            raise ClassificationError("min_samples_split must be at least 2.")

        if min_samples_leaf < 1:
            raise ClassificationError("min_samples_leaf must be at least 1.")

        if max_depth is not None and max_depth <= 0:
            raise ClassificationError("max_depth must be greater than zero.")

        if n_jobs < -1 or n_jobs == 0:
            raise ClassificationError("n_jobs must be -1 or a positive integer.")

        if isinstance(max_features, int) and max_features <= 0:
            raise ClassificationError("max_features must be greater than zero.")

        if isinstance(max_features, float) and not (0.0 < max_features <= 1.0):
            raise ClassificationError("max_features must be in (0.0, 1.0].")

        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._max_features = max_features
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._random_state = random_state
        self._n_jobs = n_jobs

        assert _SklearnRandomForest is not None  # narrows type for mypy after the availability check above

        model: Any = _SklearnRandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
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
        Return whether the optional scikit-learn dependency is installed.
        """
        return _SKLEARN_AVAILABLE

    @override
    def name(self) -> str:
        """
        Return classifier name.
        """
        return "random_forest"
