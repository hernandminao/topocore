"""
topocore.processing.classification.gradient_boost
==================================================

Gradient Boost classifier for point cloud classification.

Uses scikit-learn's ``HistGradientBoostingClassifier``, a histogram-
based gradient boosting implementation that is fast and memory-
efficient for large point clouds.

scikit-learn is an optional dependency (extra ``ml``), handled with
the same lazy-import strategy as ``random_forest.py``.

References
----------
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient
  Boosting Decision Tree. NeurIPS. (HistGradientBoosting follows the
  same histogram-based split-finding strategy.)

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
    from sklearn.ensemble import (  # type: ignore
        HistGradientBoostingClassifier as _SklearnHistGB,
    )

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without scikit-learn installed
    _SklearnHistGB = None
    _SKLEARN_AVAILABLE = False

_NOT_INSTALLED_ERROR = (
    "scikit-learn is not installed. Install it with `pip install topocore[ml]` "
    "(or `pip install scikit-learn`) to use GradientBoostClassifier."
)


class GradientBoostClassifier(MachineLearningClassifier):
    """
    Gradient Boost classifier for point cloud classification.

    Parameters
    ----------
    max_iter
        Number of boosting iterations (trees).
    max_depth
        Maximum depth of each tree. ``None`` means no limit.
    learning_rate
        Shrinkage applied to each tree's contribution.
    max_leaf_nodes
        Maximum number of leaves per tree.
    l2_regularization
        L2 regularization strength.
    random_state
        Random seed for reproducibility.
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
        constructor parameter is outside its valid range (PR21
        remediation, ML-VALIDATION-001 -- see RandomForestClassifier's
        own docstring for the full rationale).
    """

    __slots__ = (
        "_l2_regularization",
        "_learning_rate",
        "_max_depth",
        "_max_iter",
        "_max_leaf_nodes",
        "_random_state",
    )

    def __init__(
        self,
        *,
        max_iter: int = 100,
        max_depth: int | None = None,
        learning_rate: float = 0.1,
        max_leaf_nodes: int | None = 31,
        l2_regularization: float = 0.0,
        random_state: int | None = 42,
        feature_names: list[str] | None = None,
        k: int = 10,
        radius: float = 1.0,
        ground_method: str = "grid",
    ) -> None:
        """
        Initialize Gradient Boost classifier.
        """
        if not _SKLEARN_AVAILABLE:
            raise ClassificationError(_NOT_INSTALLED_ERROR)

        if max_iter <= 0:
            raise ClassificationError("max_iter must be greater than zero.")

        if learning_rate <= 0:
            raise ClassificationError("learning_rate must be greater than zero.")

        if l2_regularization < 0:
            raise ClassificationError("l2_regularization must be non-negative.")

        if max_leaf_nodes is not None and max_leaf_nodes < 2:
            raise ClassificationError("max_leaf_nodes must be at least 2 when specified.")

        if max_depth is not None and max_depth <= 0:
            raise ClassificationError("max_depth must be greater than zero.")

        self._max_iter = max_iter
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._max_leaf_nodes = max_leaf_nodes
        self._l2_regularization = l2_regularization
        self._random_state = random_state

        assert _SklearnHistGB is not None  # narrows type for mypy after the availability check above

        model: Any = _SklearnHistGB(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            random_state=random_state,
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
        return "gradient_boost"
