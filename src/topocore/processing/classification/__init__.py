"""
topocore.processing.classification
====================================

Point cloud classification.

This package assigns ASPRS classification codes to points, either
through hand-crafted geometric rules or through trained machine
learning classifiers sharing a single feature-matrix pipeline.

Public API
----------
- ClassificationResult: result container (labels, confidence, class extraction)
- Classifier / MLClassifier: abstract interfaces
- RuleBasedClassifier: geometric rule-based classifier
- MachineLearningClassifier: shared ML classifier implementation (fit/classify/save/load)
- RandomForestClassifier / GradientBoostClassifier: always-available ML classifiers
- LightGBMClassifier / XGBoostClassifier: optional ML classifiers (``None`` if not installed)
- ClassificationManager: high-level manager with automatic method selection

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .base import ClassificationResult, Classifier, MLClassifier
from .gradient_boost import GradientBoostClassifier
from .lightgbm import LightGBMClassifier
from .manager import ClassificationManager
from .ml import MachineLearningClassifier
from .random_forest import RandomForestClassifier
from .rules import RuleBasedClassifier
from .xgboost import XGBoostClassifier

__all__ = [
    "ClassificationManager",
    "ClassificationResult",
    "Classifier",
    "GradientBoostClassifier",
    "LightGBMClassifier",
    "MLClassifier",
    "MachineLearningClassifier",
    "RandomForestClassifier",
    "RuleBasedClassifier",
    "XGBoostClassifier",
]
