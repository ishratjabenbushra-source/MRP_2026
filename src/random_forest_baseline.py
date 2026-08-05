"""
Tree-based model utilities for the BRFSS diabetes prediction project.

This module contains Random Forest model-building and training functions for
binary and multiclass diabetes classification tasks.

Random Forest is used as a nonlinear baseline model because it can capture
complex relationships among health, demographic, behavioral, and lifestyle
features without requiring feature scaling.
"""

from typing import Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def build_random_forest_model(
    class_weight: Optional[str] = None,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    random_state: int = 42,
    n_jobs: int = -1
) -> RandomForestClassifier:
    """
    Build a Random Forest classification model.

    Parameters
    ----------
    class_weight : str or None, optional
        Class weighting strategy. Use None for baseline Random Forest.
        Use "balanced" to give higher weight to minority classes.
        Default is None.
    n_estimators : int, optional
        Number of trees in the forest. Default is 200.
    max_depth : int or None, optional
        Maximum depth of each decision tree. If None, trees are expanded
        until all leaves are pure or contain fewer than min_samples_split
        samples. Default is None.
    min_samples_split : int, optional
        Minimum number of samples required to split an internal node.
        Default is 2.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    n_jobs : int, optional
        Number of CPU cores to use. Use -1 to use all available cores.
        Default is -1.

    Returns
    -------
    model : RandomForestClassifier
        Configured Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )

    return model


def train_random_forest_model(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestClassifier:
    """
    Train a Random Forest classification model.

    Parameters
    ----------
    model : RandomForestClassifier
        Random Forest classifier to train.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.

    Returns
    -------
    model : RandomForestClassifier
        Trained Random Forest classifier.
    """
    model.fit(X_train, y_train)
    return model