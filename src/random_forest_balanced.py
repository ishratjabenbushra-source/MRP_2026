"""
Balanced tree-based model utilities for the BRFSS diabetes prediction project.

This module contains Random Forest models with class weighting enabled.
Class weighting is used to reduce the effect of class imbalance by giving
higher importance to minority classes during training.
"""

from typing import Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def build_balanced_random_forest_model(
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    random_state: int = 42,
    n_jobs: int = -1
) -> RandomForestClassifier:
    """
    Build a class-weighted Random Forest classification model.

    This model uses class_weight="balanced", which automatically adjusts
    class weights inversely proportional to class frequencies in the training
    data. This is useful for imbalanced datasets such as the BRFSS diabetes
    datasets.

    Parameters
    ----------
    n_estimators : int, optional
        Number of decision trees in the forest. Default is 200.
    max_depth : int or None, optional
        Maximum depth of each tree. If None, trees are expanded fully.
        Default is None.
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
        Configured class-weighted Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs
    )

    return model


def train_balanced_random_forest_model(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestClassifier:
    """
    Train a class-weighted Random Forest classification model.

    Parameters
    ----------
    model : RandomForestClassifier
        Class-weighted Random Forest model to train.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.

    Returns
    -------
    model : RandomForestClassifier
        Trained class-weighted Random Forest classifier.
    """
    model.fit(X_train, y_train)
    return model