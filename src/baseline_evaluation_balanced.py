"""
Baseline model training utilities for the BRFSS diabetes prediction project.

This module contains Logistic Regression baseline models for binary and
multiclass diabetes prediction. It supports both ordinary baseline training
and class-weighted training for imbalanced datasets.
"""

from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_regression_model(
    task_type: str = "binary",
    class_weight: Optional[str] = None,
    random_state: int = 42
) -> Pipeline:
    """
    Build a Logistic Regression pipeline with feature scaling.

    Parameters
    ----------
    task_type : str, optional
        Classification type. Use "binary" for binary classification or
        "multiclass" for three-class classification.
    class_weight : str or None, optional
        Class weighting strategy. Use "balanced" to handle class imbalance.
        Use None for ordinary baseline Logistic Regression.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline containing StandardScaler and LogisticRegression.
    """
    if task_type not in ["binary", "multiclass"]:
        raise ValueError("task_type must be either 'binary' or 'multiclass'.")

    if task_type == "binary":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state
        )
    else:
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state,
            multi_class="multinomial",
            solver="lbfgs"
        )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier)
        ]
    )

    return model


def train_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Train a scikit-learn model pipeline.

    Parameters
    ----------
    model : Pipeline
        Model pipeline to train.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.

    Returns
    -------
    Pipeline
        Trained model pipeline.
    """
    model.fit(X_train, y_train)
    return model