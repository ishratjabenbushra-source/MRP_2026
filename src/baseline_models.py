"""
Model training utilities for the BRFSS diabetes prediction project.

This module currently includes baseline Logistic Regression models for
binary and multiclass diabetes classification.
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

    Logistic Regression is sensitive to feature scale, so StandardScaler is
    included before model training.

    Parameters
    ----------
    task_type : str, optional
        Type of classification task. Use:
        - "binary" for binary diabetes prediction
        - "multiclass" for three-class diabetes prediction
        Default is "binary".
    class_weight : str or None, optional
        Class weighting strategy. Use "balanced" to address class imbalance,
        or None for baseline training. Default is None.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        A scikit-learn pipeline containing StandardScaler and LogisticRegression.

    Raises
    ------
    ValueError
        If task_type is not "binary" or "multiclass".
    """
    if task_type not in ["binary", "multiclass"]:
        raise ValueError("task_type must be either 'binary' or 'multiclass'.")

    if task_type == "binary":
        logistic_regression = LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state
        )

    else:
        logistic_regression = LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state,
            multi_class="multinomial",
            solver="lbfgs"
        )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", logistic_regression)
        ]
    )

    return model


def train_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Train a machine learning model.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Model pipeline to train.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Trained model pipeline.
    """
    model.fit(X_train, y_train)
    return model