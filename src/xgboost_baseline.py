"""
XGBoost model utilities for the BRFSS diabetes prediction project.

This module contains functions for building and training baseline XGBoost
classification models for both binary and multiclass diabetes prediction.

XGBoost is included as a high-performance gradient boosting model because it
can capture nonlinear relationships and feature interactions in structured
health indicator data.
"""

import pandas as pd
from xgboost import XGBClassifier


def build_xgboost_model(
    task_type: str = "binary",
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1
) -> XGBClassifier:
    """
    Build a baseline XGBoost classification model.

    Parameters
    ----------
    task_type : str, optional
        Classification type. Use:
        - "binary" for binary diabetes prediction
        - "multiclass" for three-class diabetes prediction.
        Default is "binary".
    n_estimators : int, optional
        Number of boosting rounds/trees. Default is 200.
    max_depth : int, optional
        Maximum tree depth. Default is 4.
    learning_rate : float, optional
        Step size shrinkage used in boosting. Default is 0.1.
    subsample : float, optional
        Fraction of training samples used for each tree. Default is 0.8.
    colsample_bytree : float, optional
        Fraction of features used for each tree. Default is 0.8.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    n_jobs : int, optional
        Number of CPU cores used. Default is -1.

    Returns
    -------
    model : XGBClassifier
        Configured XGBoost classifier.

    Raises
    ------
    ValueError
        If task_type is not "binary" or "multiclass".
    """
    if task_type not in ["binary", "multiclass"]:
        raise ValueError("task_type must be either 'binary' or 'multiclass'.")

    if task_type == "binary":
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs
        )

    else:
        model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=3,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs
        )

    return model


def train_xgboost_model(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> XGBClassifier:
    """
    Train an XGBoost classification model.

    Parameters
    ----------
    model : XGBClassifier
        XGBoost classifier to train.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.

    Returns
    -------
    model : XGBClassifier
        Trained XGBoost classifier.
    """
    model.fit(X_train, y_train)
    return model