"""
Reduced-feature model utilities for the BRFSS diabetes prediction project.

This module trains selected machine learning models using a reduced set of
features. The goal is to evaluate whether compact diabetes risk prediction
models can achieve performance comparable to full-feature models.

The reduced feature set is expected to come from feature selection methods
such as SHAP, LASSO, RFE, or Mutual Information.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_reduced_balanced_logistic_regression_model(
    task_type: str = "binary",
    random_state: int = 42,
    max_iter: int = 1000
) -> Pipeline:
    """
    Build a balanced Logistic Regression model for reduced-feature data.

    Logistic Regression requires feature scaling, so StandardScaler is included
    in the pipeline. Class weighting is used to address class imbalance.

    Parameters
    ----------
    task_type : str, optional
        Classification task type. Use:
        - "binary" for binary diabetes prediction
        - "multiclass" for three-class diabetes prediction.
        Default is "binary".
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    max_iter : int, optional
        Maximum number of iterations for model convergence. Default is 1000.

    Returns
    -------
    model : Pipeline
        Scikit-learn pipeline containing StandardScaler and LogisticRegression.

    Raises
    ------
    ValueError
        If task_type is not "binary" or "multiclass".
    """
    if task_type not in ["binary", "multiclass"]:
        raise ValueError("task_type must be either 'binary' or 'multiclass'.")

    if task_type == "binary":
        classifier = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_state
        )
    else:
        classifier = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
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


def build_reduced_balanced_xgboost_model(
    task_type: str = "binary",
    scale_pos_weight: Optional[float] = None,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1
) -> XGBClassifier:
    """
    Build a balanced XGBoost model for reduced-feature data.

    For binary classification, scale_pos_weight is used to handle imbalance.
    For multiclass classification, sample weights should be passed during
    model fitting instead.

    Parameters
    ----------
    task_type : str, optional
        Classification task type. Use:
        - "binary" for binary diabetes prediction
        - "multiclass" for three-class diabetes prediction.
        Default is "binary".
    scale_pos_weight : float or None, optional
        Ratio of negative to positive samples for binary imbalance handling.
        Required for binary classification if imbalance handling is desired.
    n_estimators : int, optional
        Number of boosting rounds/trees. Default is 200.
    max_depth : int, optional
        Maximum depth of each tree. Default is 4.
    learning_rate : float, optional
        Boosting learning rate. Default is 0.1.
    subsample : float, optional
        Fraction of samples used per boosting round. Default is 0.8.
    colsample_bytree : float, optional
        Fraction of features used per tree. Default is 0.8.
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
            scale_pos_weight=scale_pos_weight,
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


def train_reduced_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: Optional[np.ndarray] = None
):
    """
    Train a reduced-feature model.

    Parameters
    ----------
    model : object
        Scikit-learn compatible model or pipeline.
    X_train : pd.DataFrame
        Reduced training feature matrix.
    y_train : pd.Series
        Training target labels.
    sample_weight : np.ndarray or None, optional
        Optional sample weights for imbalance handling. Mainly used for
        multiclass XGBoost. Default is None.

    Returns
    -------
    model : object
        Trained model.
    """
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    return model