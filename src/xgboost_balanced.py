"""
Balanced XGBoost model utilities for the BRFSS diabetes prediction project.

This module contains functions for building and training imbalance-aware
XGBoost models for binary and multiclass diabetes classification.

For binary classification, scale_pos_weight is used to give more importance
to the minority diabetes/prediabetes class.

For multiclass classification, sample weights are computed from class
frequencies and passed during model training.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def calculate_binary_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calculate scale_pos_weight for binary XGBoost classification.

    scale_pos_weight is calculated as:

        number of negative samples / number of positive samples

    This helps XGBoost pay more attention to the minority positive class.

    Parameters
    ----------
    y_train : pd.Series
        Binary training labels where 0 represents the majority class and
        1 represents the minority class.

    Returns
    -------
    scale_pos_weight : float
        Weight ratio used by XGBoost for binary imbalance handling.

    Raises
    ------
    ValueError
        If y_train does not contain exactly two classes.
    """
    class_counts = y_train.value_counts().sort_index()

    if len(class_counts) != 2:
        raise ValueError("scale_pos_weight is only valid for binary classification.")

    negative_count = class_counts.iloc[0]
    positive_count = class_counts.iloc[1]

    scale_pos_weight = negative_count / positive_count

    return scale_pos_weight


def build_balanced_binary_xgboost_model(
    scale_pos_weight: float,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1
) -> XGBClassifier:
    """
    Build an imbalance-aware binary XGBoost classifier.

    Parameters
    ----------
    scale_pos_weight : float
        Ratio of negative to positive samples in the training data.
    n_estimators : int, optional
        Number of boosting rounds/trees. Default is 200.
    max_depth : int, optional
        Maximum depth of each tree. Default is 4.
    learning_rate : float, optional
        Boosting learning rate. Default is 0.1.
    subsample : float, optional
        Fraction of training samples used per tree. Default is 0.8.
    colsample_bytree : float, optional
        Fraction of features used per tree. Default is 0.8.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    n_jobs : int, optional
        Number of CPU cores to use. Default is -1.

    Returns
    -------
    model : XGBClassifier
        Configured binary XGBoost classifier with scale_pos_weight.
    """
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

    return model


def build_balanced_multiclass_xgboost_model(
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1
) -> XGBClassifier:
    """
    Build an imbalance-aware multiclass XGBoost classifier.

    For multiclass classification, class imbalance is handled by passing
    sample weights during model fitting.

    Parameters
    ----------
    n_estimators : int, optional
        Number of boosting rounds/trees. Default is 200.
    max_depth : int, optional
        Maximum depth of each tree. Default is 4.
    learning_rate : float, optional
        Boosting learning rate. Default is 0.1.
    subsample : float, optional
        Fraction of training samples used per tree. Default is 0.8.
    colsample_bytree : float, optional
        Fraction of features used per tree. Default is 0.8.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    n_jobs : int, optional
        Number of CPU cores to use. Default is -1.

    Returns
    -------
    model : XGBClassifier
        Configured multiclass XGBoost classifier.
    """
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


def compute_multiclass_sample_weights(y_train: pd.Series) -> np.ndarray:
    """
    Compute sample weights for imbalanced multiclass classification.

    Sample weights are calculated using class frequency so that minority
    classes receive higher weight during training.

    Parameters
    ----------
    y_train : pd.Series
        Multiclass training labels.

    Returns
    -------
    sample_weights : np.ndarray
        Array of sample weights aligned with y_train.
    """
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    return sample_weights


def train_xgboost_model(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray | None = None
) -> XGBClassifier:
    """
    Train an XGBoost classification model.

    Parameters
    ----------
    model : XGBClassifier
        XGBoost model to train.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.
    sample_weight : np.ndarray or None, optional
        Optional sample weights for handling class imbalance.
        Default is None.

    Returns
    -------
    model : XGBClassifier
        Trained XGBoost classifier.
    """
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight
    )

    return model