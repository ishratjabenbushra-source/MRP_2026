"""
Evaluation utilities for the BRFSS diabetes prediction project.

This module evaluates binary and multiclass classification models using
standard classification metrics, confusion matrices, and ROC-AUC where
appropriate.
"""

from typing import Dict, Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


def evaluate_binary_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate a binary classification model.

    Parameters
    ----------
    model : object
        Trained scikit-learn compatible model or pipeline.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True binary labels.

    Returns
    -------
    results : dict
        Dictionary containing accuracy, precision, recall, F1-score,
        ROC-AUC, confusion matrix, and classification report.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0)
    }

    return results


def evaluate_multiclass_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate a multiclass classification model.

    Macro-averaged precision, recall, and F1-score are used because the
    multiclass BRFSS dataset is highly imbalanced, especially for the
    prediabetes class.

    Parameters
    ----------
    model : object
        Trained scikit-learn compatible model or pipeline.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True multiclass labels.

    Returns
    -------
    results : dict
        Dictionary containing accuracy, macro precision, macro recall,
        macro F1-score, confusion matrix, and classification report.
    """
    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0)
    }

    return results


def summarize_results(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert selected evaluation metrics into a one-row DataFrame.

    Parameters
    ----------
    results : dict
        Dictionary returned by an evaluation function.

    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame containing only scalar evaluation metrics.
    """
    scalar_results = {
        key: value
        for key, value in results.items()
        if not isinstance(value, (list, tuple)) and key not in ["confusion_matrix", "classification_report"]
    }

    return pd.DataFrame([scalar_results])