"""
SHAP and permutation importance utilities for the BRFSS diabetes prediction project.

This module supports model explainability for selected diabetes prediction models.
It includes SHAP analysis for tree-based models such as XGBoost and permutation
importance for any scikit-learn compatible model.
"""

from typing import Optional

import pandas as pd
import shap
from sklearn.inspection import permutation_importance


def sample_explanation_data(
    X: pd.DataFrame,
    sample_size: int = 2000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample a smaller subset of data for explainability analysis.

    SHAP can be computationally expensive on large datasets. This function
    samples a manageable subset while preserving the original feature columns.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    sample_size : int, optional
        Number of rows to sample. Default is 2000.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    X_sample : pd.DataFrame
        Sampled feature matrix.
    """
    if len(X) <= sample_size:
        return X.copy()

    return X.sample(
        n=sample_size,
        random_state=random_state
    )


def compute_tree_shap_values(
    model,
    X_sample: pd.DataFrame
):
    """
    Compute SHAP values for a tree-based model.

    This function is intended for models such as XGBoost, Random Forest,
    and other tree ensemble models.

    Parameters
    ----------
    model : object
        Trained tree-based model.
    X_sample : pd.DataFrame
        Feature matrix used for SHAP explanation.

    Returns
    -------
    explainer : shap.Explainer
        SHAP explainer object.
    shap_values : shap.Explanation
        Computed SHAP values for the input data.
    """
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    return explainer, shap_values


def get_mean_absolute_shap_importance(
    shap_values,
    feature_names: list
) -> pd.DataFrame:
    """
    Create a feature importance table using mean absolute SHAP values.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values returned by the SHAP explainer.
    feature_names : list
        List of feature names.

    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame containing features and mean absolute SHAP importance values.
    """
    mean_abs_values = abs(shap_values.values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_values
    }).sort_values(
        by="mean_abs_shap",
        ascending=False
    )

    return importance_df


def compute_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: str,
    n_repeats: int = 5,
    random_state: int = 42,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Compute permutation importance for a trained model.

    Permutation importance measures how much model performance decreases when
    a feature is randomly shuffled. Larger decreases indicate more important
    features.

    Parameters
    ----------
    model : object
        Trained scikit-learn compatible model or pipeline.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True test labels.
    scoring : str
        Scoring metric used to measure performance decrease.
        Example: "f1" for binary classification or "f1_macro" for multiclass.
    n_repeats : int, optional
        Number of times to permute each feature. Default is 5.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    n_jobs : int, optional
        Number of CPU cores to use. Default is -1.

    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame containing permutation importance mean and standard deviation.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values(
        by="importance_mean",
        ascending=False
    )

    return importance_df