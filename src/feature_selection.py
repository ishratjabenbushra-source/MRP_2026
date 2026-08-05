"""
Feature selection utilities for the BRFSS diabetes prediction project.

This module contains feature selection methods used to identify a reduced set
of BRFSS health indicators for compact diabetes risk prediction.

The implemented methods are:
1. SHAP-based ranking
2. LASSO-based feature selection
3. Recursive Feature Elimination (RFE)
4. Mutual Information

These methods support the research question focused on whether a compact
screening tool can maintain predictive performance using fewer features.
"""

from typing import List, Optional

import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_top_shap_features(
    shap_importance_df: pd.DataFrame,
    feature_col: str = "feature",
    importance_col: str = "mean_abs_shap",
    top_n: int = 10
) -> List[str]:
    """
    Select top features from a SHAP importance DataFrame.

    Parameters
    ----------
    shap_importance_df : pd.DataFrame
        DataFrame containing feature names and SHAP importance values.
    feature_col : str, optional
        Name of the column containing feature names. Default is "feature".
    importance_col : str, optional
        Name of the column containing SHAP importance values.
        Default is "mean_abs_shap".
    top_n : int, optional
        Number of top features to select. Default is 10.

    Returns
    -------
    top_features : list of str
        List of top feature names ranked by SHAP importance.

    Raises
    ------
    ValueError
        If required columns are not found in the input DataFrame.
    """
    required_cols = {feature_col, importance_col}

    if not required_cols.issubset(shap_importance_df.columns):
        raise ValueError(
            f"Input DataFrame must contain columns: {required_cols}"
        )

    top_features = (
        shap_importance_df
        .sort_values(by=importance_col, ascending=False)
        .head(top_n)[feature_col]
        .tolist()
    )

    return top_features


def lasso_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 10,
    random_state: int = 42,
    max_iter: int = 2000
) -> pd.DataFrame:
    """
    Perform LASSO-style feature selection using L1-regularized Logistic Regression.

    L1 regularization can shrink less useful feature coefficients toward zero.
    Features are ranked using the absolute value of their learned coefficients.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.
    top_n : int, optional
        Number of top-ranked features to return. Default is 10.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    max_iter : int, optional
        Maximum number of optimization iterations. Default is 2000.

    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame containing feature names and absolute LASSO coefficients.
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    max_iter=max_iter,
                    random_state=random_state,
                    class_weight="balanced"
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    classifier = model.named_steps["classifier"]

    coefficients = classifier.coef_

    if coefficients.ndim == 2:
        coefficient_importance = abs(coefficients).mean(axis=0)
    else:
        coefficient_importance = abs(coefficients)

    importance_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "lasso_importance": coefficient_importance
        }
    ).sort_values(
        by="lasso_importance",
        ascending=False
    )

    return importance_df.head(top_n)


def rfe_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 10,
    random_state: int = 42,
    max_iter: int = 1000
) -> pd.DataFrame:
    """
    Perform Recursive Feature Elimination using Logistic Regression.

    RFE repeatedly removes the least important features based on model
    coefficients until the requested number of features remains.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.
    top_n : int, optional
        Number of features to select. Default is 10.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    max_iter : int, optional
        Maximum number of iterations for Logistic Regression. Default is 1000.

    Returns
    -------
    selected_features_df : pd.DataFrame
        DataFrame containing selected feature names and RFE rankings.
    """
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=random_state,
                    class_weight="balanced"
                )
            )
        ]
    )

    # RFE needs direct access to coef_, so we use the classifier separately
    # after scaling the input manually.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    classifier = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        class_weight="balanced"
    )

    rfe = RFE(
        estimator=classifier,
        n_features_to_select=top_n
    )

    rfe.fit(X_scaled, y_train)

    selected_features_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "selected": rfe.support_,
            "rfe_rank": rfe.ranking_
        }
    ).sort_values(
        by="rfe_rank",
        ascending=True
    )

    return selected_features_df.head(top_n)


def mutual_information_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Perform feature selection using Mutual Information.

    Mutual Information measures the dependency between each feature and the
    target variable. Higher scores indicate stronger relationships.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.
    top_n : int, optional
        Number of top-ranked features to return. Default is 10.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    mi_df : pd.DataFrame
        DataFrame containing feature names and mutual information scores.
    """
    mi_scores = mutual_info_classif(
        X_train,
        y_train,
        random_state=random_state
    )

    mi_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "mutual_information": mi_scores
        }
    ).sort_values(
        by="mutual_information",
        ascending=False
    )

    return mi_df.head(top_n)


def create_reduced_feature_dataset(
    X: pd.DataFrame,
    selected_features: List[str]
) -> pd.DataFrame:
    """
    Create a reduced feature dataset using selected features.

    Parameters
    ----------
    X : pd.DataFrame
        Original feature matrix.
    selected_features : list of str
        Feature names to keep.

    Returns
    -------
    X_reduced : pd.DataFrame
        Reduced feature matrix containing only selected features.

    Raises
    ------
    ValueError
        If any selected feature is missing from X.
    """
    missing_features = [
        feature for feature in selected_features if feature not in X.columns
    ]

    if missing_features:
        raise ValueError(f"Missing selected features in X: {missing_features}")

    X_reduced = X[selected_features].copy()

    return X_reduced