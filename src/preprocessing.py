"""
Preprocessing utilities for the BRFSS diabetes prediction project.

This module prepares the binary and multiclass BRFSS diabetes datasets for
machine learning by separating features and target variables and creating
stratified train-test splits.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_features_target(
    df: pd.DataFrame,
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate a dataset into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing both predictors and the target variable.
    target_col : str
        Name of the target column to separate from the predictors.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix containing all predictor variables.
    y : pd.Series
        Target vector containing class labels.

    Raises
    ------
    ValueError
        If the target column is not found in the dataset.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create a stratified train-test split.

    Stratification preserves the original class distribution in both the
    training and testing sets, which is important for imbalanced datasets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float, optional
        Proportion of the dataset to include in the test split.
        Default is 0.20.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Testing feature matrix.
    y_train : pd.Series
        Training target vector.
    y_test : pd.Series
        Testing target vector.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def prepare_modeling_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare a BRFSS dataset for machine learning.

    This function combines feature-target separation and stratified train-test
    splitting into one reusable preprocessing step.

    Parameters
    ----------
    df : pd.DataFrame
        Input BRFSS diabetes dataset.
    target_col : str
        Name of the target variable. Use:
        - 'Diabetes_binary' for the binary dataset
        - 'Diabetes_012' for the multiclass dataset
    test_size : float, optional
        Proportion of data used for testing. Default is 0.20.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Testing feature matrix.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Testing labels.
    """
    X, y = split_features_target(df, target_col)

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def print_class_distribution(y: pd.Series, dataset_name: str = "Dataset") -> None:
    """
    Print class counts and class percentages for a target vector.

    This is useful after train-test splitting to verify that stratification
    preserved the original class distribution.

    Parameters
    ----------
    y : pd.Series
        Target vector containing class labels.
    dataset_name : str, optional
        Name shown in the printed output. Default is 'Dataset'.

    Returns
    -------
    None
    """
    counts = y.value_counts().sort_index()
    percentages = y.value_counts(normalize=True).sort_index() * 100

    distribution = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(2)
    })

    print(f"\nClass distribution for {dataset_name}:")
    print(distribution)