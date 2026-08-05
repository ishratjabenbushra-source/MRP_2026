"""
SMOTE preprocessing utilities for the BRFSS diabetes prediction project.

This module applies Synthetic Minority Oversampling Technique (SMOTE) only to
training data. SMOTE should never be applied to the test set because the test
set must represent the original real-world class distribution.
"""

from typing import Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to the training data.

    SMOTE creates synthetic minority-class examples to reduce class imbalance.
    This function should only be applied to the training set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    X_train_smote : pd.DataFrame
        SMOTE-resampled training feature matrix.
    y_train_smote : pd.Series
        SMOTE-resampled training labels.
    """
    smote = SMOTE(random_state=random_state)

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    X_train_smote = pd.DataFrame(
        X_resampled,
        columns=X_train.columns
    )

    y_train_smote = pd.Series(
        y_resampled,
        name=y_train.name
    )

    return X_train_smote, y_train_smote