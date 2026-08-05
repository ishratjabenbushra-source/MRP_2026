"""
Hyperparameter tuning utilities for the BRFSS diabetes prediction project.

This module performs cross-validation-based hyperparameter tuning for the
strongest candidate models identified during earlier experiments.

"""

from typing import Dict, Any

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


def tune_model_randomized_search(
    model,
    param_distributions: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str,
    n_iter: int = 15,
    cv_splits: int = 3,
    random_state: int = 42,
    n_jobs: int = -1
) -> RandomizedSearchCV:
    """
    Tune a model using RandomizedSearchCV with stratified cross-validation.

    Parameters
    ----------
    model : object
        Scikit-learn compatible model or pipeline to tune.
    param_distributions : dict
        Dictionary of hyperparameter search spaces.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.
    scoring : str
        Scoring metric used for model selection.
        Use "f1" for binary classification and "f1_macro" for multiclass.
    n_iter : int, optional
        Number of random parameter combinations to test. Default is 15.
    cv_splits : int, optional
        Number of cross-validation folds. Default is 3.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    n_jobs : int, optional
        Number of CPU cores to use. Default is -1.

    Returns
    -------
    search : RandomizedSearchCV
        Fitted RandomizedSearchCV object containing the best estimator,
        best parameters, and best cross-validation score.
    """
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1
    )

    search.fit(X_train, y_train)

    return search


def get_best_model_from_search(search: RandomizedSearchCV):
    """
    Extract the best estimator from a fitted RandomizedSearchCV object.

    Parameters
    ----------
    search : RandomizedSearchCV
        Fitted hyperparameter search object.

    Returns
    -------
    best_model : object
        Best model found during hyperparameter tuning.
    """
    return search.best_estimator_


def summarize_tuning_results(search: RandomizedSearchCV) -> pd.DataFrame:
    """
    Summarize the best hyperparameter tuning result.

    Parameters
    ----------
    search : RandomizedSearchCV
        Fitted hyperparameter search object.

    Returns
    -------
    summary_df : pd.DataFrame
        One-row DataFrame containing best score and best parameters.
    """
    summary_df = pd.DataFrame(
        {
            "best_cv_score": [search.best_score_],
            "best_params": [search.best_params_]
        }
    )

    return summary_df