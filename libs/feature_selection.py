from datetime import datetime
from typing import Tuple, Any

import numpy as np
from numpy import ndarray, dtype
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore")
# Suppress the ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

"""
====================================================
Lasso
====================================================
"""


#############
# source: https://towardsdatascience.com/feature-selection-in-machine-learning-using-lasso-regression-7809c7c2771a
# use neg_mean_squared_error because the grid search tries to maximize the performance metrics,
# so we add a minus sign to minimize the mean squared error.


def lasso_regressor_run(df_historic: pd.DataFrame, target: str) -> tuple[
    ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
    """
    This function applies the Lasso regression to select the best features for the target variable.

    Copy the input DataFrame to avoid modifying the original data.
    Separate the predictors and target variable.
    Convert the predictors and target variable to numpy arrays.
    Create a pipeline with StandardScaler and LassoCV.
    Fit the pipeline to the data.
    Retrieve the best alpha value from the fitted model.
    Extract the coefficients from the model.
    Determine the good and bad variables based on the coefficients.
    Return the good variables, coefficients, and bad variables.

    :param df_historic: DataFrame with the historic data
    :param target: Variable to predict
    :return: Variables with non-zero coefficients and variables with zero coefficients
    """

    df_train = df_historic.copy()

    # Predictors and target separation
    predictors = df_train.drop([target], axis=1)
    variables = predictors.columns
    predictors = predictors.to_numpy()

    reals = df_train[target].to_numpy()

    # Pipeline with StandardScaler and LassoCV
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Ensure features are scaled
        ('model', LassoCV(alphas=[1e-2, 2e-2, 1e-1, 2e-1, 1, 10, 20], max_iter=20000, tol=1e-4))  # Using LassoCV
    ])

    # Fit the LassoCV model
    lasso_cv = pipeline.fit(predictors, reals)

    # Get the best alpha from LassoCV
    best_alpha = lasso_cv.named_steps['model'].alpha_
    # print(f"Best alpha selected: {best_alpha}")

    # Get the coefficients (features' importance)
    coefficients = lasso_cv.named_steps['model'].coef_

    # Calculate importance and get the good and bad variables
    importance = np.abs(coefficients)
    good_variables = np.array(variables)[importance > 0]  # Variables with non-zero coefficients
    bad_variables = np.array(variables)[importance == 0]  # Variables with zero coefficients (discarded)

    # Return the good variables and bad variables
    return good_variables, bad_variables
