import pandas as pd
from libs.error_metrics import error_metrics_evaluation
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# or suppress only specific LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning)


"""
=======================================================================================================================
LIGHT GBM
=======================================================================================================================
https://machinelearningmastery.com/light-gradient-boosted-machine-lightgbm-ensemble/
https://www.kaggle.com/bjoernholzhauer/lightgbm-tuning-with-optuna
"""


def light_gbm_regressor(x_train: pd.DataFrame, y_train: pd.DataFrame, grid_search: bool) -> LGBMRegressor:
    """
    This function trains a LightGBM regressor with the best hyperparameters found using GridSearchCV or
    with the hyperparameters defined in the function.
    ....
    :param x_train: dataframe with the training data
    :param y_train: dataframe with the target variable
    :param grid_search: if true, the function gets the hyperparameters using GridSearchCV
    :return: model with the trained regressor obtained from the LightGBM with the best hyperparameters or with the hyperparameters defined in the function
    """

    # -------------------------- #
    #       TRAIN LIGHTGBM       #
    # -------------------------- #

    if grid_search:
        # Define the hyperparameter grid
        param_grid = {
            'num_leaves': [15, 30, 45],
            'learning_rate': [0.15, 0.2],
            'n_estimators': [1000, 1500],
            'min_child_samples': [50, 70],
            'max_depth': [9, 10]
        }
        # initialize the regressor
        model = LGBMRegressor()
        # Define the GridSearchCV object
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2
        )
        # Fit the GridSearchCV object to the data
        grid_search.fit(x_train, y_train)
        # Print the best hyperparameters
        print("Best hyperparameters:", grid_search.best_params_)
        # fit the lgbm with the best hyperparameters
        model = LGBMRegressor(
            boosting_type='dart',
            num_leaves=grid_search.best_params_['num_leaves'],
            learning_rate=grid_search.best_params_['learning_rate'],
            n_estimators=grid_search.best_params_['n_estimators'],
            min_child_samples=grid_search.best_params_['min_child_samples'],
            max_depth=grid_search.best_params_['max_depth'],
            verbose=-1
        )
    if not grid_search:
        # use the best found hyperparameters in training
        model = LGBMRegressor(
            # boosting_type='dart',
            learning_rate=0.05,  # FF:0.15
            max_depth=9,
            min_data_in_leaf=20,
            n_estimators=150,
            num_leaves=30,
            force_row_wise=True,
            verbose=-1
        )

    # fit the model on the whole train dataset
    model.fit(x_train, y_train)

    return model
