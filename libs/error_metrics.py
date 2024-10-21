import math
from typing import Dict
from numpy.ma import MaskedArray
import sklearn.utils.fixes
import numpy as np

sklearn.utils.fixes.MaskedArray = MaskedArray
import pandas as pd


def error_metrics_evaluation(predictions: pd.core.series.Series,
                             real_value: pd.core.series.Series,
                             id_forecast: str) -> Dict:
    absolute_differences = (predictions - real_value).abs()
    quadratic_differences = (predictions - real_value) ** 2

    real_avg = real_value.mean()
    real_max = real_value.max()

    # Mean error
    bias_value = absolute_differences.mean()
    # MAE - Mean absolute error
    mae_value = absolute_differences.mean()
    # nMAE - Mean absolute error
    nmae_value = mae_value*100/real_max
    # MSE - Mean squared error
    mse_value = quadratic_differences.mean()
    # RMSE - Root mean square error
    rmse_value = math.sqrt(mse_value)
    # NRMSE
    nrmse_value = rmse_value * 100 / real_avg
    # NRMSE_MAX
    nrmse_value_max = rmse_value * 100 / real_max
    # MAPE
    mape_value = mae_value * 100 / real_avg
    # NMAPE
    nmape_value = mae_value * 100 / real_max

    # Dictionary with all error metrics
    metrics_dict = {'ID': id_forecast,
                    'MAPE': round(mape_value,4),
                    'NMAPE': round(nmape_value,4),
                    'NRMSE': round(nrmse_value,4),
                    'NRMSE MAX': round(nrmse_value_max,4),
                    'RMSE': round(rmse_value,4),
                    'MSE': round(mse_value,4),
                    'MAE': round(mae_value,4),
                    'NMAE': round(nmae_value,4),
                    'BIAS': round(bias_value,4)}

    return metrics_dict


#
def market_performace_evaluation(predictions: pd.core.series.Series,
                                 real_value: pd.core.series.Series,
                                 c_dam: pd.core.series.Series,
                                 c_up: pd.core.series.Series,
                                 c_down: pd.core.series.Series,
                                 id_forecast: str) -> Dict:

    factor1 = predictions*c_dam
    factor2 = np.where(predictions >= real_value, (predictions - real_value)*c_down , (real_value-predictions)*c_up)

    metrics_market_performance = (factor1+factor2).sum()

    metrics_market_performance_dict = {'ID': id_forecast,
                                       'MMF': round(metrics_market_performance, 4)
                                       }
    return metrics_market_performance_dict
