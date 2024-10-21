import os
import pickle
from datetime import datetime

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def dimension_reduction_pca_train(features: list,
                                  explained_variance: float,
                                  meteo_df: pd.DataFrame,
                                  save: bool,
                                  time_variable: str,
                                  meteo_id_variable: str,
                                  folder: str,
                                  asset_id_name: str) -> pd.DataFrame:
    """
    :param meteo_df: dataframe containing meteorology data
    :param features: meteorology indicators to be considered
    :param explained_variance: number of variables must explain >= explained_variance
    :param folder: folder where the models will be saved
    :param meteo_id_variable:
    :param time_variable: variable that represents time for meteo data
    :param save: if true save csv files
    :param asset_id_name: asset id name to which the PCA features belong
    :return: models dataframe containing components for all selected meteorological indicators
    """

    pca_features = pd.DataFrame()
    print("** Performing PCA **")
    for i in range(len(features)):
        print(f"Feature {i+1}/{len(features)}: {features[i]}")

        feature_forecast = meteo_df[[meteo_id_variable, time_variable, features[i]]]

        ###
        if len(feature_forecast) == 0:
            continue

        # Change dataframe structure in order to have a meteo location in each column
        feature_forecast_pivoted = feature_forecast.pivot(index=time_variable,
                                                          columns=meteo_id_variable,
                                                          values=features[i])

        # Update columns name considering the meteo variable analyzed
        feature_forecast_pivoted = feature_forecast_pivoted.add_prefix(features[i] + '_')

        # Drop columns which contain all NaN values
        feature_forecast_pivoted = feature_forecast_pivoted.dropna()

        # Dimensionality reduction
        # PCA
        # 1. Standardizing the features
        scaler = StandardScaler()
        feature_forecast_pivoted_standardized = scaler.fit_transform(feature_forecast_pivoted)

        # 2. PCA using explained variance
        pca = PCA(explained_variance)
        principal_components = pca.fit_transform(feature_forecast_pivoted_standardized)
        principal_components_df = pd.DataFrame(principal_components, index=feature_forecast_pivoted.index)
        principal_components_df = principal_components_df.add_prefix(features[i] + '_')

        pca_features = pd.concat([pca_features, principal_components_df], axis=1)

        # save models
        pickle.dump(pca, open(
            folder + os.sep + 'pca_model' + os.sep + 'pca_asset_id_' + str(asset_id_name) + '_' + features[
                i] + '.pkl',
            'wb'))
        # save scaler
        pickle.dump(scaler, open(
            folder + os.sep + 'pca_scaler' + os.sep + 'scaler_asset_id_' + str(asset_id_name) + '_' + features[
                i] + '.pkl',
            'wb'))

    if save:
        # save pca_features
        pca_features.to_csv(
            folder + os.sep + 'asset_id_' + str(asset_id_name) + '_' + 'pca_features.csv')

    return pca_features


def import_pca_from_file(folder: str,
                         asset_id_name: str,
                         time_variable: str) -> pd.DataFrame:
    """
    Import PCA features from file previously saved
    :param folder: folder where the file is saved
    :param asset_id_name: asset id name to which the PCA features belong
    :param time_variable: variable that represents time for meteo data
    :return:
    """
    pca_features = pd.read_csv(
        folder + os.sep + 'asset_id_' + str(asset_id_name) + '_' + 'pca_features.csv')

    pca_features[time_variable] = pd.to_datetime(pca_features[time_variable])
    pca_features = pca_features.set_index(time_variable)

    return pca_features


def dimension_reduction_pca_run(features: list,
                                meteo_df: pd.DataFrame,
                                time_variable: str,
                                meteo_id_variable: str,
                                folder: str,
                                asset_id_name: str) -> pd.DataFrame:

    pca_features = pd.DataFrame()
    print("** Performing PCA **")
    for i in range(len(features)):

        print(f"Feature {i + 1}/{len(features)}: {features[i]}")

        feature_forecast = meteo_df[[meteo_id_variable, time_variable, features[i]]]

        if len(feature_forecast) == 0:
            continue

        # Change dataframe structure in order to have a meteo location in each column
        feature_forecast_pivoted = feature_forecast.pivot(index=time_variable,
                                                          columns=meteo_id_variable,
                                                          values=features[i])

        # Update columns name considering the meteo variable analyzed
        feature_forecast_pivoted = feature_forecast_pivoted.add_prefix(features[i] + '_')

        # Drop columns which contain all NaN values
        feature_forecast_pivoted = feature_forecast_pivoted.dropna()

        # Dimensionality reduction (PCA)
        # 1. load the scaler
        filename = folder + os.sep + 'pca_scaler' + os.sep + 'scaler_asset_id_' + str(asset_id_name) + '_' + features[
                    i] + '.pkl'
        try:
            scaler = pickle.load(open(filename, 'rb'))
            feature_forecast_pivoted_standardized = scaler.transform(feature_forecast_pivoted)
        except FileNotFoundError:
            print(f"Missing scaler pickle file in {filename}")
            return pd.DataFrame()

        # 2. load models
        filename = folder + os.sep + 'pca_model' + os.sep + 'pca_asset_id_' + str(asset_id_name) + '_' + features[
                    i] + '.pkl'
        try:
            pca = pickle.load(open(filename, 'rb'))
            principal_components = pca.transform(feature_forecast_pivoted_standardized)
            principal_components_df = pd.DataFrame(principal_components, index=feature_forecast_pivoted.index)
            principal_components_df = principal_components_df.add_prefix(features[i] + '_')
            pca_features = pd.concat([pca_features, principal_components_df], axis=1)
        except FileNotFoundError:
            print(f"missing PCA pickle file in {filename}")
            return pd.DataFrame()

    return pca_features
