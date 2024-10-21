import datetime
import os
import pickle

import pandas as pd

# Custom libraries
from libs import models_specs as mspec, dimensionality_reduction as dimred, feature_engineering as feng


def main():
    """
    ================================================================
    GET DATA:
        - Meteorology
    ================================================================
    """

    # -------------------------- #
    #  METEO DATA AND SETTINGS   #
    # -------------------------- #
    try:
        meteorology_df = pd.read_csv("prediction_data/meteorology.csv", sep=';')
    except FileNotFoundError:
        print(
            f"\n ** ERROR: No meteorology data available. Please include a meteorology.csv file in prediction folder **")
        return

    if len(meteorology_df) == 0:
        print(f"\n ** ERROR: No meteorology data available. Please verify the content of the meteorology.csv file **")
        return

    # -------------------------- #
    # Define the name of park, asset or country that is being analyzed
    asset_id_name_variable = meteorology_df.columns[0]
    asset_id_name = meteorology_df[meteorology_df.columns[0]].unique()[0]
    # Define the id variables
    meteo_id_variable = meteorology_df.columns[1]
    # Define the time variable for the meteo dataset
    time_variable_meteo = meteorology_df.columns[2]
    # Define all the meteorology features
    features = meteorology_df.columns[3:]
    # Change the time variable to datetime
    meteorology_df[time_variable_meteo] = pd.to_datetime(meteorology_df[time_variable_meteo])
    # meteo points info
    meteo_points = meteorology_df[meteo_id_variable].unique()

    # Infer the frequency of the meteorology_df time series
    unique_times = pd.Series(meteorology_df[time_variable_meteo].unique()).sort_values()
    # Compute the time differences
    time_diffs = unique_times.diff().dropna()
    # Find the minimum time difference
    min_freq = time_diffs.min()
    # Convert to pandas frequency string
    min_freq_str = pd.infer_freq(
        pd.Series([unique_times.iloc[0], unique_times.iloc[0] + min_freq, unique_times.iloc[0] + 2 * min_freq]))

    # Predictions date range
    date_range = pd.DataFrame(pd.date_range(start=meteorology_df[time_variable_meteo].min(),
                                            end=meteorology_df[time_variable_meteo].max(),
                                            freq=min_freq_str),
                              columns=[time_variable_meteo])

    if len(meteo_points) > 1:
        input_pca = dimred.dimension_reduction_pca_run(
            meteo_df=meteorology_df,
            features=features,
            folder=mspec.folder,
            time_variable=time_variable_meteo,
            meteo_id_variable=meteo_id_variable,
            asset_id_name=asset_id_name)

        if len(input_pca) == 0:
            print("\nERROR: PCA features could not be created. Required .pkl files for input variables are missing or unavailable.")
            return
    else:
        input_pca = meteorology_df.copy()

    # ADD PCA features to date range
    input_dataframe = pd.merge(date_range, input_pca, on=time_variable_meteo)
    input_dataframe = input_dataframe.dropna()
    # Create results dataframe for model 1 with the corresponding predictions dates
    prediction_dataframe = input_dataframe[[time_variable_meteo]].copy()
    # Continue to create input dataframe
    input_dataframe = feng.create_calendar_features(data=input_dataframe,
                                                    date_variable=time_variable_meteo)

    # -------------------------- #
    #       LOAD THE MODEL       #
    # -------------------------- #
    model_filename = (
            'models' + os.sep + 'regressor' + os.sep + 'regressor_asset_id_' + str(asset_id_name) + '.pkl')

    try:
        regressor = pickle.load(open(model_filename, 'rb'))
    except FileNotFoundError:
        print(f"missing pickle file in {model_filename}")

    # Final input dataframe
    try:
        input_dataframe = input_dataframe[regressor.feature_name_].copy()
    except KeyError:
        print("\n** ERROR: Input dataframe does not contain the necessary variables **")
        return

    # -------------------------- #
    #         APPLY MODEL        #
    # -------------------------- #
    # Apply model to the input dataframe
    predictions = regressor.predict(input_dataframe)

    # -------------------------- #
    #       FINAL DATAFRAME      #
    # -------------------------- #
    prediction_dataframe['forecast_value'] = predictions

    # Join the final dataframe from model1 with the original date_range
    prediction_dataframe = date_range.merge(prediction_dataframe, on=time_variable_meteo, how='left')

    # Correct possible negative values
    prediction_dataframe.loc[prediction_dataframe['forecast_value'] < 0, 'forecast_value'] = 0

    # Add column with the forecasted date and a column with the asset_id_name
    prediction_dataframe[asset_id_name_variable] = asset_id_name
    # date the forecast was made
    forecast_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_dataframe['forecast_date'] = forecast_date

    # Save the predictions
    forecast_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_dataframe[[asset_id_name_variable, time_variable_meteo, 'forecast_value', 'forecast_date']].to_csv(
        'prediction_data/predictions_asset_' + str(asset_id_name) +'_'+ str(forecast_date) + '_.csv', sep=';', index=False)

    return


if __name__ == "__main__":
    main()
