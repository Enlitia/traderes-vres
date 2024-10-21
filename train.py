import os
import pickle
import pandas as pd

# Custom libraries
from libs import models_specs as mspec, dimensionality_reduction as dimred, feature_engineering as feng
from libs.feature_selection import lasso_regressor_run
from libs.error_metrics import error_metrics_evaluation
from libs.models_catalog import light_gbm_regressor
from libs.graphical_analysis import create_graphical_analysis


def main():
    """
    This function is the main function to train the model. It performs the following steps:
    - Get the meteorology and power data
    - Prepare the data
    - Perform PCA
    - Perform feature selection
    - Train the model
    - Save the model
    - Test the model
    - Apply the model to the whole dataset
    - Save the final model
    """

    """
    ================================================================
    GET DATA:
        - Meteorology
        - Power
    ================================================================
    """

    # -------------------------- #
    #  METEO DATA AND SETTINGS   #
    # -------------------------- #
    try:
        meteorology_df = pd.read_csv("train_data/meteorology.csv", sep=';')
    except FileNotFoundError:
        print(f"\n ** ERROR: No meteorology data available. Please include a meteorology.csv file in train_data folder **")
        return

    if len(meteorology_df) == 0:
        print(f"\n ** ERROR: No meteorology data available. Please verify the content of the meteorology.csv file **")
        return

    # Define the id variables
    meteo_id_variable = meteorology_df.columns[0]
    # Define the time variable for the meteo dataset
    time_variable_meteo = meteorology_df.columns[1]
    # Define all the meteorology features
    features = meteorology_df.columns[2:]
    # Change the time variable to datetime
    meteorology_df[time_variable_meteo] = pd.to_datetime(meteorology_df[time_variable_meteo])
    # meteo points info
    meteo_points = meteorology_df[meteo_id_variable].unique()

    if len(meteo_points) == 1:
        print(f"** Warning: Only one meteo point available. Please include more meteo points in the meteorology.csv file. PCA will not be perfomed **")

    # -------------------------- #
    #  POWER DATA AND SETTINGS   #
    # -------------------------- #
    try:
        power_df = pd.read_csv("train_data/power.csv", sep=';')
    except FileNotFoundError:
        print(f"\n ** ERROR: No historical power data available. Please include a power.csv file in train_data folder **")
        return

    if len(power_df) == 0:
        print(f"\n ** ERROR: No historical power data available. Please verify the content of the power.csv file **")
        return

    # Define the id variables
    power_id_variable = power_df.columns[0]
    # Define the name of park, asset or country that is being analyzed
    asset_id_name = power_df[power_id_variable].unique()[0]
    # Define the time variable for the target dataset
    time_variable_power = power_df.columns[1]
    # Define the target variable
    target_variable = power_df.columns[2]
    # Change the time variable to datetime
    power_df[time_variable_power] = pd.to_datetime(power_df[time_variable_power])

    # -------------------------- #
    #      CALENDAR FEATURES     #
    # -------------------------- #
    power_df = feng.create_calendar_features(power_df, time_variable_power)

    # -------------------------- #
    #      TRAIN-TEST Split      #
    # -------------------------- #
    # Sort the dataset by the time variable (if it's not already sorted)
    power_df = power_df.sort_values(by=[time_variable_power])

    # Calculate the 80% index
    total_rows = len(power_df)
    # Convert to an integer
    split_index = int(total_rows * 0.8)

    # Set start_train to the minimum date
    start_train = power_df[time_variable_power].min()
    # Set start_test to the date at the 80% index
    start_test = power_df[time_variable_power].iloc[split_index]

    # Set train and test end dataframes
    # Train
    historic_meteo_train_df = meteorology_df[meteorology_df[time_variable_meteo] < start_test].copy()
    historic_power_train_df = power_df[power_df[time_variable_power] < start_test].copy()
    # Test
    historic_meteo_test_df = meteorology_df[meteorology_df[time_variable_meteo] >= start_test].copy()
    historic_power_test_df = power_df[power_df[time_variable_power] >= start_test].copy()

    """
    ================================================================
    DATA PREPARATION:
        - PCA meteo features
        - Feature Selection
    ================================================================
    """

    # -------------------------- #
    #     PCA METEO FEATURES     #
    # -------------------------- #

    # reduce dimensionality of the features using PCA
    if mspec.pca and len(meteo_points) > 1:
        train_pca_features = dimred.dimension_reduction_pca_train(
            meteo_df=historic_meteo_train_df,
            features=features,
            explained_variance=0.85,
            save=False,
            folder=mspec.folder,
            time_variable=time_variable_meteo,
            meteo_id_variable=meteo_id_variable,
            asset_id_name=asset_id_name)
    # or load PCA features from csv
    elif not mspec.pca and len(meteo_points) > 1:
        train_pca_features = dimred.import_pca_from_file(folder=mspec.folder,
                                                         asset_id_name=asset_id_name,
                                                         time_variable=time_variable_meteo)
        train_pca_features = train_pca_features[train_pca_features[time_variable_meteo] < start_test].copy()
    # if only one meteo point is available, PCA is not performed
    else:
        train_pca_features = historic_meteo_train_df.copy()
        train_pca_features = train_pca_features.set_index(time_variable_meteo)
        train_pca_features = train_pca_features.drop([meteo_id_variable], axis=1)

    # Create TRAIN dataframe
    historic_power_train_df = historic_power_train_df.set_index(time_variable_power)
    # add new features to the models features
    df_train = historic_power_train_df.join(train_pca_features)
    # remove NAs
    df_train = df_train.drop([power_id_variable], axis=1)
    df_train = df_train.reset_index()
    # create calendar features
    df_train = feng.create_calendar_features(df_train, time_variable_power)
    # remove NAs
    df_train = df_train.dropna()

    # -------------------------- #
    #      FEATURE SELECTION     #
    # -------------------------- #

    if mspec.feature_selection:
        # Prepare dataset for feature selection
        feature_selection_df = df_train.copy()
        feature_selection_df = feature_selection_df.drop([time_variable_power], axis=1).copy()
        feature_selection_df = feature_selection_df.dropna().copy()
        # calculate good and bad variables
        selected_features, bad_variables = lasso_regressor_run(feature_selection_df, target_variable)

        # Apply linear regression after feature selection
        selected_features = selected_features.tolist()
        selected_features.extend([time_variable_power, target_variable])
        # final dataset considering the selected features
        training_data = df_train[selected_features].copy()
    else:
        training_data = df_train.copy()

    """
    ================================================================
    MODEL TRAINING:
        - Train LightGBM
        - Save models
    ================================================================
    """

    # -------------------------- #
    #    TRAIN DATASET           #
    # -------------------------- #
    # Training dataset
    x_train = training_data.drop([target_variable, time_variable_power], axis=1)
    y_train = training_data[target_variable]

    # -------------------------- #
    #       TRAIN LIGHTGBM       #
    # -------------------------- #
    print("\n ** Training LightGBM **")
    # fit the model on the whole train dataset
    regressor_model = light_gbm_regressor(x_train=x_train, y_train=y_train,
                                          grid_search=mspec.hyper_parameters_grid_search)

    # -------------------------------------- #
    #    APPLY MODEL TO TRAINING DATASET     #
    # -------------------------------------- #
    # predict with the training data to overfitting analysis
    predictions = regressor_model.predict(x_train)
    predictions = pd.DataFrame(predictions, index=training_data.index)
    predictions = predictions.rename(columns={0: 'forecast_value'})

    # Join predictions
    final_dataset = pd.concat([training_data, predictions], axis=1)

    # error evaluation for overfitting analysis
    metrics = error_metrics_evaluation(final_dataset['forecast_value'],
                                       final_dataset[target_variable],
                                       'training')
    print(f"Training metrics: {metrics}")

    """
    ================================================================
    MODEL TESTING:
        - Apply PCA to test dataset
        - Apply LightGBM to test dataset
    ================================================================
    """

    if mspec.pca and len(meteo_points) > 1:
        print("\n ** Apply trained PCA to test dataset **")
        test_pca_features = dimred.dimension_reduction_pca_run(
            meteo_df=historic_meteo_test_df,
            features=features,
            folder=mspec.folder,
            time_variable=time_variable_meteo,
            meteo_id_variable=meteo_id_variable,
            asset_id_name=asset_id_name)
    elif not mspec.pca and len(meteo_points) > 1:
        print("\n ** Apply trained PCA to test dataset **")
        test_pca_features = dimred.import_pca_from_file(folder=mspec.folder,
                                                        asset_id_name=asset_id_name,
                                                        time_variable=time_variable_meteo)
        test_pca_features = test_pca_features[test_pca_features[time_variable_meteo] >= start_test].copy()
    else:
        # if only one meteo point is available, PCA is not performed
        test_pca_features = historic_meteo_test_df.copy()
        test_pca_features = test_pca_features.set_index(time_variable_meteo)
        test_pca_features = test_pca_features.drop([meteo_id_variable], axis=1)

    # Create TRAIN dataframe
    historic_power_test_df = historic_power_test_df.set_index(time_variable_power)
    # add new features to the models features
    df_test = historic_power_test_df.join(test_pca_features)
    # remove NAs
    df_test = df_test.drop([power_id_variable], axis=1)
    df_test = df_test.reset_index()

    if mspec.feature_selection:
        testing_data = df_test[selected_features].copy()
    else:
        testing_data = df_test.copy()

    print("\n ** Apply trained regressor to the test dataset **")

    x_test = testing_data.drop([target_variable, time_variable_power], axis=1)

    predictions = regressor_model.predict(x_test)
    predictions = pd.DataFrame(predictions, index=testing_data.index)
    predictions = predictions.rename(columns={0: 'forecast_value'})

    # Join predictions
    test_data = pd.concat([testing_data, predictions], axis=1)

    # Correct possible negative values
    test_data.loc[test_data['forecast_value'] < 0, 'forecast_value'] = 0

    # error evaluation for overfitting analysis
    metrics = error_metrics_evaluation(test_data['forecast_value'],
                                       test_data[target_variable],
                                       'test')
    print(f"Test metrics: {metrics}")

    if mspec.graphical_output:
        # Create graphical analysis
        create_graphical_analysis( df=test_data,
                                   date_variable=time_variable_power,
                                   target_variable=target_variable,
                                   forecast_variable='forecast_value')

    """
    ================================================================
    TRAIN WITH ALL DATA
    ================================================================
    """

    # -------------------------- #
    #     PCA METEO FEATURES     #
    # -------------------------- #

    # reduce dimensionality of the features using PCA
    if mspec.pca and len(meteo_points) > 1:
        print("\n ** Apply PCA to the complete dataset **")
        train_pca_features = dimred.dimension_reduction_pca_train(
            meteo_df=meteorology_df,
            features=features,
            explained_variance=0.85,
            save=True,
            folder=mspec.folder,
            time_variable=time_variable_meteo,
            meteo_id_variable=meteo_id_variable,
            asset_id_name=asset_id_name)
    elif not mspec.pca and len(meteo_points) > 1:
        print("\n ** Apply PCA to the complete dataset **")
        # load PCA features from csv
        train_pca_features = dimred.import_pca_from_file(folder=mspec.folder,
                                                         asset_id_name=asset_id_name,
                                                         time_variable=time_variable_meteo)
    else:
        # if only one meteo point is available, PCA is not performed
        train_pca_features = meteorology_df.copy()
        train_pca_features = train_pca_features.set_index(time_variable_meteo)
        train_pca_features = train_pca_features.drop([meteo_id_variable], axis=1)

    # Create TRAIN dataframe
    historic_power_train_df = power_df.set_index(time_variable_power)
    # add new features to the models features
    df_train = historic_power_train_df.join(train_pca_features)
    # remove NAs
    df_train = df_train.drop([power_id_variable], axis=1)
    df_train = df_train.reset_index()
    # remove NAs
    df_train = df_train.dropna()

    # -------------------------- #
    #      FEATURE SELECTION     #
    # -------------------------- #

    if mspec.feature_selection:
        # prepare dataset for feature selection
        feature_selection_df = df_train.copy()
        feature_selection_df = feature_selection_df.drop([time_variable_power], axis=1).copy()
        feature_selection_df = feature_selection_df.dropna().copy()
        # calculate good and bad variables
        selected_features, bad_variables = lasso_regressor_run(feature_selection_df, target_variable)

        # Apply linear regression after feature selection
        selected_features = selected_features.tolist()
        selected_features.extend([time_variable_power, target_variable])
        # final dataset considering the selected features
        training_data = df_train[selected_features].copy()

    if not mspec.feature_selection:
        training_data = df_train.copy()

    print("\n** Training regressor with the complete dataset **")
    # Training dataset
    x_train = training_data.drop([target_variable, time_variable_power], axis=1)
    y_train = training_data[target_variable]
    # fit the model on the whole train dataset
    regressor_model = light_gbm_regressor(x_train=x_train, y_train=y_train,
                                          grid_search=mspec.hyper_parameters_grid_search)
    if mspec.save_regressor:
        model_filename = (
                'models' + os.sep + 'regressor' + os.sep + 'regressor_asset_id_' + str(asset_id_name) + '.pkl')
        # use the 'wb' mode to write the binary data
        with open(model_filename, 'wb') as file:
            # serialize the model and write it to the file
            pickle.dump(regressor_model, file)
        print(f"** Model saved in {model_filename} **")

    return


if __name__ == "__main__":
    main()
