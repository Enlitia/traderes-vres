#!/usr/bin/env python3
import numpy as np
import pandas as pd

def create_calendar_features(data: pd.DataFrame, date_variable: str) -> pd.DataFrame:
    """Criar features de calendário necessárias ao processo"""
    df_new_features = data.copy(deep=True)

    # Data
    df_new_features['date'] = df_new_features[date_variable].apply(lambda x: x.date())
    df_new_features['date'] = pd.to_datetime(df_new_features['date'])
    # Ano, mês, semana, dia, hora e minuto
    # df_new_features['year'] = df_new_features[date_variable].dt.year
    df_new_features['month'] = df_new_features[date_variable].dt.month
    df_new_features['week'] = df_new_features[date_variable].dt.isocalendar().week.astype("int64")
    # df_new_features['day'] = df_new_features[date_variable].dt.day

    # df_new_features['dayofweek'] = df_new_features[date_variable].dt.dayofweek
    df_new_features['hour'] = df_new_features[date_variable].dt.hour
    df_new_features['minute'] = df_new_features[date_variable].dt.minute
    df_new_features['hour_decimal'] = df_new_features['hour'] + df_new_features['minute']/ 60

    # Variáveis circulares
    df_new_features['hour_xx'] = np.cos(df_new_features['hour'] * 2 * np.pi / 24)
    df_new_features['hour_yy'] = np.sin(df_new_features['hour'] * 2 * np.pi / 24)
    df_new_features['month_xx'] = np.cos(df_new_features['month'] * 2 * np.pi / 12)
    df_new_features['month_yy'] = np.sin(df_new_features['month'] * 2 * np.pi / 12)
    df_new_features['week_xx'] = np.cos(df_new_features['week'] * 2 * np.pi / 52)
    df_new_features['week_yy'] = np.sin(df_new_features['week'] * 2 * np.pi / 52)

    # Estação do ano
    # df_new_features['season'] = df_new_features['month'].apply(lambda row: date_to_season(row))
    # df_new_features['period_day'] = df_new_features['hour'].apply(lambda row: hour_to_period_of_day(row))
    # Período do dia
    # df_new_features['period_day'] = df_new_features['hour'].apply(lambda row: hour_to_period_of_day(row))

    return df_new_features.drop(columns=['date', 'hour','minute'])


# Criar função período do dia
# 1 <- madrugada: [0, 6[   2 <- manhã:[6,12 [     3 <- tarde: [12,18[     4 <- noite:[18,24[
def hour_to_period_of_day(hour):
    period = None
    if 0 <= hour < 6:
        period = 1
    elif 6 <= hour < 12:
        period = 2
    elif 12 <= hour < 18:
        period = 3
    elif 18 <= hour < 24:
        period = 4
    else:
        raise ValueError(f"Incorrect hour of the day")
    return period


# Criar função (aprox) estação do ano
# 1 <- Inverno: Dez Jan Fev     2 <- Primavera: Mar Abr Maio    3 <- Verão: Jun Jul Ago     4 <- Outono: Set Out Nov
def date_to_season(month):
    season = None
    if month in [12, 1, 2]:
        season = 1
    elif month in [3, 4, 5]:
        season = 2
    elif month in [6, 7, 8]:
        season = 3
    elif month in [9, 10, 11]:
        season = 4
    else:
        raise ValueError(f"Incorrect month of the year")
    return season


# Criar função estação do ano - abordagem mais exacta
def date_to_season_other_approach(df: pd.DataFrame) -> pd.DataFrame:
    df['season2'] = pd.cut((df.month * 100 + df.day - 320) % 1300, [0, 300, 602, 900, 1300],
                           labels=['spring', 'summer', 'autumn', 'winter'])
    return df