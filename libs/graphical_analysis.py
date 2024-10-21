import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go


def create_graphical_analysis(df: pd.DataFrame, date_variable: str, target_variable: str,
                              forecast_variable: str) -> None:
    """
    Create a graphical analysis with the power and the forecasted variables
    :param df: dataframe with the data
    :param date_variable: variable with the time information
    :param target_variable: variable with the observed power
    :param forecast_variable: variable with the forecasted power
    :return:
    """

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=df[date_variable], y=df[target_variable], name="Observed", line=dict(color='black')),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=df[date_variable], y=df[forecast_variable], name="Forecasted",
                   line=dict(color='red')),
        secondary_y=False
    )

    # Add figure title
    fig.update_layout(
        title_text="Power versus Forecast"
    )
    # Set x-axis title
    fig.update_xaxes(title_text="xaxis title", rangeslider_visible=True)
    # Set y-axes titles
    fig.update_yaxes(title_text="<b>primary</b> Power (MW)", secondary_y=False)
    fig.show(renderer="browser")

    return None
