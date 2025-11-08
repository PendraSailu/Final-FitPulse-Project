import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from prophet import Prophet


def extract_tsfresh_features(df, column_name, time_column, forecast_periods=60):
    df = df.sort_values(by=time_column)
    df = df[[time_column, column_name]].reset_index(drop=True)

    tsfresh_df, y = make_forecasting_frame(df[column_name], kind="data", max_timeshift=5, rolling_direction=1)

    features = extract_features(tsfresh_df, column_id="id", column_sort="time")

    prophet_df = df.rename(columns={time_column: "ds", column_name: "y"})

    model = Prophet(seasonality_mode='additive')
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)

    return features, forecast, model  