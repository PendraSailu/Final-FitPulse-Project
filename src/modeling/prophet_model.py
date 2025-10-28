import pandas as pd
import numpy as np
from prophet import Prophet

def fit_prophet_model(series: pd.Series, forecast_periods: int = 0, freq: str = 'min'):
    series = pd.to_numeric(series, errors='coerce').dropna()

    df = series.reset_index()
    df.columns = ['ds', 'y']

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_periods, freq=freq)

    forecast = model.predict(future)

    forecast_df = df.copy()
    forecast_df['yhat'] = forecast['yhat'][:len(df)]
    forecast_df['residual'] = forecast_df['y'] - forecast_df['yhat']

    return forecast_df, model


if __name__ == '__main__':
    s = pd.Series([70, 72, 75, 78, 120, 85, 80])
    s.index = pd.date_range('2023-01-01', periods=len(s), freq='T')

    model_df, model = fit_prophet_model(s, forecast_periods=10)
    print(model_df)
