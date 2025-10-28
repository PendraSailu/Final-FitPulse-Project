import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

def extract_tsfresh_features(series: pd.Series, window_size: int = 30, step: int = 10, verbose: bool = False) -> pd.DataFrame:

    # Validate and clean input
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return pd.DataFrame()

    # Create rolling windows → multiple "id"s
    data = []
    ids = []
    idx = 0
    for start in range(0, len(series) - window_size, step):
        end = start + window_size
        window = series.iloc[start:end]
        if len(window) == window_size:
            temp_df = pd.DataFrame({
                "id": idx,
                "time": np.arange(len(window)),
                "value": window.values
            })
            data.append(temp_df)
            ids.append(idx)
            idx += 1

    if not data:
        return pd.DataFrame()

    df_all = pd.concat(data, ignore_index=True)

    # Extract features
    X = extract_features(
        df_all,
        column_id="id",
        column_sort="time",
        disable_progressbar=True,
        n_jobs=0
    )
    X = impute(X)
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    # Drop constant columns
    constant_cols = [col for col in X.columns if X[col].std() == 0]
    X = X.drop(columns=constant_cols, errors="ignore")

    if verbose:
        print(f"✅ Generated {X.shape[0]} samples × {X.shape[1]} features")

    return X
