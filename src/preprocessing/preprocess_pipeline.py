import pandas as pd
from pathlib import Path


def clean_timestamps(df: pd.DataFrame, timestamp_col='timestamp') -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]  

    if timestamp_col not in df.columns:
        for possible in ["timestamp", "time", "datetime", "date"]:
            if possible in df.columns:
                timestamp_col = possible
                break
        else:
            raise KeyError(f"No timestamp-like column found. Got: {df.columns.tolist()}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    return df


def convert_numeric_columns(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    df = df.copy()
    if columns is None:
        columns = df.columns.difference(['timestamp'])
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def fill_missing_values(df: pd.DataFrame, method='ffill') -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns
    if method == 'ffill':
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
    elif method == 'bfill':
        df[numeric_cols] = df[numeric_cols].fillna(method='bfill')
    else:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df

def resample_data(df: pd.DataFrame, timestamp_col='timestamp', freq='1T') -> pd.DataFrame:
    df = clean_timestamps(df, timestamp_col)
    df = df.set_index(timestamp_col)
    numeric_cols = df.select_dtypes(include='number')
    resampled = numeric_cols.resample(freq).mean().ffill()
    return resampled

def preprocess_pipeline(file_path: str, timestamp_col='timestamp', freq='1T') -> pd.DataFrame:
    path = Path(file_path)
    df = pd.read_csv(path) if path.suffix == '.csv' else pd.read_json(path)

    
    df = convert_numeric_columns(df)

   
    df = clean_timestamps(df, timestamp_col)

    
    df = fill_missing_values(df)

   
    df = resample_data(df, timestamp_col, freq)
    return df

if __name__ == '__main__':
    sample_path = '../../data/sample_heart_rate.csv'
    processed = preprocess_pipeline(sample_path)
    print(processed.head())
