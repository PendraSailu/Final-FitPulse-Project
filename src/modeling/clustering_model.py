import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.dropna(axis=1, how="all")

    if df.isna().sum().sum() > 0:
        imputer = SimpleImputer(strategy="mean")
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


    df = df.loc[:, df.nunique() > 1]
    if df.empty:
        raise ValueError("All columns are constant — no data to cluster.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled  
def kmeans_cluster(df: pd.DataFrame, n_clusters: int = 3) -> np.ndarray:
    if df.empty:
        return np.array([])

    X_scaled = _prepare_features(df)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels


def dbscan_cluster(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    if df.empty:
        return np.array([])

    X_scaled = _prepare_features(df)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    return labels


if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "heart_rate": [65, 68, 75, 80],
        "steps": [0, 10, 200, 500],
        "sleep_hours": [0, 0, 0, 0]  
    })

    k_labels = kmeans_cluster(sample_df, n_clusters=2)
    print("KMeans labels:", k_labels)

    d_labels = dbscan_cluster(sample_df, eps=3, min_samples=2)
    print("DBSCAN labels:", d_labels)
