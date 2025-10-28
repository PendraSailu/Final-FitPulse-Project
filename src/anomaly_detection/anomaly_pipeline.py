import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
class AnomalyDetector:
    def __init__(self, rule_thresholds=None, std_multiplier=2):
        self.rule_thresholds = rule_thresholds or {'heart_rate': (50, 120)}
        self.std_multiplier = std_multiplier

    def rule_based(self, df: pd.DataFrame, feature='heart_rate'):
        """
        Detect anomalies using simple rule-based thresholds.
        """
        low, high = self.rule_thresholds.get(feature, (50, 120))
        df['anomaly_rule'] = ((df[feature] < low) | (df[feature] > high)).astype(int)
        return df

    def model_based(self, df: pd.DataFrame, residual_col='residuals'):
        """
        Detect anomalies using model residuals (e.g., Prophet forecast errors).
        """
        if residual_col not in df.columns:
            raise ValueError(f"'{residual_col}' column not found in DataFrame.")
        
        mean, std = df[residual_col].mean(), df[residual_col].std()
        upper, lower = mean + self.std_multiplier * std, mean - self.std_multiplier * std

        df['anomaly_model'] = ((df[residual_col] > upper) | (df[residual_col] < lower)).astype(int)
        return df

    def cluster_based(self, feature_df: pd.DataFrame, method='kmeans', eps=0.5, n_clusters=3):
        """
        Cluster similar behavior patterns using KMeans or DBSCAN.
        """
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            model = DBSCAN(eps=eps, min_samples=5)
        
        labels = model.fit_predict(feature_df)
        feature_df['cluster'] = labels
        return feature_df, model

    

