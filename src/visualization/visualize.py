import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA

def plot_metric(df, metric_col='heart_rate'):
    fig = px.line(df, x=df.index, y=metric_col, title=f"{metric_col} over Time")
    return fig

def plot_residuals(df):
    fig = px.scatter(df, x=df.index, y='residual', title="Model Residuals")
    return fig

def plot_clusters(features: pd.DataFrame, labels, title="🔍 Cluster Visualization"):
    """
    2D PCA visualization for clustering results.
    Works safely with mixed or partially missing data.
    """
    df = features.copy()

    df = df.select_dtypes(include='number').dropna(axis=1, how='all').fillna(0)

    if df.shape[1] < 2:
        raise ValueError("Not enough numeric features for PCA visualization.")

    pca = PCA(n_components=2)
    components = pca.fit_transform(df)

    plot_df = pd.DataFrame({
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "Cluster": labels
    })

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        title=title,
        color_continuous_scale="Turbo",
        hover_data=["Cluster"]
    )

    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey")))
    fig.update_layout(template="plotly_white")

    return fig

def plot_rule_anomalies(df: pd.DataFrame, feature='heart_rate'):
    """
    Interactive scatter plot for rule-based anomalies.
    """
    if 'anomaly_rule' not in df.columns:
        raise ValueError("No 'anomaly_rule' column found. Run rule_based() first.")

    fig = px.scatter(
        df,
        x=df.index,
        y=feature,
        color=df['anomaly_rule'].map({0: 'Normal', 1: 'Anomaly'}),
        title=f"Rule-based Anomalies in {feature.capitalize()}",
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=1, color="DarkSlateGrey")))
    return fig


def plot_model_anomalies(df: pd.DataFrame, residual_col='residual'):
    """
    Interactive scatter plot for model-based anomalies.
    """
    if 'anomaly_model' not in df.columns:
        raise ValueError("No 'anomaly_model' column found. Run model_based() first.")

    fig = px.scatter(
        df,
        x=df.index,
        y=residual_col,
        color=df['anomaly_model'].map({0: 'Normal', 1: 'Anomaly'}),
        title="Model-based Anomaly Detection (Residuals)",
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=1, color="DarkSlateGrey")))
    return fig
