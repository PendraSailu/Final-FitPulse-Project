import sys
import os
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ingestion.data_load_pipeline import load_data
from src.preprocessing.preprocess_pipeline import preprocess_pipeline
from src.featuring.feature_extraction import extract_tsfresh_features
from src.modeling.prophet_model import fit_prophet_model
from src.modeling.clustering_model import kmeans_cluster,dbscan_cluster
from src.visualization.visualize import plot_clusters,plot_metric,plot_model_anomalies,plot_residuals,plot_rule_anomalies
from anomaly_detection.anomaly_pipeline import AnomalyDetector


# Page config & styling
st.set_page_config(
    page_title="FitPulse - Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f8fafc;
    font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    color: #1e293b;
}
[data-testid="stHeader"] {
    background-color: #ffffff !important;
    border-bottom: 2px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    padding: 10px 0;
}
h1, h2, h3 {
    color: #0f172a;
    font-weight: 700;
}

.block-container {
    padding: 2rem 2.5rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    margin-top: 10px;
}
.stTabs [data-baseweb="tab"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    color: #1e293b;
    font-weight: 600;
    border-radius: 10px;
    padding: 8px 18px;
    transition: all 0.3s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #0073e6;
    color: #ffffff;
    transform: translateY(-2px);
}
.stTabs [aria-selected="true"] {
    background: #0073e6 !important;
    color: #ffffff !important;
    border-color: #0073e6 !important;
    box-shadow: 0 4px 10px rgba(0,115,230,0.25);
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
    box-shadow: 2px 0 10px rgba(0,0,0,0.04);
    padding: 25px 22px;
}
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #0f172a;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 18px;
    margin-bottom: 18px;
}
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span {
    color: #334155;
    font-size: 14px;
}

.stFileUploader {
    background-color: #f9fafb;
    border: 1px dashed #cbd5e1;
    border-radius: 10px;
    padding: 16px;
    transition: all 0.3s ease;
}
.stFileUploader:hover {
    background-color: #eef6ff;
    border-color: #0073e6;
    box-shadow: 0 4px 12px rgba(0,115,230,0.15);
    transform: translateY(-2px);
}
.stFileUploader label {
    color: #0f172a;
    font-weight: 600;
}
.stFileUploader button {
    background: linear-gradient(90deg, #0073e6, #0096c7);
    color: white !important;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 8px 14px;
    transition: all 0.3s ease;
}
.stFileUploader button:hover {
    background: linear-gradient(90deg, #005bb5, #0077b6);
    transform: translateY(-2px);
}

.stSelectbox, .stSlider, .stRadio, .stTextInput {
    background: #f9fafb !important;
    border-radius: 10px;
    padding: 10px 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: all 0.2s ease;
}
.stSelectbox:hover, .stSlider:hover, .stRadio:hover, .stTextInput:hover {
    border-color: #0073e6;
    box-shadow: 0 2px 8px rgba(0,115,230,0.15);
}

.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, #0073e6, #0096c7);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 18px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    transition: all 0.25s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: linear-gradient(90deg, #005bb5, #0077b6);
    transform: translateY(-3px);
    box-shadow: 0 6px 14px rgba(0,0,0,0.2);
}

[data-testid="stMetric"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    transition: all 0.25s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.1);
}

.card {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    transition: all 0.25s ease;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}
.anomaly-rule { background-color: #fff4e6; color: #d97706; }
.anomaly-model { background-color: #ecfdf5; color: #047857; }
.anomaly-cluster { background-color: #f5f3ff; color: #6d28d9; }

[data-testid="stDataFrame"] {
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

@media (max-width: 768px) {
    [data-testid="stSidebar"] { padding: 15px; }
    .stButton>button { width: 100%; }
}
</style>
""", unsafe_allow_html=True)

st.title("🏋️‍♂️ FitPulse Dashboard")
# Sidebar - File Upload + Options
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV/JSON", type=["csv", "json"])
metric_selection = None
raw_df, processed = None, None
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
    color: #e3f2fd;
    padding: 25px 20px;
    border-right: 1px solid rgba(255,255,255,0.1);
    box-shadow: 4px 0 10px rgba(0,0,0,0.3);
}

[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #90caf9;
    font-size: 20px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
    margin-bottom: 20px;
    border-bottom: 2px solid #1565c0;
    padding-bottom: 6px;
}

[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span {
    color: #cfd8dc;
    font-size: 14px;
}

.stFileUploader {
    background-color: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 10px;
    padding: 15px;
    transition: all 0.3s ease;
}
.stFileUploader:hover {
    background-color: rgba(255,255,255,0.15);
    transform: translateY(-3px);
}

.stFileUploader button {
    background: linear-gradient(90deg, #1e88e5, #42a5f5);
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 8px 14px;
    transition: all 0.3s ease;
}
.stFileUploader button:hover {
    background: linear-gradient(90deg, #1565c0, #1e88e5);
    transform: translateY(-2px);
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    padding: 10px 20px;
    color: white;
    background: linear-gradient(90deg, #1565c0, #42a5f5);
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #0d47a1, #1976d2);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(33,150,243,0.3);
}

[data-testid="stSidebar"]::-webkit-scrollbar {
    width: 8px;
}
[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: #1565c0;
    border-radius: 10px;
}
[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
    background: #42a5f5;
}
</style>
""", unsafe_allow_html=True)


if uploaded_file:
    df = load_data(uploaded_file)
    st.sidebar.success("✅ File loaded successfully!")
    raw_df = df.copy()  

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    processed = preprocess_pipeline(tmp_path)
    os.remove(tmp_path)

    date_cols = [c for c in processed.columns if 'time' in c.lower() or 'date' in c.lower()]
    numeric_cols = [c for c in processed.select_dtypes(include=np.number).columns if c not in date_cols]

    if len(numeric_cols) > 0:
        metric_selection = st.sidebar.selectbox(
            "Select Metric Column", numeric_cols, key="metric_selection_sidebar"
        )
    else:
        st.sidebar.error("No numeric columns available for selection.")
    if "heart_rate" in df.columns:
        df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
        rule_based_count = int((df["heart_rate"] > 100).sum())
    else:
        rule_based_count = 0

tabs = st.tabs(["Raw Data", "Preprocessing", "Forecasting", "Anomalies", "Clustering", "Export Reports"])

if uploaded_file and metric_selection:
    with tabs[0]:
        st.subheader("📊 Raw Data Preview")
        st.dataframe(raw_df.head() if raw_df is not None else "Upload a file to preview raw data.")

    with tabs[1]:
        st.subheader("🧹 Processed Data")
        st.dataframe(processed.head() if processed is not None else "Upload a file to see processed data.")

    with tabs[2]:
        st.subheader("📉 Advanced Forecasting Visualization")
        series = processed[metric_selection]

        df_prophet = pd.DataFrame({
            'ds': processed[date_cols[0]] if date_cols else pd.date_range(start='2023-01-01', periods=len(series)),
            'y': series
        }).dropna(subset=['y'])
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], errors='coerce')
        df_prophet = df_prophet.dropna(subset=['ds'])

        forecast_periods = 60
        forecast_df, model = fit_prophet_model(
            df_prophet.set_index('ds')['y'],
            forecast_periods=forecast_periods,
            freq='D'
        )
        forecast_df = forecast_df.reset_index(drop=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'],
                                 mode='lines+markers', name='Actual',
                                 line=dict(color='#c1272d', width=2),
                                 marker=dict(size=5)))
        if 'yhat' in forecast_df.columns:
            future_dates = pd.date_range(df_prophet['ds'].max(), periods=forecast_periods + 1, freq='D')[1:]
            future_forecast = forecast_df['yhat'].iloc[-forecast_periods:]

            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_forecast,
                mode='lines',
                name='Forecast (Next 60 Days)',
                line=dict(color='#009688', width=3, dash='dot')
            ))
            fig.add_vrect(
                x0=future_dates.min(),
                x1=future_dates.max(),
                fillcolor='rgba(255, 99, 71, 0.1)',  
                layer='below',
                line_width=0,
                annotation_text="Forecast Zone",
                annotation_position="top right",
                annotation_font_size=12,
                annotation_font_color="gray"
            )

        fig.update_layout(title="Actual vs 60-Day Forecast", template="plotly_white",
                          xaxis=dict(title="Date"), yaxis=dict(title=metric_selection),
                          legend=dict(orientation="h", y=-0.2), height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("Rolling Trend Comparison")
        df_trend = df_prophet.copy()
        df_trend["Short_MA"] = df_trend["y"].rolling(window=5).mean()
        df_trend["Long_MA"] = df_trend["y"].rolling(window=20).mean()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=df_trend["ds"], y=df_trend["y"], mode="lines", name="Raw Data", line=dict(color="#c1272d")))
        fig_trend.add_trace(go.Scatter(x=df_trend["ds"], y=df_trend["Short_MA"], mode="lines", name="Short-Term MA", line=dict(color="#ffa600")))
        fig_trend.add_trace(go.Scatter(x=df_trend["ds"], y=df_trend["Long_MA"], mode="lines", name="Long-Term MA", line=dict(color="#003f5c", dash="dot")))
        fig_trend.update_layout(template="plotly_white", title="Short vs Long-Term Trend", height=500)
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("Forecast Error Distribution")
        if 'residual' in forecast_df.columns:
            fig_resid = px.histogram(forecast_df['residual'], nbins=30, marginal="box",
                                     title="Distribution of Forecast Residuals",
                                     color_discrete_sequence=['#c1272d'])
            st.plotly_chart(fig_resid, use_container_width=True)

        st.markdown("### 📊 Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        min_val, mean_val, max_val = series.min(), series.mean(), series.max()
        trend_direction = "⬆️ Increasing" if series.iloc[-1] > series.iloc[0]*1.05 else (
                          "⬇️ Decreasing" if series.iloc[-1] < series.iloc[0]*0.95 else "➖ Stable")
        col1.metric("Min", f"{min_val:.2f}")
        col2.metric("Mean", f"{mean_val:.2f}")
        col3.metric("Max", f"{max_val:.2f}")
        col4.metric("Trend", trend_direction)

    with tabs[3]:
        st.subheader("⚠️ Health Anomaly Detection")

        df_anom = processed.copy()
        detector = AnomalyDetector()
        
        method = st.radio("Select Detection Method", ["Rule-based", "Model-based", "Cluster-based"], horizontal=True)

        if method == "Rule-based":
            low_thr = st.slider("Low Threshold", 40, 100, 50)
            high_thr = st.slider("High Threshold", 100, 200, 120)
            detector.thresholds = {metric_selection: (low_thr, high_thr)}
            df_anom = detector.rule_based(df_anom, feature=metric_selection)

            fig_rule = go.Figure()
            fig_rule.add_trace(go.Scatter(
                x=df_anom.index, y=df_anom[metric_selection],
                mode='lines',
                name='Metric',
                line=dict(color='#0073e6', width=3, shape='spline'),
                hovertemplate='Index: %{x}<br>Value: %{y}<extra></extra>'
            ))
            fig_rule.add_trace(go.Scatter(
                x=df_anom.index, y=np.where(df_anom["anomaly_rule"], df_anom[metric_selection], None),
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=10, symbol='circle-open'),
                hovertemplate='Anomaly<br>Index: %{x}<br>Value: %{y}<extra></extra>'
            ))
            fig_rule.add_hrect(
                y0=low_thr, y1=high_thr,
                fillcolor='rgba(0, 179, 255, 0.15)',
                line_width=0,
                annotation_text="Normal Range",
                annotation_position="top left",
                annotation_font_size=12,
                annotation_font_color="#0073e6"
            )
            fig_rule.update_layout(
                template="plotly_dark",
                title="🚨 Rule-based Anomaly Detection",
                xaxis_title="Time Index",
                yaxis_title=metric_selection,
                height=500,
                legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig_rule, use_container_width=True)
            
            
            st.markdown(f"""
            <div style="background-color:#ffccbc;
                        padding: 20px;
                        border-radius: 12px;
                        text-align: center;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin-top:10px;">
                <h4 style="color:#d84315; margin:0;">🚨 Rule-based Anomalies Detected</h4>
                <h1 style="color:#d84315; margin:5px 0;">{int(df_anom['anomaly_rule'].sum())}</h1>
            </div>
            """, unsafe_allow_html=True)
        if method == "Model-based":
            if "residual" not in forecast_df.columns:
                st.warning("Residuals missing — run forecasting first.")
            else:
                df_anom = forecast_df.copy()
                df_anom = detector.model_based(df_anom, residual_col='residual')

                thr = 2 * df_anom["residual"].std()

                fig_mod = go.Figure()
                fig_mod.add_trace(go.Scatter(
                    x=df_anom.index, y=df_anom["residual"],
                    mode="lines",
                    name="Residuals",
                    line=dict(color="#00b894", width=3, shape='spline'),
                    hovertemplate='Index: %{x}<br>Residual: %{y}<extra></extra>'
                ))
                fig_mod.add_trace(go.Scatter(
                    x=df_anom.index, y=np.where(df_anom["anomaly_model"], df_anom["residual"], None),
                    mode="markers",
                    name="Anomaly",
                    marker=dict(color='red', size=12, symbol='x'),
                    hovertemplate='Anomaly<br>Index: %{x}<br>Residual: %{y}<extra></extra>'
                ))
                fig_mod.add_hrect(
                    y0=-thr, y1=thr,
                    fillcolor='rgba(0, 184, 212,0.15)',
                    line_width=0,
                    annotation_text="Normal Residual Range",
                    annotation_position="top left",
                    annotation_font_color="#00b894"
                )
                fig_mod.update_layout(
                    template="plotly_dark",
                    title="🧠 Residual-based Model Anomaly Detection",
                    xaxis_title="Time Index",
                    yaxis_title="Residual",
                    height=500,
                    legend=dict(orientation="h", y=-0.2)
                )
                st.plotly_chart(fig_mod, use_container_width=True)
 
                
                valid_records = len(raw_df) if raw_df is not None else len(df_anom)
                anomaly_count = int(df_anom['anomaly_model'].sum())
                
                st.markdown(f"""
                <div style="background-color:#b2dfdb;
                            padding: 20px;
                            border-radius: 12px;
                            text-align: center;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin-top:10px;">
                    <h4 style="color:#00695c; margin:0;">🧠 Model-based Anomalies Detected</h4>
                    <h1 style="color:#00695c; margin:5px 0;">{int(df_anom['anomaly_model'].sum())}</h1>
                    <p style="color:#555; margin:0;">Detected via residual deviation</p>
                </div>
                """, unsafe_allow_html=True)

                
        if method == "Cluster-based":
            numeric_cols = df_anom.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) < 2:
                st.warning("Not enough numeric columns for clustering.")
            else:
                cluster_method = st.selectbox("Select Cluster Method", ["KMeans", "DBSCAN"])
                df_anom, cluster_model = detector.cluster_based(df_anom[numeric_cols], method=cluster_method.lower())
                
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(df_anom[numeric_cols])
                df_vis = pd.DataFrame(reduced, columns=["PC1", "PC2"])
                df_vis["Cluster"] = df_anom["cluster"].astype(str)
                df_vis["Anomaly"] = df_anom["cluster"].astype(bool)

                fig_cluster = px.scatter(
                    df_vis, x="PC1", y="PC2",
                    color="Cluster", symbol="Anomaly",
                    title=f"🟣 {cluster_method} Cluster-Based Anomaly Detection",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    symbol_map={False: "circle", True: "x"},
                    hover_data={"PC1":True, "PC2":True, "Cluster":True, "Anomaly":True}
                )

                fig_cluster.update_traces(marker=dict(size=12, opacity=0.9, line=dict(width=1, color='white')))
                fig_cluster.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_cluster, use_container_width=True)
                st.markdown(f"""
                <div style="background-color:#e1bee7;
                            padding: 20px;
                            border-radius: 12px;
                            text-align: center;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin-top:10px;">
                    <h4 style="color:#6a1b9a; margin:0;">🟣 Cluster-based Anomalies Detected</h4>
                    <h1 style="color:#6a1b9a; margin:5px 0;">{int(df_anom['cluster'].sum())}</h1>
                    <p style="color:#555; margin:0;">Highlighted via {cluster_method}</p>
                </div>
                """, unsafe_allow_html=True)  
        if method == "Rule-based":
            anomaly_df = df_anom[df_anom["anomaly_rule"] == True].copy()
        elif method == "Model-based":
            anomaly_df = df_anom[df_anom["anomaly_model"] == True].copy()
        elif method == "Cluster-based":
            anomaly_df = df_anom[df_anom["cluster"] == True].copy()
            
    with tabs[4]:
        st.subheader("Clustering Analysis")
        series = processed[metric_selection].copy().reset_index(drop=True)
        if series.nunique() <= 1:
            st.warning(f"'{metric_selection}' has no variation — clustering cannot be applied.")
            st.stop()

        def rolling_extract_safe(series, window_size=30, step=10):
            from tsfresh import extract_features
            from tsfresh.utilities.dataframe_functions import impute
            series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                return pd.DataFrame()
            data_parts = []
            idx = 0
            for start in range(0, len(series) - window_size + 1, step):
                window = series.iloc[start:start + window_size]
                if len(window) == window_size:
                    temp = pd.DataFrame({"id": idx, "time": np.arange(window_size), "value": window.values})
                    data_parts.append(temp)
                    idx += 1
            if not data_parts:
                return pd.DataFrame()
            df_all = pd.concat(data_parts, ignore_index=True)
            X = extract_features(df_all, column_id="id", column_sort="time", disable_progressbar=True, n_jobs=0)
            X = impute(X).replace([np.inf, -np.inf], 0).fillna(0)
            const_cols = [c for c in X.columns if X[c].std() == 0]
            return X.drop(columns=const_cols, errors='ignore')

        n_points = len(series)
        window_size = max(10, n_points // 5)
        step = max(5, window_size // 2)

        try:
            features = rolling_extract_safe(series, window_size=window_size, step=step)
            if features.shape[0] < 3 or features.shape[1] < 2:
                st.warning("Not enough data or variance for clustering. Try a larger dataset.")
                st.stop()

            klabels = kmeans_cluster(features)
            dlabels = dbscan_cluster(features)

            pca = PCA(n_components=min(3, features.shape[1], features.shape[0]))
            reduced = pca.fit_transform(features)
            df_vis = pd.DataFrame(reduced, columns=[f"PC{i+1}" for i in range(reduced.shape[1])])
            df_vis["KMeans"] = klabels.astype(str)
            df_vis["DBSCAN"] = dlabels.astype(str)

            if df_vis.shape[1] >= 3:
                fig3d = px.scatter_3d(df_vis, x="PC1", y="PC2", z="PC3",
                                      color="KMeans", title="KMeans Clusters (3D View)",
                                      color_discrete_sequence=px.colors.qualitative.Set2)
                fig3d.update_layout(template="plotly_white", height=450)
                st.plotly_chart(fig3d, use_container_width=True)

            fig2d = px.scatter(df_vis, x="PC1", y="PC2", color="DBSCAN",
                               symbol="KMeans", title="2D Comparison",
                               color_discrete_sequence=px.colors.qualitative.Bold)
            fig2d.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig2d, use_container_width=True)

            cluster_counts = pd.Series(klabels).value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Count"]
            fig_bar = px.bar(cluster_counts, x="Cluster", y="Count", color="Cluster", text="Count",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

            if len(np.unique(klabels)) > 1:
                sil = silhouette_score(features, klabels)
                st.success(f"Silhouette Score: {sil:.3f}")
            else:
                st.info("Only one cluster detected — silhouette not applicable.")

            output_df = features.copy()
            output_df["KMeans_Cluster"] = klabels
            output_df["DBSCAN_Cluster"] = dlabels

        except Exception as e:
            st.error(f"⚠️ Clustering failed: {e}")  
    
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    import io

    with tabs[5]:
        st.subheader("📄 Export Reports")

        if uploaded_file and processed is not None and anomaly_df is not None:
            csv_file = anomaly_df.to_csv(index=False).encode('utf-8')
            st.download_button(
            label="⬇️ Download Anomaly CSV",
            data=csv_file,
            file_name="anomaly_report.csv",
            mime="text/csv"
            )
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()

            title = Paragraph("FitPulse - Detailed Anomaly Report", styles['Title'])
            table_data = [anomaly_df.columns.to_list()] + anomaly_df.values.tolist()

            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
                ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 8),
                ('GRID', (0,0), (-1,-1), 0.5, colors.gray)
            ]))

            doc.build([title, table])

            st.download_button(
            label="📄 Download Anomaly PDF Report",
            data=buffer.getvalue(),
            file_name="anomaly_report.pdf",
            mime="application/pdf"
            )

        else:
            st.warning("Upload and process data to enable export.")

