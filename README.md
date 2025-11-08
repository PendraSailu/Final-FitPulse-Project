### FitPulse – Health Anomaly Detection from Wearable Fitness Data

FitPulse is a system that analyzes time-series data from wearable fitness devices (such as heart rate, steps, and sleep patterns) to detect unusual health behavior. The system preprocesses raw fitness data, extracts meaningful features, forecasts trends, detects anomalies, and visualizes insights through an interactive Streamlit dashboard.


## 📌 Problem Statement

Wearable fitness devices generate large amounts of continuous health data such as heart rate, step count, and sleep duration. Manually analyzing this data is time-consuming and often inaccurate, leading users to miss early signs of health issues.  
This project aims to develop an automated method to process fitness data, detect unusual patterns, and present meaningful insights in a simple and understandable way.


## 📝 Abstract

FitPulse automatically identifies abnormal patterns in fitness data by performing data preprocessing, feature extraction, trend forecasting, and anomaly detection.  
TSFresh is used to extract important statistical features, Prophet is used to model time-based trends, and clustering methods (KMeans & DBSCAN) uncover unusual behavior.  
All results are displayed in a user-friendly Streamlit dashboard where users can upload data, view trends, detect anomalies, and export reports in CSV or PDF formats.


## 🎯 Objectives

- Collect and preprocess health data from wearable devices
- Extract meaningful features using **TSFresh**
- Model time-series trends and seasonality using **Facebook Prophet**
- Detect anomalies using rule-based, model-based, and clustering approaches
- Provide an interactive **Streamlit dashboard** for visualization and reporting
- Identify **point**, **contextual**, and **collective** anomalies for early health monitoring

## 🧱 System Workflow

<img width="751" height="352" alt="image" src="https://github.com/user-attachments/assets/9c00e291-86e8-4915-9972-f8bbe5cce828" />

---

## 🧠 Types of Anomalies Detected

| Type | Meaning | Example |
|------|---------|---------|
| **Point Anomaly** | A single value is significantly different | Sudden heart rate spike (180 bpm at rest) |
| **Contextual Anomaly** | Abnormal only in certain conditions | High heart rate during sleep |
| **Collective Anomaly** | A sequence of values shows abnormal behavior | Consistently low sleep over several days |

---

## 📂 Project Structure


<img width="251" height="520" alt="image" src="https://github.com/user-attachments/assets/970d4209-9ef1-42b4-9e83-74ee9cafe9f9" />

---

## 🔧 Tools & Technologies

| Category | Tools |
|---------|-------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Feature Extraction | TSFresh |
| Forecasting | Facebook Prophet |
| Clustering | Scikit-learn (KMeans, DBSCAN) |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Export Formats | CSV, PDF |

---

## 🧪 Milestones

### **Milestone 1 – Data Collection & Preprocessing**
- Accepts CSV/JSON input
- Removes duplicates and fills missing data
- Aligns timestamps and normalizes values

# Program Outputs: 
# Raw Data Preview
<img width="2400" height="1185" alt="image" src="https://github.com/user-attachments/assets/265ba2eb-2c31-4094-86c9-dad8fa56abf6" />
# Processed Data View
<img width="2400" height="1222" alt="image" src="https://github.com/user-attachments/assets/2454cceb-9412-4ce0-897d-53bce8a14517" />

### **Milestone 2 – Feature Extraction & Modeling**
- Uses **TSFresh** to extract statistical trends
- Uses **Prophet** to model time-series patterns
- Uses **KMeans** and **DBSCAN** to group similar behaviors

# 60-Day Forecast Visualization
<img width="2398" height="1233" alt="image" src="https://github.com/user-attachments/assets/88b58710-987e-4f84-a486-93967c1db3fe" />
- Forecast Residuals and Summary Metrics
<img width="2398" height="1235" alt="image" src="https://github.com/user-attachments/assets/62ff17bb-46f0-4f14-9a68-c5756d3d9694" />
- Cluster Distribution and Silhouette Score
<img width="2400" height="1199" alt="image" src="https://github.com/user-attachments/assets/6b4d1d89-9fa8-40fd-bbe5-35606f4174c1" />

### **Milestone 3 – Anomaly Detection**
- **Rule-based**: Based on threshold limits
- **Model-based**: Forecast vs actual deviation
- **Cluster-based**: Outliers from cluster groupings
# Rule-Based Anomaly Detection
<img width="1306" height="1229" alt="image" src="https://github.com/user-attachments/assets/051e940f-6dcf-4a25-870c-d0ce4cd7143e" />
# Model-Based Anomaly Detection
  <img width="1255" height="1224" alt="image" src="https://github.com/user-attachments/assets/c3f492cc-ad64-48f8-9f9e-46d493a9eaf9" />
# Cluster-Based Anomaly Detection (K-Means Method)
  <img width="1165" height="1248" alt="image" src="https://github.com/user-attachments/assets/0ae4e7ac-7f32-4070-9b47-47de9ba860af" />

### **Milestone 4 – Dashboard & Visualization**
- User uploads data and runs detection process
- Interactive plots show trends and anomalies
- Results exportable in **CSV or PDF**
# Export Reports

<img width="2400" height="1227" alt="image" src="https://github.com/user-attachments/assets/d0c4c6f0-da94-47ba-a534-650561c8ac5c" />

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py


## 🧱 System Workflow

