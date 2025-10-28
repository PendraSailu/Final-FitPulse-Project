# 🩺 FitPulse: Health Anomaly Detection from Fitness Devices  

> **AI-powered anomaly detection system** that analyzes heart rate, steps, and sleep data from fitness trackers to detect irregular health patterns using advanced time-series modeling and clustering techniques.  

## 🚀 Overview  

**FitPulse** is a data-driven health monitoring system that identifies unusual patterns in physiological and activity data.  
It leverages **machine learning**, **time-series feature extraction**, and **visual analytics** to detect **point**, **contextual**, and **collective anomalies** — enabling early detection of potential health issues.  

## 🧠 Key Features  

- ✅ **Multi-source Data Import** – Supports CSV and JSON fitness data (heart rate, steps, sleep).  
- ✅ **Automated Preprocessing** – Cleans timestamps, aligns time intervals, and fixes missing values.  
- ✅ **Feature Extraction** – Uses **TSFresh** for extracting rich statistical features from time-series data.  
- ✅ **Seasonality Modeling** – Employs **Facebook Prophet** for detecting trends and cyclic health patterns.  
- ✅ **Behavioral Clustering** – Identifies behavioral patterns using **KMeans** and **DBSCAN**.  
- ✅ **Anomaly Detection** –  
  - **Rule-based:** Threshold-based deviation detection.  
  - **Model-based:** Prophet residuals and clustering outliers.  
- ✅ **Interactive Dashboard** – Built with **Streamlit** to visualize insights and export anomaly reports.  

## 🏗️ Project Architecture  

<img width="747" height="354" alt="image" src="https://github.com/user-attachments/assets/608203d3-ae49-4838-8731-25525b160310" />


---

## 🧩 Milestones  

| Milestone | Description |
|------------|--------------|
| **1️⃣ Data Collection & Preprocessing** | Imported heart rate, steps, and sleep data → cleaned, aligned, and resampled. |
| **2️⃣ Feature Extraction & Modeling** | Generated TSFresh features and modeled seasonality with Prophet. |
| **3️⃣ Anomaly Detection & Visualization** | Implemented threshold-based and model-based anomaly detection → plotted with Matplotlib & Plotly. |
| **4️⃣ Streamlit Dashboard & Reporting** | Built an interactive dashboard for data upload, anomaly insights, and exportable reports (CSV/PDF). |

---

## 🧰 Tools & Technologies  

| Category | Tools/Frameworks |
|-----------|------------------|
| **Language** | Python |
| **Libraries** | Pandas, NumPy, Matplotlib, Plotly, Scikit-learn |
| **Feature Extraction** | [TSFresh](https://tsfresh.readthedocs.io/en/latest/) |
| **Time Series Modeling** | [Facebook Prophet](https://facebook.github.io/prophet/) |
| **Clustering Algorithms** | KMeans, DBSCAN |
| **Dashboard Framework** | [Streamlit](https://streamlit.io/) |
| **Data Formats** | CSV, JSON |

---

## ⚙️ How It Works  

1. **Upload** your fitness data (CSV/JSON) in the dashboard.  
2. **Preprocessing** aligns timestamps, fills gaps, and cleans data.  
3. **TSFresh** extracts hundreds of time-series statistical features.  
4. **Prophet** models trends and seasonal behavior in the data.  
5. **Clustering (KMeans/DBSCAN)** identifies behavioral groups.  
6. **Anomaly Engine** flags:  
   - **Point anomalies:** sudden spikes/drops.  
   - **Contextual anomalies:** abnormal in context (e.g., high heart rate while sleeping).  
   - **Collective anomalies:** sustained unusual behavior (e.g., poor sleep for several days).  
7. **Visualize & Export** results via the Streamlit dashboard.

---

## 📊 Example Use Cases  

| Scenario | Anomaly Example |
|-----------|----------------|
| **Heart Rate** | Sudden jump to 180 bpm while idle |
| **Sleep** | Less than 3 hours of sleep for 7 consecutive nights |
| **Steps** | Unusually low step count for several days |

---

## 📁 Project Structure  
<img width="253" height="502" alt="image" src="https://github.com/user-attachments/assets/45a7de7a-932f-4752-af52-a48a394b783e" />




