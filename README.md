# ⚡ Outage Alert App

A smart Streamlit web app that predicts **hourly power outages** based on weather-like inputs, time features, and recent power history.  
Built with **Random Forest**, this tool helps users plan charging or work time ahead of possible outages.

---

## 🚀 Demo

🔗 **Live App:** (https://outage-predictor.streamlit.app/)

---

## 🧠 Overview

The **Outage Alert App** estimates the probability of a power outage (`OFF = 0`) for a given hour and location.  
It combines features like:

- 📅 Time & day (hour, weekday, weekend)
- 🌦️ Environmental inputs (temperature, humidity, rain)
- ⚙️ Recent outage history (1-hour and 24-hour lags)
- 🏙️ Location-based one-hot encoding

---

## ✨ Key Features

✅ **Single-hour Prediction** – Input your time, weather, and recent outage duration to get an instant forecast.  
✅ **24-hour Outlook** – See outage probability trends across the day.  
✅ **Explainability (SHAP)** – Understand why the model made a prediction (e.g., rain, lag features).  
✅ **Threshold Control** – Adjust sensitivity for outage alerts.  
✅ **Interactive Charts** – Visualize outage probability throughout the day.  

---

## 🧩 Model

The app uses a **Random Forest Classifier** trained on time + weather + lag-based features.  
Predictions are expressed as probabilities:  
- **OFF = 0 (Outage)**  
- **ON = 1 (Power available)**  

> For best results, users can provide the recent power status 1 hour and 24 hours ago.

---

## 🖥️ Tech Stack

- **Frontend / UI:** Streamlit  
- **Model Handling:** scikit-learn, joblib  
- **Explainability:** SHAP  
- **Data Manipulation:** Pandas, NumPy  

---

## ⚙️ Installation (Local)

If you want to run this app locally:

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/outage-alert-app.git
cd outage-alert-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
