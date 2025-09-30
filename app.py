# app.py
import os
from datetime import datetime, date, time, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Outage Alert App", layout="centered")

# ---------- CONFIG ----------
MODEL_PATH = "power_outage_rf_model.pkl"   # put your model here
DEV_MODE = os.getenv("DEV_MODE", "0") == "1"

DEFAULT_FEATURE_COLUMNS = [
    "temp_C", "humidity_pct", "rain_mm", "rain_flag", "outage_prob_model",
    "duration_off_hours", "hour", "dayofweek", "is_weekend",
    "power_lag1", "power_lag24", "temp_rolling3",
    "location_Enugu", "location_Lagos", "location_Nsukka"
]

# ---------- CACHED MODEL LOADER ----------
@st.cache_resource
def try_load_model(path=MODEL_PATH):
    try:
        if not os.path.exists(path):
            return None, f"Model file not found at `{path}`."
        m = joblib.load(path)
        return m, None
    except Exception as e:
        return None, f"Failed to load model: {e}"

# ---------- HELPERS ----------
def get_feature_columns(model):
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if os.path.exists("X_train_processed.csv"):
        try:
            cols = pd.read_csv("X_train_processed.csv", nrows=0).columns.tolist()
            return cols
        except Exception:
            pass
    return DEFAULT_FEATURE_COLUMNS.copy()

def build_sample_df(dt, location, temp_C, humidity_pct, rain_mm, duration_off_hours,
                    power_lag1=None, power_lag24=None, feature_columns=None):
    if power_lag1 is None:
        power_lag1 = 1
    if power_lag24 is None:
        power_lag24 = 1

    features = {
        "temp_C": float(temp_C),
        "humidity_pct": float(humidity_pct),
        "rain_mm": float(rain_mm),
        "rain_flag": 1 if float(rain_mm) > 0 else 0,
        "outage_prob_model": 0.5,
        "duration_off_hours": int(duration_off_hours),
        "hour": int(dt.hour),
        "dayofweek": int(dt.weekday()),
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "power_lag1": int(power_lag1),
        "power_lag24": int(power_lag24),
        "temp_rolling3": float(temp_C),
    }

    if feature_columns:
        loc_cols = [c for c in feature_columns if c.startswith("location_")]
        for loc_col in loc_cols:
            features[loc_col] = 1 if loc_col.endswith("_" + location) else 0

    df = pd.DataFrame([features])
    if feature_columns:
        df = df.reindex(columns=feature_columns, fill_value=0)
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)

def outage_probability_from_model(model, X):
    if model is None:
        raise RuntimeError("Model not loaded")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        try:
            classes = list(model.classes_)
        except Exception:
            classes = None
        if classes and 0 in classes:
            off_idx = classes.index(0)
            return probs[:, off_idx]
        elif probs.shape[1] == 2:
            return probs[:, 0]
        else:
            pred = model.predict(X)
            return np.array([1.0 if p == 0 else 0.0 for p in pred])
    else:
        pred = model.predict(X)
        return np.array([1.0 if p == 0 else 0.0 for p in pred])

def summarize_windows(dts, probs, threshold=0.5, min_len=1):
    windows, current, cur_probs, cur_times = [], None, [], []
    for dt, p in zip(dts, probs):
        if p >= threshold:
            if current is None:
                current = dt
            cur_probs.append(p)
            cur_times.append(dt)
        else:
            if current is not None:
                if len(cur_times) >= min_len:
                    windows.append({
                        "start": cur_times[0], "end": cur_times[-1],
                        "avg": float(np.mean(cur_probs)), "max": float(np.max(cur_probs)),
                        "hours": len(cur_times)
                    })
                current, cur_probs, cur_times = None, [], []
    if current is not None and len(cur_times) >= min_len:
        windows.append({
            "start": cur_times[0], "end": cur_times[-1],
            "avg": float(np.mean(cur_probs)), "max": float(np.max(cur_probs)),
            "hours": len(cur_times)
        })
    return windows

# ---------- APP ----------
model, load_err = try_load_model(MODEL_PATH)
feature_columns = get_feature_columns(model)

st.title("⚡ Outage Alert App")

if model is not None:
    st.success("Model loaded. Predictions enabled.")
else:
    st.error("Model not loaded. Place your model beside `app.py`.")

st.markdown("Predict hour-level outage probability (OFF = outage).")

# ----- Single-hour prediction -----
st.header("Predict power status (single hour)")

col1, col2, col3 = st.columns([1.2, 1, 1])
with col1:
    date_val = st.date_input("Date", value=date.today())
    time_val = st.time_input("Time", value=time(19, 0))
    dt = datetime.combine(date_val, time_val)
with col2:
    location = st.selectbox("Location", ["Enugu", "Lagos", "Nsukka"])
    temp_C = st.number_input("Temperature (°C)", value=28.0, format="%.1f")
    humidity_pct = st.number_input("Humidity (%)", value=70.0, format="%.1f")
with col3:
    rain_mm = st.number_input("Rain (mm)", value=0.0, format="%.2f")
    duration_off_hours = st.number_input("Recent duration off (hrs), leave at 0 if light is ON currently", value=0, min_value=0, max_value=72)

with st.expander("Optional: provide recent power history"):
    lag1_choice = st.selectbox("Power 1 hour ago", ["Unknown", "ON (1)", "OFF (0)"], index=0)
    lag24_choice = st.selectbox("Power 24 hours ago", ["Unknown", "ON (1)", "OFF (0)"], index=0)
    def choice_to_int(ch):
        if ch == "ON (1)": return 1
        if ch == "OFF (0)": return 0
        return None
    power_lag1 = choice_to_int(lag1_choice)
    power_lag24 = choice_to_int(lag24_choice)

threshold = st.slider("Alert threshold", 0.1, 0.95, 0.6, 0.05)

if st.button("Predict now"):
    if model is None:
        st.error("Model not loaded.")
    else:
        sample = build_sample_df(dt, location, temp_C, humidity_pct, rain_mm, duration_off_hours,
                                 power_lag1, power_lag24, feature_columns)
        prob_off = float(outage_probability_from_model(model, sample)[0])

        st.metric("Outage probability (OFF)", f"{prob_off*100:.1f}%")
        if prob_off >= threshold:
            st.error(f"⚠️ Likely outage at {dt.strftime('%Y-%m-%d %H:%M')} ({prob_off*100:.1f}%)")
        else:
            st.success(f"✅ Power likely ON at {dt.strftime('%Y-%m-%d %H:%M')} ({(1-prob_off)*100:.1f}%)")

        with st.expander("Why this result? (short)"):
            st.write("The model uses weather, time, and recent power history (lag features).")
            st.write("Inputs snapshot (trimmed):")
            st.dataframe(sample.T.rename(columns={0: "value"}).head(30))

        # ---------------- Explainability ----------------
        with st.expander("Why this result? (detailed explainability)"):
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(sample)
                st.write("Feature contributions (using SHAP values):")
                shap_df = pd.DataFrame({
                    "feature": sample.columns,
                    "value": sample.iloc[0].values,
                    "shap_value": shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
                })
                shap_df["abs_val"] = shap_df["shap_value"].abs()
                shap_df = shap_df.sort_values("abs_val", ascending=False).head(10)
                st.dataframe(shap_df[["feature", "value", "shap_value"]])
            except Exception as e:
                if hasattr(model, "feature_importances_"):
                    fi = pd.DataFrame({
                        "feature": feature_columns,
                        "importance": model.feature_importances_
                    }).sort_values("importance", ascending=False).head(10)
                    st.write("Top 10 important features (fallback):")
                    st.dataframe(fi)
                else:
                    st.info("Explainability not available for this model.")
                st.caption(f"(Details: {e})")

# ----- 24-hour outlook -----
st.header("Next 24 hours outlook")

o_col1, o_col2 = st.columns([2, 1])
with o_col1:
    start_date = st.date_input("Outlook start date", value=date.today())
    start_hour = st.selectbox("Start hour", list(range(24)), 0)
    horizon_hours = st.number_input("Hours to forecast", 6, 72, 24, 1)

    if st.button("Run outlook"):
        if model is None:
            st.error("Model not loaded.")
        else:
            start_dt = datetime.combine(start_date, time(start_hour, 0))
            rows, times = [], []
            for h in range(int(horizon_hours)):
                t = start_dt + timedelta(hours=h)
                times.append(t)
                temp = 27 + 3 * np.sin(2 * np.pi * (t.hour / 24))
                humidity = 65 + 8 * -np.sin(2 * np.pi * (t.hour / 24))
                rain = 0.0
                pl1 = power_lag1 if h == 0 else None
                pl24 = power_lag24 if h == 0 else None
                rows.append(build_sample_df(t, location, temp, humidity, rain, 0, pl1, pl24, feature_columns))
            batch = pd.concat(rows, ignore_index=True)
            probs = outage_probability_from_model(model, batch)
            df_out = pd.DataFrame({"datetime": times, "prob_off": probs})
            st.line_chart(df_out.set_index("datetime")["prob_off"])

            windows = summarize_windows(times, probs, threshold)
            if windows:
                st.markdown("### Alert windows (prob ≥ threshold)")
                for w in windows:
                    st.warning(
                        f"**{w['max']*100:.0f}%** chance OFF from **{w['start'].strftime('%a %H:%M')}** "
                        f"to **{w['end'].strftime('%a %H:%M')}** ({w['hours']} hrs)"
                    )
            else:
                st.success("No alert windows above threshold.")
with o_col2:
    st.markdown("**Tips:**")
    st.write("- Provide recent power history for best accuracy.")
    st.write("- Model accuracy depends on good lag data.")
    st.write("- Replace `power_outage_rf_model.pkl` to update the model.")

if DEV_MODE:
    st.info("DEV_MODE enabled — developer tools active.")
