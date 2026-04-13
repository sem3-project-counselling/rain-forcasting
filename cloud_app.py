import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="RainCast Cloud | Rain Forecast", 
    page_icon="🌧️",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color: #f8fafc;}
.main-title {font-size: 2.3rem; font-weight: 800; color: #0f172a;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌧️ RainCast Cloud Demo</div>', unsafe_allow_html=True)

# Simplified prediction function for cloud demo
@st.cache_data
def predict_rain(min_temp, max_temp, hum9, hum15, press9, press15, wind9, wind15, rain_today):
    # Rule-based prediction (demo only - mimics ML behavior)
    humidity_avg = (hum9 + hum15) / 2
    pressure_drop = press9 - press15
    temp_diff = max_temp - min_temp
    
    score = 0
    score += humidity_avg / 100 * 0.3
    score += (1 - pressure_drop / 50) * 0.25 if pressure_drop > 0 else 0.25
    score += (1 if rain_today else 0) * 0.2
    score += min(temp_diff / 20, 0.25)
    
    probability = min(score * 100, 95)
    prediction = 1 if probability > 50 else 0
    
    return prediction, probability

st.markdown("### 📝 Enter Weather Data")
col1, col2 = st.columns(2)

with col1:
    min_temp = st.number_input("Min Temp (°C)", 0.0, 50.0, 20.0)
    max_temp = st.number_input("Max Temp (°C)", 0.0, 50.0, 30.0)
    hum9 = st.number_input("Humidity 9AM (%)", 0, 100, 60)
    rain_today = st.selectbox("Rain Today", ["No", "Yes"]) == "Yes"

with col2:
    hum15 = st.number_input("Humidity 3PM (%)", 0, 100, 55)
    press9 = st.number_input("Pressure 9AM (hPa)", 900, 1100, 1015)
    press15 = st.number_input("Pressure 3PM (hPa)", 900, 1100, 1012)
    wind9 = st.number_input("Wind Speed 9AM (km/h)", 0, 100, 10)
    wind15 = st.number_input("Wind Speed 3PM (km/h)", 0, 100, 15)

if st.button("🔮 Predict Tomorrow's Rain", type="primary"):
    pred, prob = predict_rain(min_temp, max_temp, hum9, hum15, press9, press15, wind9, wind15, rain_today)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if pred == 1:
            st.error("🌧️ **RAIN PREDICTED**")
        else:
            st.success("☀️ **NO RAIN**")
    
    with col2:
        st.metric("Rain Chance", f"{prob:.1f}%")
    
    with col3:
        risk = "HIGH" if prob > 70 else "MEDIUM" if prob > 40 else "LOW"
        st.metric("Risk", risk)
    
    st.progress(prob / 100)
    
    st.balloons()

st.caption("💡 Cloud-ready demo. Local app (with ML model) running at localhost:8501")
