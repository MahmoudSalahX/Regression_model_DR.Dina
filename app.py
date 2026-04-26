import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

# 1. Page Config: "wide" layout and a custom icon make it look professional
st.set_page_config(page_title="AI Traffic Forecaster", page_icon="🚦", layout="wide")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('shallow_nn.keras')
    scaler_mean = np.load('scaler_mean.npy')
    scaler_scale = np.load('scaler_scale.npy')
    return model, scaler_mean, scaler_scale

model, scaler_mean, scaler_scale = load_assets()

# Main Title Area
st.title("🚦 AI Traffic Volume Forecaster")
st.markdown("Predictive analytics for interstate traffic based on real-time meteorological conditions.")
st.divider() # Adds a clean horizontal line

# 2. Move Inputs to a Sidebar
with st.sidebar:
    st.header("⚙️ Condition Inputs")
    st.write("Adjust the weather parameters to run a new simulation.")
    
    # Using sliders instead of number inputs makes it much more interactive
    feature_1_celsius = st.slider("Temperature (°C)", min_value=-20.0, max_value=40.0, value=20.0, step=0.5)
    feature_2 = st.slider("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=50.0, step=5.0)
    
    st.divider()
    # Making the button stand out with a primary type
    run_prediction = st.button("Generate Forecast", type="primary", use_container_width=True)

# 3. Default state / Action state
if run_prediction:
    # Math and prediction
    feature_1_kelvin = feature_1_celsius + 273.15
    raw_input = np.array([[feature_1_kelvin, feature_2]])
    scaled_input = (raw_input - scaler_mean) / scaler_scale
    prediction = model.predict(scaled_input)
    predicted_vol = int(prediction[0][0])
    
    # Determine traffic severity for the UI
    if predicted_vol > 4500:
        status, delta_color = "Heavy Congestion", "inverse"
    elif predicted_vol > 2500:
        status, delta_color = "Moderate Traffic", "off"
    else:
        status, delta_color = "Smooth Flow", "normal"

    # 4. Display KPI Metrics in 3 neat columns
    st.subheader("📊 Live Forecast Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Predicted Volume (Vehicles)", value=f"{predicted_vol:,}", delta=status, delta_color=delta_color)
    with col2:
        st.metric(label="Simulated Temp", value=f"{feature_1_celsius}°C")
    with col3:
        st.metric(label="Cloud Cover", value=f"{int(feature_2)}%")

    st.divider()

    # 5. Generate a Fancy Interactive Chart
    st.subheader("📈 Projected 24-Hour Traffic Distribution")
    st.write("This chart projects how the predicted volume might distribute across a standard day.")
    
    # We create a simulated bell curve around your predicted volume to make a beautiful chart
    hours = np.arange(0, 24, 1)
    # Simulate a morning peak and evening peak based on the base prediction
    morning_peak = predicted_vol * np.exp(-0.2 * (hours - 8)**2) 
    evening_peak = predicted_vol * 1.2 * np.exp(-0.2 * (hours - 17)**2)
    base_traffic = predicted_vol * 0.2
    
    simulated_traffic = morning_peak + evening_peak + base_traffic
    
    # Put it in a pandas dataframe and chart it
    chart_data = pd.DataFrame({
        "Time of Day (Hours)": hours,
        "Estimated Vehicles": simulated_traffic
    }).set_index("Time of Day (Hours)")
    
    # native Streamlit area chart looks incredibly sleek
    st.area_chart(chart_data, color="#ff4b4b")

else:
    # What the user sees before they click the button
    st.info("👈 Adjust the sliders in the sidebar and click **Generate Forecast** to see the AI's prediction and analytics.")
