import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="Regression Predictor", layout="centered")

# Load model and scaler data
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('shallow_nn.keras')
    scaler_mean = np.load('scaler_mean.npy')
    scaler_scale = np.load('scaler_scale.npy')
    return model, scaler_mean, scaler_scale

model, scaler_mean, scaler_scale = load_assets()

st.title("Traffic Volume Predictor")
st.write("Enter the weather conditions below to predict the interstate traffic volume.")

# User inputs tailored to your specific dataset
col1, col2 = st.columns(2)
# 1. Change the label and the default value to Celsius
with col1: 
    feature_1_celsius = st.number_input("Temperature (°C)", value=20.0) 

if st.button("Predict Traffic Volume"):
    # 2. Convert the Celsius input to Kelvin before predicting
    feature_1_kelvin = feature_1_celsius + 273.15
    
    # Use the converted Kelvin value in your array
    raw_input = np.array([[feature_1_kelvin, feature_2]])
    
    # ... the rest of the code stays exactly the same
with col2: 
    # clouds_all is a percentage from 0 to 100
    feature_2 = st.number_input("Cloud Cover (%)", value=50.0) 

if st.button("Predict Traffic Volume"):
    # Create the input array
    raw_input = np.array([[feature_1, feature_2]])
    
    # Scale the input using the Kaggle training metrics
    scaled_input = (raw_input - scaler_mean) / scaler_scale
    
    # Predict
    prediction = model.predict(scaled_input)
    
    # Output formatted as an integer since vehicles are whole numbers
    st.success(f"Predicted Traffic Volume: {int(prediction[0][0])} vehicles")
