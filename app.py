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
with col1: 
    # temp is usually recorded in Kelvin in these standard datasets
    feature_1 = st.number_input("Temperature", value=290.0) 
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