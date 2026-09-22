import streamlit as st
import numpy as np
import tensorflow as tf

st.title("Machine Temperature Predictor")

model = tf.keras.models.load_model('machine_temperature_rnn2.keras')

t1_temp = st.number_input("Previous Timestamp 1 - Temperature", value=81.0)
t1_vib = st.number_input("Previous Timestamp 1 - Vibration", value=3.5)
t2_temp = st.number_input("Previous Timestamp 2 - Temperature", value=83.0)
t2_vib = st.number_input("Previous Timestamp 2 - Vibration", value=3.6)

if st.button("Predict Next Temperature"):
    input_data = np.array([[[t1_temp, t1_vib], [t2_temp, t2_vib]]], dtype=np.float32)
    prediction = model.predict(input_data)
    pred_temp = prediction[0][0]
    pred_vib = prediction[0][1]
    st.success(f"Predicted Next Machine Temperature: {pred_temp:.2f} °C")
    st.success(f"Predicted Next Machine Vibration: {pred_vib:.2f}")
