import streamlit as st
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("machine_temperature_rnn.keras")

st.title(" Predictor")


training_hours = float(st.text_input(
    "Temperature"
))

attendance = float(st.text_input(
    "Vibration"
))

if st.button("Predict Performance"):

    # Prepare input - Model expects (batch, timesteps=2, features=2)
    input_data = np.array([[training_hours, attendance]], dtype=np.float32)  # (1, 2)
    input_data = np.repeat(input_data[:, np.newaxis, :], 2, axis=1)  # (1, 2, 2)

    # Prediction
    probability = model.predict(input_data, verbose=0)[0][0]

    # if probability >= 0.5:
    #     prediction = "Good"
    # else:
    #     prediction = "Needs Improvement"

    # st.subheader("Prediction")
    # st.success(prediction)

    st.write(
        "Probability:",
        round(float(probability)),
        "%"
    )