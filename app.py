import streamlit as st
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("machine_temperature_rnn.keras")

st.title("Machine Performance Predictor")

# 1. Aligned labels with meaningful variable names
temperature = st.number_input(
    label="Temperature",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=0.1,
    format="%.2f"
)

vibration = st.number_input(
    label="Vibration",
    min_value=0.0,
    max_value=100.0,
    value=2.1,
    step=0.1,
    format="%.2f"
)

if st.button("Predict Performance"):

    # 2. Prepare input using the corrected variable names
    # Model expects shape: (batch_size=1, timesteps=2, features=2)
    input_data = np.array([[temperature, vibration]], dtype=np.float32)  # Shape: (1, 2)
    input_data = np.repeat(input_data[:, np.newaxis, :], 2, axis=1)        # Shape: (1, 2, 2)

    # Prediction
    probability = model.predict(input_data, verbose=0)[0][0]

    # 3. Fixed Indentation (using consistent spaces)
    st.write(f"Probability: {float(probability):.2f}")
