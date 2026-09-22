import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

st.title("Machine Temperature Predictor")
st.write("Predict the next machine temperature using the last two timestamp readings.")

# Input fields for Timestamp 1
st.subheader("Previous Timestamp 1")
col1, col2 = import streamlit as st
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
st.columns(2)
t1_temp = col1.number_input("Previous Timestamp 1 - Temperature", value=81.0)
t1_vib = col2.number_input("Previous Timestamp 1 - Vibration", value=3.5)

# Input fields for Timestamp 2
st.subheader("Previous Timestamp 2")
col3, col4 = st.columns(2)
t2_temp = col3.number_input("Previous Timestamp 2 - Temperature", value=83.0)
t2_vib = col4.number_input("Previous Timestamp 2 - Vibration", value=3.6)

if st.button("Predict Next Temperature"):
    # Load the model
    model = load_model("machine_temperature_rnn.keras")

    # Prepare sequence: shape (1, 2, 2) -> (samples, time_steps, features)
    input_seq = np.array([[[t1_temp, t1_vib], [t2_temp, t2_vib]]])

    # Predict
    prediction = model.predict(input_seq)

    # Display Result
    st.success(f"Predicted Next Machine Temperature:\n{prediction[0][0]:.2f} °C")

