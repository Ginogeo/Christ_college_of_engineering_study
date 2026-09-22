import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

st.title("Machine Temperature Predictor")
st.write("Predict the next machine temperature using the last two timestamp readings.")

# Input fields for Timestamp 1
st.subheader("Previous Timestamp 1")
col1, col2 = st.columns(2)
t1_temp = col1.number_input("Previous Timestamp 1 - Temperature", value=81.0)
t1_vib = col2.number_input("Previous Timestamp 1 - Vibration", value=3.5)

# Input fields for Timestamp 2
st.subheader("Previous Timestamp 2")
col3, col4 = st.columns(2)
t2_temp = col3.number_input("Previous Timestamp 2 - Temperature", value=83.0)
t2_vib = col4.number_input("Previous Timestamp 2 - Vibration", value=3.6)

if st.button("Predict Next Temperature"):
    # Load the model
    model = load_model("machine_temperature_rnn2.keras")

    # Prepare sequence: shape (1, 2, 2) -> (samples, time_steps, features)
    input_seq = np.array([[[t1_temp, t1_vib], [t2_temp, t2_vib]]])

    # Predict
    prediction = model.predict(input_seq)

    # Display Result
    st.success(f"Predicted Next Machine Temperature:\n{prediction[0][0]*100:.2f} °C")

