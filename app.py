import streamlit as st
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("employee_performance_ann.keras")

st.title("Employee Performance Predictor")

st.write("Enter the employee details:")

training_hours = st.number_input(
    "Training Hours",
    min_value=0,
    max_value=100,
    value=5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=70
)

if st.button("Predict Performance"):

    # Prepare input
    input_data = np.array([[training_hours, attendance]])

    # Prediction
    probability = model.predict(input_data, verbose=0)[0][0]

    if probability >= 0.5:
        prediction = "Good"
    else:
        prediction = "Needs Improvement"

    st.subheader("Prediction")
    st.success(prediction)

    st.write(
        "Good Probability:",
        round(float(probability) * 100, 2),
        "%"
    )