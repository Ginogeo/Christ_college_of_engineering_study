import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

model_path = Path(__file__).parent / "poly_model_ac.pkl"
model = joblib.load(model_path)

st.title("Price Prediction")
st.write("Enter the details")

ac_units= st.number_input("Area", min_value=0.0, step=0.5)
# bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
# floors = st.number_input("No. of Floors", min_value=0, step=1)

if st.button("Predict"):
	input_data = pd.DataFrame({"AC_Units":[ac_units]})
	prediction = model.predict(input_data)[0]
	
	if prediction:
		st.success(f"Price: {prediction:.0f}")
	else:
		st.error("Error Occured")

