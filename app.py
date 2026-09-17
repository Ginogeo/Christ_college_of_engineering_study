import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

model_path = Path(__file__).parent / "House_Price_Predictor.pkl"
model = joblib.load(model_path)

st.title("House Price Predictor")
st.write("Enter the Area , No. of Bedrromms and the age to predict the price")

area = st.number_input("Area", min_value=0.0, step=0.5)
bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
age = st.number_input("Age", min_value=0, step=1)


if st.button("Predict"):
	input_data = pd.DataFrame({"Area":[area],"Bedrooms":[bedrooms],"Age":[age]})
	prediction = model.predict(input_data)[0]
	
	if prediction:
		st.success(f"Price: {prediction:.0f}")
	else:
		st.error("Error Occured")

