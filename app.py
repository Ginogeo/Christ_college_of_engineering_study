import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

model_path = Path(__file__).parent / "House_Price_Predictor(1).pkl"
model = joblib.load(model_path)

st.title("House Price Predictor")
st.write("Enter the Area , No. of Bedrromms and the No of Floors to predict the price")

area = st.number_input("Area", min_value=0.0, step=0.5)
bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
floors = st.number_input("No. of Floors", min_value=0, step=1)

#[['Area_Sq_Ft',	'Total_Floors','Bedrooms']]
if st.button("Predict"):
	input_data = pd.DataFrame({"Area_Sq_Ft":[area],"Bedrooms":[bedrooms],"Total_Floors":[floors]})
	prediction = model.predict(input_data)[0]
	
	if prediction:
		st.success(f"Price: {prediction:.0f}")
	else:
		st.error("Error Occured")

