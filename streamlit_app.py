import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

model = joblib.load("best_model.pkl")

scaler = StandardScaler()
df = pd.read_csv("new_final_selected_features_after_vif.csv") 
X = df.drop(columns=["vomitoxin_ppb"])  
scaler.fit(X)  


st.title("DON concentration Prediction")
st.write("Enter the feature values to predict DON concentration in corn samples.")


feature_inputs = []
for feature in X.columns:
    value = st.number_input(f"{feature}", value=0.0)
    feature_inputs.append(value)

input_data = np.array(feature_inputs).reshape(1, -1)
input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data_scaled)[0]  # Make prediction
    st.success(f"Predicted Vomitoxin Level: {prediction:.4f} ppb")
