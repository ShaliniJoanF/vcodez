
import streamlit as st
import numpy as np
import joblib

model = joblib.load('/content/linear_regression_model.joblib')

PassengerId= st.number_input("enter the ID please", min_value=0.0, format="%.2f")
Pclass = st.number_input("Enter the class position ", min_value=0)




# Predict button
if st.button("Predict the survival chances"):
    # Create input array
    input_data = np.array([PassengerId, Pclass]) # predict for 1 person, 6 features

    # Make prediction
   # prediction = model.predict(input_data)[0]
    prediction1=model.predict(input_data)[0]

    #st.success(f"💰 Estimated Insurance Charges: ₹{prediction:,.2f}")
    st.success(f"💰 Estimated survival chances: ₹{prediction1:,.2f}")

