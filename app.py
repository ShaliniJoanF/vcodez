import streamlit as st
import pandas as pd
import pickle

# Load trained linear regression model
with open("titanic_linreg.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Titanic Survival with Linear Regression", layout="centered")

st.title("🚢 Titanic Survival Prediction (Linear Regression)")
st.write("Enter passenger details to check survival chance (using Linear Regression).")

# Input form
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, step=1)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, step=1)
fare = st.number_input("Ticket Fare", min_value=0.0, step=0.5, value=30.0)

# Prediction
if st.button("Predict Survival"):
    sex_encoded = 0 if sex == "male" else 1
    input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare]],
                              columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])
    prediction = model.predict(input_data)[0]

    survival = 1 if prediction >= 0.5 else 0  # Thresholding
    prob = min(max(prediction, 0), 1)  # Clamp between 0 and 1

    if survival == 1:
        st.success(f"✅ Likely Survived (predicted value: {prediction:.2f}, prob: {prob:.2f})")
    else:
        st.error(f"❌ Likely Did not Survive (predicted value: {prediction:.2f}, prob: {prob:.2f})")
