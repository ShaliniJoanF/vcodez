import streamlit as st
import pandas as pd



st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to check survival chance.")

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
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Survived with probability {prob:.2f}")
    else:
        st.error(f"❌ Did not survive (probability {prob:.2f})")

