import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Titanic Survival with Linear Regression", layout="centered")

st.title("🚢 Titanic Survival Prediction (Linear Regression)")
st.write("This app trains a Linear Regression model on the Titanic dataset each time it runs.")

# Load dataset
data = pd.read_csv("titanic.csv")

# Prepare features
X = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X = X.fillna(X.mean())
y = data["Survived"]

# Train model (no pickle)
model = LinearRegression()
model.fit(X, y)

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

    survival = 1 if prediction >= 0.5 else 0  # Threshold
    prob = min(max(prediction, 0), 1)         # Clamp 0–1

    if survival == 1:
        st.success(f"✅ Likely Survived (predicted: {prediction:.2f}, probability: {prob:.2f})")
    else:
        st.error(f"❌ Likely Did not Survive (predicted: {prediction:.2f}, probability: {prob:.2f})")
