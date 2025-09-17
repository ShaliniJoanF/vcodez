import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import streamlit as st

# Load dataset
df = pd.read_csv("Breast Cancer DataSet.csv")

# Drop empty column
df = df.drop(columns=["Unnamed: 32"])

# Encode target
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Features and target
X = df[['radius_mean']]
y = df["diagnosis"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "breast_cancer_model.pkl")

# Save feature names for frontend
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("✅ Model trained and saved!")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

model = joblib.load("breast_cancer_model.pkl")
features = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.title("🔬 Breast Cancer Diagnosis Prediction")
st.write("Enter the tumor measurement values to predict if it is **Malignant (Cancerous)** or **Benign (Non-Cancerous)**.")

# Create input form
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature}", min_value=0.0, value=0.0, step=0.01)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("🔴 Malignant (Cancerous Tumor)")
    else:
        st.success("🟢 Benign (Non-Cancerous Tumor)")

