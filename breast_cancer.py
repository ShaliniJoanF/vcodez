import numpy as np
import pandas as pd
import joblib
import streamlit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
df = pd.read_csv('Breast Cancer DataSet.csv')
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X=df.drop(columns=['diagnosis'])
y=df['diagnosis']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
joblib.dump(model, "breast_cancer_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
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

