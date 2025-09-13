import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("titanic.csv")

# Encode 'Sex' instead of dropping it
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Select features (drop non-numeric, irrelevant)
X = df.drop(["Name", "Ticket", "Cabin", "Embarked", "Survived"], axis=1)
y = df["Survived"]

X=X.dropna()
y=y[X.index]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R2 Score:", r2)

# Save model
joblib.dump(model, "linear_regression_model.joblib")
print("✅ Model saved as linear_regression_model.joblib")
