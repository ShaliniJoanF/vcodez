import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load Titanic dataset (from Kaggle)
data = pd.read_csv("data/titanic.csv")

# Select features
X = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})  # Encode gender
X = X.fillna(X.mean())  # Fill missing values

y = data["Survived"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model
with open("titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved!")
