import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


data = pd.read_csv("titanic.csv")

# Select features
X = data[["Pclass", "Sex", "Age", "Fare"]]
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X = X.fillna(X.mean())

y = data["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("titanic_linreg.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Linear Regression model trained and saved as titanic_linreg.pkl")

