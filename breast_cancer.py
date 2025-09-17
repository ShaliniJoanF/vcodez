import numpy as np
import pandas as pd

df = pd.read_csv('/content/Breast Cancer DataSet.csv')
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X=df.drop(columns=['diagnosis'])
y=df['diagnosis']
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import joblib
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
joblib.dump(model, "breast_cancer_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
