import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import joblib
import os

X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)

model = LinearRegression()
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/linear_model.joblib")

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print("Model saved to models/linear_model.joblib")
