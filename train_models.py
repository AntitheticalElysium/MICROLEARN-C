import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_regression, make_classification
import joblib
import os

os.makedirs("models", exist_ok=True)

print("Training Linear Regression...")
X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
model = LinearRegression()
model.fit(X, y)
joblib.dump(model, "models/linear_model.joblib")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

print("\nTraining Logistic Regression...")
X, y = make_classification(
    n_samples=100,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    random_state=42,
)
model = LogisticRegression()
model.fit(X, y)
joblib.dump(model, "models/logistic_model.joblib")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

print("\nTraining Decision Tree...")
X, y = make_classification(
    n_samples=100,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    random_state=42,
)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X, y)
joblib.dump(model, "models/tree_model.joblib")
print(f"Tree depth: {model.get_depth()}")
print(f"Number of leaves: {model.get_n_leaves()}")

print("\nAll models saved to models/")
