import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('insurance.csv')


print(df.head())
print("\nMissing Values:\n", df.isna().sum())


X = df[['bmi']]
y = df['charges']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining shape: {X_train.shape}")
print(f"Testing shape: {X_test.shape}")
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


model = LinearRegression()
model.fit(X_train_poly, y_train)

print("\nModel Training Completed")

y_pred = model.predict(X_test_poly)

print("\nSample Predictions:", y_pred[:5])
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))