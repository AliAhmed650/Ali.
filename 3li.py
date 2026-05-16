import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('insurance.csv')


print("بيانات الملف:")
print(df.head())



df_final = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)



X = df_final.drop('charges', axis=1)
y = df_final['charges']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\n--- نتائج النموذج ---")
print(f"دقة النموذج (R-squared): {r2_score(y_test, y_pred):.2f}")
print(f"متوسط الخطأ (MSE): {mean_squared_error(y_test, y_pred):.2f}")

coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nتأثير كل متغير على التكلفة (Charges):")
print(coeff_df)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('insurance.csv')

print("Data Preview:")
print(df.head())

df_final = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

X = df_final.drop('charges', axis=1)
y = df_final['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Model Results ---")
print(f"R-squared: {r2_score(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nFeature Coefficients:")
print(coeff_df)
