import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("insurance.csv")


# فصل الميزات (X) والهدف (y)
X = df.drop('charges', axis=1)
y = df['charges']


numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")


models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Support Vector Regression': SVR(kernel='rbf')
}

results = []


for model_name, model in models.items():

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])


    pipeline.fit(X_train, y_train)


    predictions = pipeline.predict(X_test)


    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    results.append({
        'Model': model_name,
        'R2 Score': r2,
        'RMSE': rmse
    })

    print("="*50)
    print(f"Model: {model_name}")
    print(f"R2 Score (Accuracy equivalent): {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")


results_df = pd.DataFrame(results)
results_df.plot(x='Model', y='R2 Score', kind='bar', color='skyblue', legend=False)
plt.title("Comparison of Regression Models Accuracy (R2 Score)")
plt.ylabel("R2 Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()