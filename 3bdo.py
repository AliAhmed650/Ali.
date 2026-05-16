import pandas as pd

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'Breast_Cancer_Wisconsin_Diagnostic.csv'
df = pd.read_csv(DATA_PATH)
df.head()
print('Dataset shape:', df.shape)
print('\nColumns:')
print(df.columns.tolist())
df.info()

print(df['Diagnosis'].value_counts())

plt.figure(figsize=(5, 4))
df['Diagnosis'].value_counts().plot(kind='bar')
plt.title('Diagnosis Class Distribution')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.show()

processed_df = df.copy()


if 'ID' in processed_df.columns:
    processed_df = processed_df.drop(columns=['ID'])


label_encoder = LabelEncoder()
processed_df['Diagnosis'] = label_encoder.fit_transform(processed_df['Diagnosis'])

mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print('Diagnosis encoding:', mapping)

processed_df.head()

X = processed_df.drop(columns=['Diagnosis'])
y = processed_df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('\ny_train distribution:')
print(y_train.value_counts())
print('\ny_test distribution:')
print(y_test.value_counts())

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=200, random_state=42))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='rbf', probability=True, random_state=42))
    ])
}

results = []
trained_models = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    results.append({
        'Dataset': 'Breast Cancer Wisconsin Diagnostic',
        'Model': model_name,
        'Accuracy': acc
    })
    trained_models[model_name] = model

    print('=' * 70)
    print(model_name)
    print('Accuracy:', round(acc, 4))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    comparison_df = pd.DataFrame(results)
    comparison_df['Accuracy (%)'] = (comparison_df['Accuracy'] * 100).round(2)
    comparison_df = comparison_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

    best_row = comparison_df.iloc[0]
    print('Best model:', best_row['Model'])
    print('Best accuracy:', best_row['Accuracy (%)'], '%')

