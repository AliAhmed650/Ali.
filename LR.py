import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
#%%

df.info()


df.tail(10)

X = df.drop('target', axis=1)
y = df['target']


from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#%%
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True,
            fmt='d',
            cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=5)),
    ('model', LogisticRegression(max_iter=1000))
])

scores = cross_val_score(pipe, X, y, cv=5)

print(scores)
print(scores.mean())

from sklearn.model_selection import cross_validate

results = cross_validate(
    pipe,
    X,
    y,
    cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1'],
    return_train_score=True
)

print(results)
#%%
print(results['train_accuracy'].mean())
print(results['test_accuracy'].mean())
#%%
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    pipe, X, y,
    cv=5,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# calculate mean scores
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

# convert to error
train_error = 1 - train_mean
val_error = 1 - val_mean


plt.plot(train_sizes, train_error, label="Training Error")
plt.plot(train_sizes, val_error, label="Validation Error")

plt.xlabel("Training Size")
plt.ylabel("Error")
plt.title("Training vs Generalization Error")
plt.legend()
plt.show()