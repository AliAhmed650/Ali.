import pandas as pd

df = pd.read_csv("Breast_Cancer_Wisconsin_Diagnostic.csv")
df.head()
df.isnull().sum()
X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=4
)

model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
train_acc = model.score(X_train, y_train)
print("Train Accuracy:", train_acc)
test_acc = model.score(X_test, y_test)
print("Test Accuracy:", test_acc)
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Benign", "Malignant"],
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.show()
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=le.classes_, filled=True)
plt.show()