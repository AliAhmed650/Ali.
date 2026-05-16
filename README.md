# Ali.
A Machine Learning project to classify breast tumors using Scikit-Learn).
# Breast Cancer Diagnostic System 🏥

This project focuses on classifying breast tumors as **Benign** or **Malignant** using the Breast Cancer Wisconsin Diagnostic dataset. It applies multiple machine learning classifiers to evaluate and compare their performances.

## 🛠️ Features & Techniques
- **Data Preprocessing:** Handling missing values, Label Encoding for targets, and Feature Scaling using `StandardScaler`.
- **Dimensionality Reduction:** Advanced feature transformation using Principal Component Analysis (PCA).
- **Validation:** 5-Fold Cross-Validation and Learning Curves tracking to evaluate generalization error.

## 🤖 Models Implemented
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machines (SVM)

## 📊 Evaluation Metrics
The models achieved high accuracy, evaluated through:
- Classification Reports (Precision, Recall, F1-Score)
- Confusion Matrix Visualizations

## 💻 How to Run
1. Clone the repository.
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run the Python scripts.
# Medical Insurance Cost Predictor 💰

An end-to-end regression analysis project designed to forecast medical insurance charges based on patient demographics and health metrics (Age, BMI, Smoking status, etc.).

## 🛠️ Features & Techniques
- **Data Preprocessing:** Automated categorical data encoding using `OneHotEncoder` and `pd.get_dummies`.
- **Data Pipelines:** Streamlined workflow using Scikit-Learn's `Pipeline` and `ColumnTransformer`.
- **Feature Exploration:** Implementation of single-feature Polynomial Regression (Degree 2) focusing on BMI correlations.

## 🤖 Models Implemented
- Linear Regression
- Random Forest Regressor
- Support Vector Regression (SVR)

## 📊 Metrics Tracked
- **R2 Score** (Coefficient of Determination)
- **RMSE** (Root Mean Squared Error)

## 💻 How to Run
1. Clone the repository.
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run the Python scripts.
