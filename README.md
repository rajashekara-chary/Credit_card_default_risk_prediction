# Credit Card Default Risk Prediction

## 📌 Project Overview

This project focuses on predicting whether a credit card customer is likely to **default on their payment** based on their previous payment behavior, billing information, payment amounts, and customer-related attributes.

This is a **binary classification problem**, where:

* `Y` → Customer is likely to default
* `N` → Customer is not likely to default

The objective is to build a machine learning model that can help identify customers who are at higher risk of credit card default.

---

## 🎯 Problem Statement

To identify whether a credit card customer will **default or not default** based on their previous bill payment history and available customer information.

**Target Variable:** `Default`

**Problem Type:** Binary Classification

---

## 📊 Dataset

The dataset contains **30,000 customer records and 25 columns**.

The dataset includes information such as:

* Credit limit (`LIMIT_BAL`)
* Previous payment status (`PAY_0` to `PAY_6`)
* Previous bill amounts (`BILL_AMT1` to `BILL_AMT6`)
* Previous payment amounts (`PAY_AMT1` to `PAY_AMT6`)
* Customer demographic information
* Default status (`Default`)

The dataset was checked for missing values and duplicate records. The notebook shows **no missing values and no duplicate rows**.

---

## 🔄 Project Workflow

The project follows this machine learning workflow:

Data Collection
      ↓
Data Understanding
      ↓
Data Cleaning
      ↓
Class Balancing
      ↓
Problem Definition
      ↓
Feature Selection
      ↓
Train-Test Split
      ↓
Data Preprocessing
      ↓
Model Building
      ↓
Hyperparameter Tuning
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
Deployment


---

## 🧹 Data Cleaning

The dataset was loaded using Pandas and basic data understanding was performed using:

* `df.info()`
* `df.shape`
* `df.describe()`
* `df.head()`
* `df.isnull().sum()`
* `df.duplicated().sum()`

The target column initially contained an extra space in its name, so it was renamed from `default ` to `Default`.

---

## ⚖️ Handling Class Imbalance

The target classes were imbalanced, with fewer default customers than non-default customers.

To address this, the majority class (`N`) was **downsampled** to 14,512 records and then combined with the minority class (`Y`). The resulting dataset was shuffled.

This was done to improve the model's ability to identify default customers instead of simply favoring the majority class.

---

## 🎯 Feature Selection

Mutual Information was used to identify features that have a relationship with the target variable.

```python
from sklearn.feature_selection import mutual_info_classif

mutual_info = mutual_info_classif(X, y)
```

The most informative features in the notebook included:

| Feature     | Mutual Information |
| ----------- | -----------------: |
| `PAY_0`     |           0.090458 |
| `PAY_2`     |           0.056765 |
| `PAY_5`     |           0.043340 |
| `PAY_3`     |           0.040396 |
| `PAY_4`     |           0.039530 |
| `PAY_AMT1`  |           0.031485 |
| `PAY_6`     |           0.026929 |
| `PAY_AMT2`  |           0.023598 |
| `PAY_AMT3`  |           0.021846 |
| `LIMIT_BAL` |           0.017759 |

The payment-status variables were among the most informative features for predicting default.

---

## 🧮 Input and Target Variables

The target variable was separated from the input variables.

```python
y = df['Default']

X = df.drop(
    columns=['ID', 'SEX', 'EDUCATION',
             'MARRIAGE', 'Default', 'AGE']
)
```

The `ID` column was excluded because it is an identifier rather than a predictive feature. The demographic columns removed from the modeling dataset were `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE`.

---

## ✂️ Train-Test Split

The data was divided into training and testing datasets using an **80:20 split**.

`stratify=y` was used to preserve the class distribution between the training and testing datasets.

---

## 🔧 Data Preprocessing

`StandardScaler` was used to standardize the numerical features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_trans = scaler.fit_transform(X_train)
X_test_trans = scaler.transform(X_test)
```

The scaler was fitted only on the training data and then applied to the test data.

---

## 🤖 Machine Learning Models

Different classification algorithms were explored to identify a suitable model for credit-card default prediction.

The project includes experimentation with classification models such as:

* K-Nearest Neighbors (KNN)
* Naive Bayes
* Logistic Regression
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* Gradient Boosting

The models were compared based on their classification performance, with particular attention to the ability to correctly identify the default class.

---

## ⚙️ Hyperparameter Tuning

Hyperparameter optimization was performed using:

* `GridSearchCV`
* `RandomizedSearchCV`

For example, KNN hyperparameters such as:

* `n_neighbors`
* `p`

were tuned to find a better-performing configuration.

The project also uses **Scikit-learn Pipelines** to combine preprocessing and model training.

---

## 📈 Model Evaluation

Because credit-card default data can be imbalanced, simply looking at accuracy can be misleading.

The project therefore focuses on classification metrics such as:

* Precision
* Recall
* F1-score
* F1 Macro
* Classification Report

Particular attention is given to the **F1-score and recall of the default class**, since correctly identifying customers who may default is important for the business problem.

---

## 🚀 Deployment

The trained machine learning model was prepared for deployment using **Streamlit**.

The application allows users to provide customer information and obtain a prediction of whether the customer is likely to default.

```text
User Input
    ↓
Preprocessing
    ↓
Trained ML Model
    ↓
Prediction
    ↓
Default / No Default
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn

### Machine Learning

* Classification
* Feature Selection
* Standardization
* Hyperparameter Tuning
* Model Evaluation
* Pipeline

### Deployment

* Streamlit

### Development Tools

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

## 📁 Project Structure

```text
Credit_card_default_risk_prediction/
│
├── app.py
├── Model.ipynb
├── credit_model.pkl
├── requirements.txt
├── Credit Card Defaulter Prediction.csv
└── README.md
```

---

## 💡 Key Learnings

Through this project, I worked on:

* Understanding a real-world classification problem
* Exploratory data understanding
* Data cleaning
* Handling class imbalance
* Feature selection using Mutual Information
* Train-test splitting with stratification
* Feature scaling using StandardScaler
* Building multiple classification models
* Hyperparameter tuning
* Using Scikit-learn pipelines
* Evaluating classification models using F1-score, recall and precision
* Saving the trained model
* Deploying the machine learning model using Streamlit
* Managing the project using Git and GitHub

---

## 🔮 Future Improvements

Potential improvements include:

* Further feature engineering
* Testing additional ensemble models
* Improving recall for the default class
* More extensive hyperparameter optimization
* Model explainability using SHAP or similar techniques
* Continuous monitoring of deployed model performance

---

## 👨‍💻 Author

**Rajasekhara Chary**

Machine Learning / Data Science Project

---

## ⭐ Project Goal

The ultimate goal of this project is to build a reliable machine learning solution that can help financial institutions **identify customers at risk of credit card default and make better risk-management decisions**.
