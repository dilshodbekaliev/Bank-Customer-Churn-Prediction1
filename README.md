# # Bank Customer Churn Prediction

## Project Overview

This project develops a machine learning model to predict customer churn in a banking environment. Customer churn occurs when clients stop using a bank's services, leading to revenue loss and increased customer acquisition costs.

The objective is to identify customers at risk of leaving so that the bank can implement targeted retention strategies.

---

## Dataset

**Source:** Kaggle Bank Churners Dataset

**Size:** 10,000+ customer records

### Features Include:

* Customer demographics
* Income category
* Credit card type
* Credit limit
* Transaction amounts
* Transaction counts
* Account activity metrics
* Customer-bank relationship information

### Target Variable

* **Attrition_Flag**

  * 0 = Existing Customer
  * 1 = Attrited Customer

---

## Project Workflow

### 1. Data Cleaning

* Removed non-predictive customer ID fields
* Replaced missing categorical values using mode imputation
* Replaced missing numerical values using median imputation
* Removed unnecessary dataset columns

### 2. Exploratory Data Analysis

* Examined feature distributions
* Investigated class imbalance
* Identified outliers using Z-score analysis
* Explored customer behavior patterns

### 3. Feature Engineering

Created additional variables:

* **Credit_Utilization_Gap**

  * Difference between credit limit and available credit

* **Activity_Ratio**

  * Customer transaction frequency relative to account age

These engineered features improved model learning and customer behavior representation.

### 4. Data Preprocessing

* Label encoded categorical variables
* Standardized numerical variables for Logistic Regression
* Applied stratified train-test split to preserve churn distribution

### 5. Machine Learning Models

The following classification algorithms were evaluated:

#### Logistic Regression

* Interpretable baseline model

#### Random Forest

* Captures nonlinear relationships
* Provides feature importance analysis

#### XGBoost

* Gradient boosting algorithm
* Strong performance on structured/tabular data
* Selected as the final model

---

## Model Evaluation

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Results

| Model               | Performance |
| ------------------- | ----------- |
| Logistic Regression | Baseline    |
| Random Forest       | Strong      |
| XGBoost             | Best        |

**Final Model:** XGBoost

**Accuracy:** ~94%

The model achieved strong churn detection performance while maintaining a low false-negative rate.

---

## Key Findings

Customers most likely to churn tend to exhibit:

* Low transaction activity
* Higher inactivity periods
* Reduced engagement with banking services
* Lower utilization of available credit

These insights can help financial institutions design proactive retention strategies.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn

---

## Project Structure

```text
├── data/
│   └── BankChurners.csv
├── notebooks/
├── src/
│   └── churn_prediction.py
├── reports/
│   └── project_presentation.pptx
├── README.md
└── requirements.txt
```

---

## Future Improvements

* Hyperparameter tuning using Grid Search
* Cross-validation
* SHAP explainability analysis
* Time-series validation
* Deployment using Streamlit or Flask
* Real-time churn prediction API

---

## Author

**Dilshodbek Aliev**

Economics with Data Science Student

Machine Learning | Data Analytics | Predictive Modeling

