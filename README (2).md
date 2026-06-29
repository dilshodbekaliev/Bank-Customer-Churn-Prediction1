# Bank Customer Churn Prediction

## Project Overview
This project develops a machine learning pipeline to predict customer churn in a banking environment. Customer churn occurs when clients stop using a bank's services, leading to revenue loss and increased customer acquisition costs.

The objective is to identify customers at risk of leaving so that the bank can implement targeted retention strategies.

---

## Dataset

**Source:** [Kaggle — Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)  
**Size:** 10,127 customer records  
**Downloaded automatically** via the Kaggle API — no manual download needed.

### Features Include:
- Customer demographics
- Income category
- Credit card type
- Credit limit
- Transaction amounts and counts
- Account activity metrics
- Customer-bank relationship information

### Target Variable
- **Attrition_Flag**
  - `0` = Existing Customer
  - `1` = Attrited Customer (churned)

---

## Project Workflow

### 1. Data Cleaning
- Removed non-predictive customer ID column
- Dropped Naive Bayes leakage columns included in the original Kaggle dataset
- Replaced `"Unknown"` placeholders with `NaN` for proper imputation
- Mode imputation for categorical missing values
- Median imputation for numerical missing values
- All imputation done **after** the train/test split to prevent data leakage

### 2. Exploratory Data Analysis
- Examined feature distributions
- Investigated class imbalance (~84% existing, ~16% churned)
- Identified outliers using Z-score analysis (kept intentionally — extreme financial behaviour is predictive)
- Explored customer behaviour patterns

### 3. Feature Engineering
Created two additional interpretable variables:

- **Credit_Utilization_Gap** — difference between credit limit and available credit
- **Activity_Ratio** — customer transaction frequency relative to account age

### 4. Data Preprocessing
Built inside a Scikit-learn `Pipeline` to eliminate any risk of data leakage:

- **Nominal features** (Gender, Card Category, Marital Status) → One-Hot Encoding
- **Ordinal features** (Education Level, Income Category) → Ordinal Encoding with explicit category order
- **Numerical features** → StandardScaler (for Logistic Regression)
- Stratified train/test split to preserve churn ratio across both sets

### 5. Machine Learning Models

#### Logistic Regression
- Interpretable baseline model
- Regularization (`C=0.1`) to prevent overfitting
- `class_weight="balanced"` to handle class imbalance

#### Random Forest
- Captures non-linear relationships
- `max_depth=8`, `min_samples_leaf=10` to control overfitting
- `class_weight="balanced"`

#### XGBoost
- Gradient boosting — best performance on structured/tabular data
- L1 + L2 regularization (`reg_alpha`, `reg_lambda`)
- `scale_pos_weight=5` to handle class imbalance
- Selected as the final model

---

## Model Evaluation

Metrics used:
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC *(primary metric — more reliable than accuracy on imbalanced data)*
- Confusion Matrix
- Train vs Test gap *(overfitting check)*
- 5-Fold Stratified Cross-Validation

### Results

| Model | Test Accuracy | ROC-AUC | Overfit Gap |
|-------|:------------:|:-------:|:-----------:|
| Logistic Regression | 85.3% | 91.9% | ✓ OK |
| Random Forest | 93.3% | 97.4% | ✓ OK |
| **XGBoost** | **97.2%** | **99.3%** | **✓ OK** |

**Final Model:** XGBoost  
**Test Accuracy:** 97.2%  
**ROC-AUC:** 99.3%  
**5-Fold CV ROC-AUC:** 99.33% ± 0.17% *(tight spread confirms no overfitting)*

---

## Key Findings

Customers most likely to churn tend to exhibit:
- Low transaction activity
- Higher inactivity periods
- Reduced engagement with banking services
- Lower utilization of available credit

These insights can help financial institutions design proactive retention strategies.

---

## Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- XGBoost
- Matplotlib & Seaborn

---

## Project Structure

```
bank-churn-prediction/
├── data/
│   └── BankChurners.csv          # Auto-downloaded via Kaggle API
├── bank_churn_prediction.py      # Main script
├── model_evaluation.png          # Accuracy, ROC-AUC, confusion matrix
├── feature_importance.png        # Top 15 XGBoost features
├── cv_scores.png                 # Cross-validation score distribution
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/bank-churn-prediction.git
cd bank-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add Kaggle credentials
# kaggle.com → Account → Create New API Token → place kaggle.json at ~/.kaggle/kaggle.json

# 4. Run
python bank_churn_prediction.py
```

---

## Future Improvements

- Hyperparameter tuning with GridSearchCV / Optuna
- SHAP explainability analysis
- Time-series / chronological train-test split
- Deployment via Streamlit or Flask
- Real-time churn prediction API

---

## Author

**Dilshodbek Aliev**  
Economics with Data Science Student  
Westminster International University in Tashkent  
Machine Learning · Data Analytics · Predictive Modeling
