# Bank Customer Churn Prediction

Predicting credit card customer churn using Logistic Regression, Random Forest, and XGBoost on the Kaggle BankChurners dataset (10,000+ records). Built with proper ML practices — no data leakage, cross-validation, class imbalance handling, and regularization.

---

## Project Structure

```
bank-churn-prediction/
├── bank_churn_prediction.py   # Main script
├── model_evaluation.png       # Accuracy, ROC-AUC, confusion matrix
├── feature_importance.png     # Top 15 XGBoost features
├── cv_scores.png              # 5-fold CV distribution (overfitting check)
└── README.md
```

---

## Dataset

**Credit Card Customers** by Sakshi Goyal — [Kaggle link](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)

10,127 customers × 21 features. Target: `Attrition_Flag` (~16% churn rate).

The script downloads the dataset automatically via the Kaggle API — no manual download needed.

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/bank-churn-prediction.git
cd bank-churn-prediction
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost kaggle
```

### 3. Set up Kaggle API credentials
1. Go to [kaggle.com](https://www.kaggle.com) → Account → **Create New API Token**
2. Place the downloaded `kaggle.json` at:
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

### 4. Run
```bash
python bank_churn_prediction.py
```

---

## Results

| Model | Test Accuracy | ROC-AUC |
|-------|:------------:|:-------:|
| Logistic Regression | 85.3% | 91.9% |
| Random Forest | 93.3% | 97.4% |
| **XGBoost** | **97.2%** | **99.3%** |

XGBoost 5-fold CV ROC-AUC: **99.33% ± 0.17%** — consistent across all folds, no overfitting.

---

## ML Design Decisions

| Problem | Solution |
|---------|----------|
| Data leakage | Train/test split happens **before** any imputation or encoding |
| Leakage columns | Dropped Naive Bayes probability columns from original dataset |
| Wrong encoding | Nominal features → One-Hot; Ordinal features → Ordinal with explicit order |
| Class imbalance (84/16) | `class_weight="balanced"` in LR & RF; `scale_pos_weight` in XGBoost |
| Overfitting | Regularization (C, max_depth, min_samples_leaf, reg_alpha/lambda, subsampling) |
| Single split luck | 5-fold stratified cross-validation on train set before touching test set |
| Misleading metric | ROC-AUC reported alongside accuracy (better for imbalanced data) |

---

## Author

**Dilshodbek Aliev** — B.Sc. Economics with Data Science, Westminster International University in Tashkent
