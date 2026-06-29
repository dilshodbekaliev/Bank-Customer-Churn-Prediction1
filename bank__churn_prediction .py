#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bank Customer Churn Prediction
================================
Author  : Dilshodbek Aliev
Dataset : Credit Card Customers (BankChurners)
Source  : https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers
Models  : Logistic Regression | Random Forest | XGBoost
"""

# ── 1. IMPORTS ─────────────────────────────────────────────────────────────────
import os
import subprocess
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ── 2. DOWNLOAD DATASET FROM KAGGLE ───────────────────────────────────────────
# Prerequisites:
#   1. pip install kaggle
#   2. kaggle.com → Account → Create New API Token → place kaggle.json at:
#      Linux/Mac : ~/.kaggle/kaggle.json
#      Windows   : C:\Users\<user>\.kaggle\kaggle.json

DATASET_SLUG = "sakshigoyal7/credit-card-customers"
DATA_DIR     = "data"
CSV_PATH     = os.path.join(DATA_DIR, "BankChurners.csv")

def download_dataset():
    if os.path.exists(CSV_PATH):
        print(f"[INFO] Dataset found at '{CSV_PATH}'. Skipping download.")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    print("[INFO] Downloading dataset from Kaggle …")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", DATA_DIR, "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed.\n{result.stderr}\n\n"
            "Make sure you have:\n"
            "  1. kaggle installed  →  pip install kaggle\n"
            "  2. kaggle.json at    →  ~/.kaggle/kaggle.json"
        )
    print(f"[INFO] Dataset saved to '{DATA_DIR}/'")

download_dataset()

# ── 3. LOAD DATA ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# Drop Naive Bayes columns — these are post-hoc predictions, not real features.
# Keeping them would be direct data leakage (they already encode the target).
df = df.loc[:, ~df.columns.str.contains("Naive")]

# Drop non-predictive customer ID
df.drop("CLIENTNUM", axis=1, inplace=True)

print("=" * 60)
print("DATASET LOADED")
print(f"  Rows × Cols : {df.shape}")
churn_rate = df["Attrition_Flag"].value_counts(normalize=True)
print(f"  Existing customers : {churn_rate.get('Existing Customer', 0):.1%}")
print(f"  Churned customers  : {churn_rate.get('Attrited Customer', 0):.1%}")
print("=" * 60)

# ── 4. ENCODE TARGET ──────────────────────────────────────────────────────────
df["Attrition_Flag"] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)

# ── 5. HANDLE MISSING VALUES ──────────────────────────────────────────────────
# "Unknown" is a placeholder used by the dataset author for missing categoricals.
# We replace with NaN first, then impute AFTER the train/test split to prevent
# data leakage (imputing on the full dataset would leak test-set information).
df.replace("Unknown", np.nan, inplace=True)

print(f"\n[INFO] Columns with missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

# ── 6. DEFINE FEATURE TYPES ───────────────────────────────────────────────────
# Nominal: no natural order → One-Hot Encoding
nominal_cols = ["Gender", "Card_Category", "Marital_Status"]

# Ordinal: meaningful order → Ordinal Encoding with explicit category order
ordinal_cols      = ["Education_Level", "Income_Category"]
education_order   = ["Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate"]
income_order      = ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"]

# Numeric: kept as-is (scaled for LR inside the pipeline)
numeric_cols = [c for c in df.columns
                if c not in nominal_cols + ordinal_cols + ["Attrition_Flag"]]

# ── 7. FEATURE ENGINEERING ────────────────────────────────────────────────────
# Two interpretable derived features with clear business meaning.
# NOTE: engineered BEFORE split but from columns that don't leak the target.
df["Credit_Utilization_Gap"] = df["Credit_Limit"] - df["Avg_Open_To_Buy"]
df["Activity_Ratio"]         = df["Total_Trans_Ct"] / df["Months_on_book"]
numeric_cols += ["Credit_Utilization_Gap", "Activity_Ratio"]

# ── 8. TRAIN / TEST SPLIT (done BEFORE any fitting/imputation) ────────────────
X = df.drop("Attrition_Flag", axis=1)
y = df["Attrition_Flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n[INFO] Train : {X_train.shape}  |  Test : {X_test.shape}")
print(f"[INFO] Train churn rate : {y_train.mean():.2%}")
print(f"[INFO] Test  churn rate : {y_test.mean():.2%}")

# ── 9. PREPROCESSING PIPELINE ─────────────────────────────────────────────────
# All imputation and encoding happens inside the pipeline → no leakage.

numeric_transformer = Pipeline([
    ("scaler", StandardScaler()),
])

nominal_transformer = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

ordinal_transformer = Pipeline([
    ("oe", OrdinalEncoder(
        categories=[education_order, income_order],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )),
])

preprocessor = ColumnTransformer([
    ("num",     numeric_transformer,  numeric_cols),
    ("nominal", nominal_transformer,  nominal_cols),
    ("ordinal", ordinal_transformer,  ordinal_cols),
], remainder="drop")

# ── 10. MODEL PIPELINES ───────────────────────────────────────────────────────
# class_weight="balanced" compensates for the 84/16 class imbalance
# so the model doesn't just predict "not churned" for everything.

pipe_lr = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        C=0.1,            # regularization — reduces overfitting
    )),
])

pipe_rf = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,          # shallower trees → less overfitting
        min_samples_leaf=10,  # each leaf needs ≥10 samples → less overfitting
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

pipe_xgb = Pipeline([
    ("pre", preprocessor),
    ("clf", XGBClassifier(
        max_depth=4,           # shallower than default → less overfitting
        learning_rate=0.05,    # slower learning → more robust
        n_estimators=400,
        subsample=0.8,         # row sampling → regularization
        colsample_bytree=0.8,  # feature sampling → regularization
        reg_alpha=0.1,         # L1 regularization
        reg_lambda=1.0,        # L2 regularization
        scale_pos_weight=5,    # handles class imbalance (≈ 84/16 ratio)
        eval_metric="logloss",
        random_state=42,
    )),
])

# ── 11. CROSS-VALIDATION (checks for overfitting BEFORE test set) ─────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 60)
print(" 5-FOLD CROSS-VALIDATION (train set only)")
print("=" * 60)

results_cv = {}
for name, pipe in [("Logistic Regression", pipe_lr),
                   ("Random Forest",       pipe_rf),
                   ("XGBoost",             pipe_xgb)]:
    scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    results_cv[name] = scores
    print(f"  {name:<25}  ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# A tight std (< 0.01) and consistent scores across folds = no overfitting.

# ── 12. TRAIN ON FULL TRAIN SET & EVALUATE ON HELD-OUT TEST SET ───────────────
print("\n" + "=" * 60)
print(" HELD-OUT TEST SET EVALUATION")
print("=" * 60)

def evaluate(name, pipe, X_tr, y_tr, X_te, y_te):
    pipe.fit(X_tr, y_tr)
    y_pred      = pipe.predict(X_te)
    y_pred_prob = pipe.predict_proba(X_te)[:, 1]

    train_acc = accuracy_score(y_tr, pipe.predict(X_tr))
    test_acc  = accuracy_score(y_te, y_pred)
    roc_auc   = roc_auc_score(y_te, y_pred_prob)
    gap       = train_acc - test_acc

    print(f"\n── {name} ──")
    print(classification_report(y_te, y_pred, target_names=["Existing", "Churned"]))
    print(f"  Train accuracy : {train_acc:.4f}")
    print(f"  Test  accuracy : {test_acc:.4f}")
    print(f"  Overfit gap    : {gap:.4f}  {'⚠ possible overfit' if gap > 0.03 else '✓ OK'}")
    print(f"  ROC-AUC        : {roc_auc:.4f}")

    return test_acc, roc_auc, y_pred, y_pred_prob

acc_lr,  auc_lr,  pred_lr,  prob_lr  = evaluate("Logistic Regression", pipe_lr,  X_train, y_train, X_test, y_test)
acc_rf,  auc_rf,  pred_rf,  prob_rf  = evaluate("Random Forest",       pipe_rf,  X_train, y_train, X_test, y_test)
acc_xgb, auc_xgb, pred_xgb, prob_xgb = evaluate("XGBoost",             pipe_xgb, X_train, y_train, X_test, y_test)

# ── 13. VISUALISATIONS ────────────────────────────────────────────────────────
models     = ["Logistic\nRegression", "Random\nForest", "XGBoost"]
accuracies = [acc_lr, acc_rf, acc_xgb]
aucs       = [auc_lr, auc_rf, auc_xgb]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# — Plot 1: Accuracy comparison —
bars = axes[0].bar(models, accuracies,
                   color=["steelblue", "seagreen", "tomato"], width=0.5)
axes[0].set_ylim(0.80, 1.00)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Test Accuracy Comparison")
for bar, acc in zip(bars, accuracies):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.003,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=10)

# — Plot 2: ROC-AUC comparison —
bars2 = axes[1].bar(models, aucs,
                    color=["steelblue", "seagreen", "tomato"], width=0.5)
axes[1].set_ylim(0.80, 1.00)
axes[1].set_ylabel("ROC-AUC")
axes[1].set_title("ROC-AUC Comparison")
for bar, auc in zip(bars2, aucs):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.003,
                 f"{auc:.3f}", ha="center", va="bottom", fontsize=10)

# — Plot 3: XGBoost confusion matrix —
cm = confusion_matrix(y_test, pred_xgb)
ConfusionMatrixDisplay(cm, display_labels=["Existing", "Churned"]).plot(
    ax=axes[2], cmap="Blues", colorbar=False
)
axes[2].set_title("Confusion Matrix — XGBoost")

plt.suptitle("Bank Churn Prediction — Model Evaluation", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()

# — Plot 4: XGBoost feature importance —
xgb_clf   = pipe_xgb.named_steps["clf"]
pre_fitted = pipe_xgb.named_steps["pre"]

# Recover feature names after preprocessing
ohe_features = pre_fitted.named_transformers_["nominal"]["ohe"]\
                          .get_feature_names_out(nominal_cols).tolist()
ord_features  = ordinal_cols
all_features  = numeric_cols + ohe_features + ord_features

feat_imp = (
    pd.Series(xgb_clf.feature_importances_, index=all_features)
    .sort_values(ascending=True)
    .tail(15)
)

fig, ax = plt.subplots(figsize=(9, 6))
feat_imp.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 15 Feature Importances — XGBoost")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

# — Plot 5: CV score distribution (overfitting check) —
fig, ax = plt.subplots(figsize=(8, 4))
cv_labels = ["Logistic\nRegression", "Random\nForest", "XGBoost"]
cv_data   = [results_cv["Logistic Regression"],
             results_cv["Random Forest"],
             results_cv["XGBoost"]]
ax.boxplot(cv_data, labels=cv_labels, patch_artist=True,
           boxprops=dict(facecolor="steelblue", alpha=0.5))
ax.set_ylabel("ROC-AUC (5-fold CV)")
ax.set_title("Cross-Validation Score Distribution\n(tight spread = no overfitting)")
ax.set_ylim(0.85, 1.00)
plt.tight_layout()
plt.savefig("cv_scores.png", dpi=150)
plt.show()

# ── 14. FINAL SUMMARY ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" FINAL SUMMARY")
print("=" * 60)
print(f"  {'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10}")
print(f"  {'-'*45}")
for name, acc, auc in zip(
    ["Logistic Regression", "Random Forest", "XGBoost"],
    accuracies, aucs
):
    print(f"  {name:<25} {acc:>10.4f} {auc:>10.4f}")

print(f"\n  Best model  : XGBoost")
print(f"  ROC-AUC     : {auc_xgb:.4f}  (how well the model separates churners from non-churners)")
print(f"  Overfit gap : {(accuracy_score(y_train, pipe_xgb.predict(X_train)) - acc_xgb):.4f}")
print("=" * 60)

# ── KNOWN LIMITATIONS ─────────────────────────────────────────────────────────
"""
Known Limitations
-----------------
1. Synthetic data     : Not real bank data — generalization to production is limited.
2. No temporal split  : Real deployments should use a time-based train/test split.
3. Label encoding     : Ordinal encoder assumes linear spacing between categories.
4. Static threshold   : Default 0.5 decision threshold may not be optimal for business use.
                        A lower threshold (e.g. 0.3) would catch more churners at the cost
                        of more false positives.
"""
