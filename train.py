import numpy as np
import pandas as pd
import os
import random

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier


# =====================================================
# 1. Reproducibility
# =====================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(42)


# =====================================================
# 2. Load Data
# =====================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "Heart Disease"
ID_COL = "id"

X = train.drop(columns=[TARGET])
y = train[TARGET]

X_test = test.copy()


# =====================================================
# 3. Safe Feature Engineering
# =====================================================

def feature_engineering(df):
    df = df.copy()
    
    if "Age" in df.columns and "Cholesterol" in df.columns:
        df["Age_Cholesterol"] = df["Age"] * df["Cholesterol"]
    
    return df

X = feature_engineering(X)
X_test = feature_engineering(X_test)


# =====================================================
# 4. Cross Validation Setup
# =====================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))


# =====================================================
# 5. Training Loop
# =====================================================

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    
    print(f"\n===== Fold {fold+1} =====")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    # -------------------------
    # LightGBM (v4+ style)
    # -------------------------
    lgb = LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.01,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=1,
        random_state=42
    )

    lgb.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[
            early_stopping(200),
            log_evaluation(0)
        ]
    )

    oof_lgb[valid_idx] = lgb.predict_proba(X_valid)[:, 1]
    test_lgb += lgb.predict_proba(X_test)[:, 1] / 5


    # -------------------------
    # XGBoost (v1.x style)
    # -------------------------
    xgb = XGBClassifier(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
        random_state=42,
        eval_metric="auc",
        use_label_encoder=False
    )

    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=200,
        verbose=False
    )

    oof_xgb[valid_idx] = xgb.predict_proba(X_valid)[:, 1]
    test_xgb += xgb.predict_proba(X_test)[:, 1] / 5


# =====================================================
# 6. Evaluation
# =====================================================

print("\n===== OOF Scores =====")
print("LGB AUC:", roc_auc_score(y, oof_lgb))
print("XGB AUC:", roc_auc_score(y, oof_xgb))

final_oof = 0.6 * oof_lgb + 0.4 * oof_xgb
final_test = 0.6 * test_lgb + 0.4 * test_xgb

print("Final Blend AUC:", roc_auc_score(y, final_oof))


# =====================================================
# 7. Debug Check
# =====================================================

print("\nPrediction Range Check:")
print("Min:", final_test.min())
print("Max:", final_test.max())
print("Mean:", final_test.mean())


# =====================================================
# 8. Submission
# =====================================================

submission = pd.read_csv("sample_submission.csv")
submission["Heart Disease"] = final_test
submission.to_csv("submission.csv", index=False)

print("\nSubmission saved successfully.")
