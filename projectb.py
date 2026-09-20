"""
DecodeLabs Project 2: Supervised Learning (Fraud Detection Pipeline)
======================================================================

Goal: Build and tune a classification model to detect fraudulent
transactions in a highly imbalanced dataset (99.83% legitimate,
0.17% fraudulent).

This script follows the "Zero-Leakage Protocol" from the brief:
    1. Ditch accuracy. Judge models with Recall, Precision, F1, ROC-AUC.
    2. Use SMOTE to interpolate and generate new fraud examples,
       never just duplicate existing ones.
    3. NEVER apply SMOTE or scalers before the Train/Test split.
    4. ALWAYS use imblearn.pipeline.Pipeline so resampling is safely
       isolated inside cross-validation.
    5. Tune preprocessing and model hyperparameters together, inside
       GridSearchCV.

Dataset note: the brief's numbers (284,807 transactions, 0.17% fraud)
match the well-known public "Credit Card Fraud Detection" dataset
(ULB / Kaggle: mlg-ulb/creditcardfraud). If that is the dataset you
were given, download creditcard.csv and place it next to this script,
then set USE_REAL_DATA = True below. Otherwise this script generates
a synthetic imbalanced dataset with the same proportions so it still
runs end-to-end as a working demo.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# STEP 0: LOAD THE DATA
# ============================================================

USE_REAL_DATA = False  # set True once you have creditcard.csv locally

def load_data():
    if USE_REAL_DATA:
        df = pd.read_csv("creditcard.csv")
        X = df.drop(columns=["Class"])
        y = df["Class"]
        return X, y

    # ---- Synthetic fallback (demo only) ----
    # Mimics the brief's real-world imbalance: ~99.83% legit, ~0.17% fraud
    X, y = make_classification(
        n_samples=20000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.9983, 0.0017],
        flip_y=0.001,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="Class")
    return X, y


# ============================================================
# STEP 1: TRAIN/TEST SPLIT FIRST (before any scaling or SMOTE)
# ============================================================

def split_data(X, y):
    """
    Stratified split: keeps the same fraud/legit ratio in both
    train and test sets. This happens BEFORE any SMOTE or scaling,
    so the test set stays a true, untouched reflection of reality.
    """
    return train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )


# ============================================================
# STEP 2: BUILD THE TWO LEAK-FREE PIPELINES
# ============================================================

def build_logistic_pipeline():
    """
    Logistic Regression needs scaling (sensitive to feature scale),
    so StandardScaler goes first, then SMOTE, then the classifier.
    All three live inside ONE imblearn Pipeline, so during
    cross-validation, scaling stats and synthetic samples are only
    ever computed from the training fold.
    """
    return ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def build_forest_pipeline():
    """
    Random Forest doesn't need scaling (tree splits are based on
    ordinal partitions, not distances), so we skip StandardScaler.
    """
    return ImbPipeline(steps=[
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])


# ============================================================
# STEP 3: HYPERPARAMETER GRIDS
# ============================================================

# Note: parameter names use double underscores to reach into pipeline steps
logistic_param_grid = {
    "smote__k_neighbors": [3, 5, 7],
    "classifier__C": [0.01, 0.1, 1.0],
}

forest_param_grid = {
    "smote__k_neighbors": [3, 5, 7],
    "classifier__max_depth": [10, 20, None],
}


# ============================================================
# STEP 4: TUNE WITH GRIDSEARCHCV (SMOTE stays safely inside each fold)
# ============================================================

def tune_model(pipeline, param_grid, X_train, y_train, model_name):
    """
    GridSearchCV re-applies the whole pipeline (scaler + SMOTE +
    classifier) fresh on every fold, for every parameter combination.
    This guarantees SMOTE never sees the validation fold, so tuning
    stays leak-free. We optimize for ROC-AUC, not accuracy.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )

    print(f"\nTuning {model_name}...")
    grid_search.fit(X_train, y_train)

    print(f"Best params for {model_name}: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC for {model_name}: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


# ============================================================
# STEP 5: FINAL EVALUATION ON UNTOUCHED TEST DATA
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):
    """
    This is the only place the test set is used, once, at the end.
    No scaling or SMOTE parameters were ever fit on this data.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"FINAL EVALUATION: {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    print("Confusion Matrix:")
    print("                Predicted Legit   Predicted Fraud")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Actual Legit         {cm[0][0]:<10}      {cm[0][1]}")
    print(f"Actual Fraud         {cm[1][0]:<10}      {cm[1][1]}")

    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\nROC-AUC:   {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    return {"roc_auc": roc_auc, "precision": precision, "recall": recall}


# ============================================================
# FULL PIPELINE
# ============================================================

def run_fraud_detection():
    print("Loading data...")
    X, y = load_data()
    print(f"Total samples: {len(y)}")
    print(f"Fraud rate: {y.mean() * 100:.3f}%")

    print("\nSplitting into train/test (80/20, stratified)...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train size: {len(y_train)}  |  Test size: {len(y_test)}")

    # --- Logistic Regression ---
    lr_pipeline = build_logistic_pipeline()
    best_lr = tune_model(lr_pipeline, logistic_param_grid, X_train, y_train, "Logistic Regression")
    lr_results = evaluate_model(best_lr, X_test, y_test, "Logistic Regression")

    # --- Random Forest ---
    rf_pipeline = build_forest_pipeline()
    best_rf = tune_model(rf_pipeline, forest_param_grid, X_train, y_train, "Random Forest")
    rf_results = evaluate_model(best_rf, X_test, y_test, "Random Forest")

    # --- Compare ---
    print(f"\n{'='*50}")
    print("MODEL COMPARISON")
    print(f"{'='*50}")
    print(f"{'Metric':<12}{'Logistic Regression':<22}{'Random Forest'}")
    print(f"{'ROC-AUC':<12}{lr_results['roc_auc']:<22.4f}{rf_results['roc_auc']:.4f}")
    print(f"{'Precision':<12}{lr_results['precision']:<22.4f}{rf_results['precision']:.4f}")
    print(f"{'Recall':<12}{lr_results['recall']:<22.4f}{rf_results['recall']:.4f}")


if __name__ == "__main__":
    run_fraud_detection()