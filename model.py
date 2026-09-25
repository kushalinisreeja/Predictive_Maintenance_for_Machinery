"""
model.py
Trains and evaluates predictive maintenance models:
1. Binary Failure Predictor (Machine Failure 0/1)
2. Multiclass Failure Mode Classifier (TWF, HDF, PWF, OSF, RNF, No Failure)
3. RUL (Remaining Useful Life) Regressor
Exports trained models and evaluation metrics to `models/`.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from features import prepare_training_data, FEATURE_COLUMNS
from data_generator import generate_predictive_maintenance_data

def train_predictive_maintenance_models(data_path: str = None, models_dir: str = "models"):
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load data
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("Dataset not found on disk, generating synthetic benchmark dataset...")
        df = generate_predictive_maintenance_data(num_samples=10000)
        
    X, y_failure, y_failure_type, y_rul = prepare_training_data(df)
    
    # 2. Train / Test Split (Stratified on Machine Failure)
    X_train, X_test, y_fail_train, y_fail_test, y_type_train, y_type_test, y_rul_train, y_rul_test = train_test_split(
        X, y_failure, y_failure_type, y_rul, test_size=0.20, random_state=42, stratify=y_failure
    )
    
    # 3. Model 1: Binary Failure Classifier (Balanced Random Forest)
    print("\nTraining Binary Failure Classifier...")
    failure_clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    failure_clf.fit(X_train, y_fail_train)
    y_fail_pred = failure_clf.predict(X_test)
    y_fail_proba = failure_clf.predict_proba(X_test)[:, 1]
    
    fail_roc_auc = float(roc_auc_score(y_fail_test, y_fail_proba))
    fail_f1 = float(f1_score(y_fail_test, y_fail_pred))
    cm = confusion_matrix(y_fail_test, y_fail_pred).tolist()
    
    print(f"Failure Classifier ROC-AUC: {fail_roc_auc:.4f} | F1: {fail_f1:.4f}")
    
    # 4. Model 2: Multiclass Failure Mode Classifier
    print("\nTraining Failure Mode Diagnostic Classifier...")
    mode_clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    mode_clf.fit(X_train, y_type_train)
    y_type_pred = mode_clf.predict(X_test)
    mode_report = classification_report(y_type_test, y_type_pred, output_dict=True)
    mode_f1_macro = float(mode_report["macro avg"]["f1-score"])
    print(f"Failure Mode Macro F1: {mode_f1_macro:.4f}")
    
    # 5. Model 3: Remaining Useful Life (RUL) Regressor
    print("\nTraining Remaining Useful Life (RUL) Regressor...")
    rul_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        random_state=42,
        n_jobs=-1
    )
    rul_reg.fit(X_train, y_rul_train)
    y_rul_pred = rul_reg.predict(X_test)
    
    rul_rmse = float(np.sqrt(mean_squared_error(y_rul_test, y_rul_pred)))
    rul_mae = float(mean_absolute_error(y_rul_test, y_rul_pred))
    rul_r2 = float(r2_score(y_rul_test, y_rul_pred))
    print(f"RUL Regressor RMSE: {rul_rmse:.2f} hrs | MAE: {rul_mae:.2f} hrs | R2: {rul_r2:.4f}")
    
    # 6. Feature Importances
    feature_importances = {
        feat: round(float(imp), 4)
        for feat, imp in zip(FEATURE_COLUMNS, failure_clf.feature_importances_)
    }
    # Sort feature importances descending
    feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
    
    # 7. Save Models and Metadata
    joblib.dump(failure_clf, os.path.join(models_dir, "failure_classifier.joblib"))
    joblib.dump(mode_clf, os.path.join(models_dir, "failure_mode_classifier.joblib"))
    joblib.dump(rul_reg, os.path.join(models_dir, "rul_regressor.joblib"))
    
    metrics = {
        "binary_classification": {
            "roc_auc": round(fail_roc_auc, 4),
            "f1_score": round(fail_f1, 4),
            "confusion_matrix": cm
        },
        "multiclass_diagnosis": {
            "macro_f1": round(mode_f1_macro, 4),
            "classes": mode_clf.classes_.tolist()
        },
        "rul_regression": {
            "rmse_hours": round(rul_rmse, 2),
            "mae_hours": round(rul_mae, 2),
            "r2_score": round(rul_r2, 4)
        },
        "feature_importances": feature_importances,
        "features": FEATURE_COLUMNS
    }
    
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\nAll models and metrics successfully saved to '{models_dir}/'")
    return metrics

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_file = os.path.join(current_dir, "data", "predictive_maintenance_data.csv")
    models_path = os.path.join(current_dir, "models")
    train_predictive_maintenance_models(data_path=data_file, models_dir=models_path)
