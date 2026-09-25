"""
predict.py
Predictive Maintenance Inference & Prescriptive Decision Engine.
Takes live or batch sensor telemetry and provides:
- Machine Health Index (0 - 100%)
- Failure Probability & Binary Risk Alert
- Diagnosed Failure Mode (TWF, HDF, PWF, OSF, RNF)
- Remaining Useful Life (RUL) estimation
- Actionable Maintenance Recommendations
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List

from features import engineer_features, FEATURE_COLUMNS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

class PredictiveMaintenanceEngine:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.failure_clf = None
        self.mode_clf = None
        self.rul_reg = None
        self.metrics = {}
        self._load_artifacts()

    def _load_artifacts(self):
        failure_model_path = os.path.join(self.models_dir, "failure_classifier.joblib")
        mode_model_path = os.path.join(self.models_dir, "failure_mode_classifier.joblib")
        rul_model_path = os.path.join(self.models_dir, "rul_regressor.joblib")
        metrics_path = os.path.join(self.models_dir, "metrics.json")

        if os.path.exists(failure_model_path):
            self.failure_clf = joblib.load(failure_model_path)
            self.mode_clf = joblib.load(mode_model_path)
            self.rul_reg = joblib.load(rul_model_path)
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    self.metrics = json.load(f)
        else:
            print("Warning: Models not loaded. Please train models first using model.py")

    def predict_single(self, telemetry: Dict[str, Any], risk_threshold: float = 0.40) -> Dict[str, Any]:
        """
        Inference for a single machine's telemetry reading.
        """
        df_raw = pd.DataFrame([telemetry])
        return self.predict_dataframe(df_raw, risk_threshold=risk_threshold)[0]

    def predict_dataframe(self, df_raw: pd.DataFrame, risk_threshold: float = 0.40) -> List[Dict[str, Any]]:
        """
        Inference for a batch dataframe.
        """
        if self.failure_clf is None:
            raise RuntimeError("Models are not loaded. Train them using model.py before inference.")

        X = engineer_features(df_raw)

        # 1. Failure Probability & Alert
        fail_probas = self.failure_clf.predict_proba(X)[:, 1]
        is_failure_predicted = (fail_probas >= risk_threshold).astype(int)

        # 2. Diagnosed Failure Mode
        mode_preds = self.mode_clf.predict(X)
        mode_probas = self.mode_clf.predict_proba(X)
        mode_classes = list(self.mode_clf.classes_)

        # 3. Remaining Useful Life (RUL)
        rul_preds = np.clip(self.rul_reg.predict(X), 1.0, 350.0).round(1)

        results = []
        for i in range(len(df_raw)):
            p_fail = float(fail_probas[i])
            predicted_mode = mode_preds[i]
            
            # If binary failure is low, ensure mode is consistent with normal health
            if p_fail < risk_threshold:
                diagnosed_mode = "No Failure"
            else:
                # If model predicted 'No Failure' but probability exceeded threshold,
                # assign the highest probable non-normal failure mode
                if predicted_mode == "No Failure":
                    probas_dict = {cls: probas for cls, probas in zip(mode_classes, mode_probas[i]) if cls != "No Failure"}
                    if probas_dict:
                        diagnosed_mode = max(probas_dict, key=probas_dict.get)
                    else:
                        diagnosed_mode = "Unspecified Anomaly"
                else:
                    diagnosed_mode = predicted_mode

            # Compute Health Index (0 to 100%)
            # High failure probability directly pulls health index down
            health_index = max(0.0, min(100.0, (1.0 - p_fail) * 100.0))
            
            # Additional penalty for high vibration or temperature
            vib = df_raw.iloc[i].get("Vibration_mms", 2.5)
            if vib > 4.0:
                health_index = max(0.0, health_index - (vib - 4.0) * 10.0)
            health_index = round(health_index, 1)

            # Assign Status Category
            if health_index >= 75.0:
                status_category = "OPTIMAL"
                status_color = "green"
            elif health_index >= 45.0:
                status_category = "WARNING"
                status_color = "orange"
            else:
                status_category = "CRITICAL"
                status_color = "red"

            # Prescriptive Action Generation
            recommendation = self._generate_recommendation(diagnosed_mode, p_fail, rul_preds[i], df_raw.iloc[i])

            res = {
                "health_index": health_index,
                "status_category": status_category,
                "status_color": status_color,
                "failure_probability": round(p_fail, 4),
                "is_failure_predicted": bool(is_failure_predicted[i]),
                "failure_mode": diagnosed_mode,
                "rul_hours": float(rul_preds[i]),
                "recommendation": recommendation,
                "computed_power_w": round(float(X.iloc[i]["Mechanical_Power_W"]), 2),
                "temp_diff_k": round(float(X.iloc[i]["Temp_Difference_K"]), 2),
                "tool_wear_strain": round(float(X.iloc[i]["Tool_Wear_Strain"]), 2)
            }
            results.append(res)

        return results

    def _generate_recommendation(self, mode: str, p_fail: float, rul: float, row: pd.Series) -> Dict[str, str]:
        if mode == "TWF":
            return {
                "priority": "HIGH",
                "action": "Replace Cutting Tool Spindle",
                "detail": f"Tool wear is critical ({row.get('Tool_Wear_min', 'N/A')} min). High risk of micro-chipping or breakage. Stop machine at the next scheduled break and replace the cutting insert."
            }
        elif mode == "HDF":
            return {
                "priority": "HIGH",
                "action": "Inspect Cooling & Heat Dissipation System",
                "detail": "Process temperature differential is abnormally low while RPM is constrained. Check coolant flow rate, clean ventilation radiators, and verify heat exchanger pump operation."
            }
        elif mode == "PWF":
            return {
                "priority": "CRITICAL",
                "action": "Check Electrical Drive & Motor Couplings",
                "detail": "Electrical/mechanical power drawn is operating outside safe bounds (<3500 W or >9000 W). Check motor windings, variable frequency drive (VFD), and mechanical binding."
            }
        elif mode == "OSF":
            return {
                "priority": "CRITICAL",
                "action": "Reduce Operating Torque / Load",
                "detail": "Extreme mechanical overstrain (Torque × Tool Wear) detected. High risk of spindle shaft fatigue. Decrease feed rate immediately and replace the worn tool."
            }
        elif mode == "RNF":
            return {
                "priority": "MEDIUM",
                "action": "Sensor Diagnostics & Calibration",
                "detail": "Transient anomalies or sensor jitter detected. Verify accelerometer wiring, recalibrate thermocouple sensors, and monitor telemetry drift."
            }
        else:
            return {
                "priority": "LOW",
                "action": "Routine Monitoring",
                "detail": f"All telemetry indicators are nominal. Asset operating smoothly with estimated {rul:.0f} operating hours of useful life remaining."
            }
