"""
features.py
Feature engineering transformers and helpers for predictive maintenance.
Computes domain-specific physical metrics:
- Temperature Differential
- Mechanical Power
- Overstrain interaction
- Thermal Stress Index
- Torque-to-Speed ratios
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

FEATURE_COLUMNS = [
    "Air_Temperature_K",
    "Process_Temperature_K",
    "Rotational_Speed_RPM",
    "Torque_Nm",
    "Tool_Wear_min",
    "Vibration_mms",
    "Temp_Difference_K",
    "Mechanical_Power_W",
    "Tool_Wear_Strain",
    "Power_To_Speed_Ratio",
    "Thermal_Stress_Index",
    "Type_M",
    "Type_H"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw dataframe or dictionary of telemetry, computes domain-specific features.
    """
    df = df.copy()
    
    # 1. Temperature Differential
    df["Temp_Difference_K"] = df["Process_Temperature_K"] - df["Air_Temperature_K"]
    
    # 2. Mechanical Power (W) = Torque (Nm) * (RPM * 2 * pi / 60)
    angular_velocity = df["Rotational_Speed_RPM"] * (2.0 * np.pi / 60.0)
    df["Mechanical_Power_W"] = df["Torque_Nm"] * angular_velocity
    
    # 3. Tool Wear Strain = Tool Wear * Torque
    df["Tool_Wear_Strain"] = df["Tool_Wear_min"] * df["Torque_Nm"]
    
    # 4. Power to Speed Ratio
    df["Power_To_Speed_Ratio"] = df["Mechanical_Power_W"] / (df["Rotational_Speed_RPM"] + 1e-5)
    
    # 5. Thermal Stress Index
    df["Thermal_Stress_Index"] = df["Process_Temperature_K"] / (df["Air_Temperature_K"] + 1e-5)
    
    # 6. Type one-hot encoding
    if "Type" in df.columns:
        df["Type_M"] = (df["Type"] == "M").astype(float)
        df["Type_H"] = (df["Type"] == "H").astype(float)
    else:
        # Default fallback if Type not provided
        if "Type_M" not in df.columns:
            df["Type_M"] = 0.0
        if "Type_H" not in df.columns:
            df["Type_H"] = 0.0
            
    # Ensure all required feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
            
    return df[FEATURE_COLUMNS]

def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Prepares X and target variables:
    - y_failure: Binary indicator (0/1)
    - y_failure_type: Multiclass label (No Failure, TWF, HDF, PWF, OSF, RNF)
    - y_rul: Remaining useful life in hours
    """
    X = engineer_features(df)
    y_failure = df["Machine_Failure"]
    y_failure_type = df["Failure_Type"]
    y_rul = df["RUL_Hours"]
    return X, y_failure, y_failure_type, y_rul
