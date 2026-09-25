"""
app.py
Interactive Streamlit Dashboard for Industrial Machinery Predictive Maintenance & Digital Twin.
Features:
- Live Digital Twin Inspector with interactive sliders and real-time physical simulation.
- Real-time Health Index, Risk Probability, Diagnosed Failure Mode, and RUL estimation.
- Prescriptive Action Engine for field technicians.
- Factory Fleet Health overview with scatter distributions.
- Explainable AI metrics and feature importances.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json

from predict import PredictiveMaintenanceEngine
from data_generator import generate_predictive_maintenance_data

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2e3646;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #9aa0a6;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
    }
    .badge-optimal {
        background-color: #0e4429;
        color: #3fb950;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-warning {
        background-color: #4d2d00;
        color: #d29922;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-critical {
        background-color: #490202;
        color: #f85149;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_engine():
    return PredictiveMaintenanceEngine()

@st.cache_data
def get_fleet_sample():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "predictive_maintenance_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = generate_predictive_maintenance_data(1000)
    return df

engine = get_engine()
fleet_df = get_fleet_sample()

# Header
st.title("⚙️ Predictive Maintenance for Industrial Machinery")
st.caption("Physics-Informed Condition Monitoring • Early Failure Detection • Remaining Useful Life (RUL) Prediction")

# Tabs
tab_inspector, tab_fleet, tab_analytics = st.tabs([
    "🔍 Live Machine Inspector & Digital Twin",
    "🏭 Factory Fleet Health Overview",
    "📊 Model Analytics & Explainability"
])

# ==========================================
# TAB 1: Live Machine Inspector & Digital Twin
# ==========================================
with tab_inspector:
    st.subheader("Asset Telemetry & Real-Time Diagnostics")
    
    col_sim_ctrl, col_sim_view = st.columns([1, 2], gap="large")
    
    with col_sim_ctrl:
        st.markdown("### 🎛️ Sensor Input Telemetry")
        
        # Scenario Quick Presets
        preset = st.selectbox(
            "Load Operating Scenario Preset:",
            [
                "Custom Configuration",
                "Healthy Machine Baseline",
                "Tool Wear Failure (TWF)",
                "Heat Dissipation Breakdown (HDF)",
                "Mechanical Overstrain (OSF)",
                "Power Inverter Anomaly (PWF)"
            ]
        )
        
        # Set default values based on preset
        if preset == "Healthy Machine Baseline":
            p_type, p_air, p_proc, p_rpm, p_torque, p_wear, p_vib = "L", 298.5, 308.7, 1520.0, 39.5, 30.0, 1.9
        elif preset == "Tool Wear Failure (TWF)":
            p_type, p_air, p_proc, p_rpm, p_torque, p_wear, p_vib = "L", 300.0, 310.2, 1480.0, 42.0, 238.0, 5.4
        elif preset == "Heat Dissipation Breakdown (HDF)":
            p_type, p_air, p_proc, p_rpm, p_torque, p_wear, p_vib = "L", 301.0, 306.5, 1280.0, 52.0, 85.0, 4.2
        elif preset == "Mechanical Overstrain (OSF)":
            p_type, p_air, p_proc, p_rpm, p_torque, p_wear, p_vib = "L", 299.5, 309.8, 1390.0, 68.0, 215.0, 5.6
        elif preset == "Power Inverter Anomaly (PWF)":
            p_type, p_air, p_proc, p_rpm, p_torque, p_wear, p_vib = "M", 300.2, 310.5, 2750.0, 65.0, 110.0, 4.9
        else:
            p_type, p_air, p_proc, p_rpm, p_torque, p_wear, p_vib = "L", 300.0, 310.0, 1500.0, 40.0, 50.0, 2.3

        var_type = st.radio("Machine Quality Variant:", ["L (Low)", "M (Medium)", "H (High)"], 
                            index=0 if p_type == "L" else (1 if p_type == "M" else 2), horizontal=True)
        type_code = var_type[0]
        
        air_temp = st.slider("Air Temperature [K]", min_value=290.0, max_value=310.0, value=float(p_air), step=0.1,
                             help="Ambient environmental temperature (300K ≈ 27°C)")
        
        proc_temp = st.slider("Process Temperature [K]", min_value=295.0, max_value=320.0, value=float(p_proc), step=0.1,
                              help="Internal operational temperature of the spindle and fluid")
        
        rpm = st.slider("Rotational Speed [RPM]", min_value=1000.0, max_value=3000.0, value=float(p_rpm), step=10.0,
                        help="Motor shaft rotational speed")
        
        torque = st.slider("Torque [Nm]", min_value=3.0, max_value=85.0, value=float(p_torque), step=0.5,
                           help="Mechanical resistance and torque load")
        
        tool_wear = st.slider("Tool Wear [min]", min_value=0.0, max_value=260.0, value=float(p_wear), step=1.0,
                              help="Accumulated operating run-time on current tool")
        
        vibration = st.slider("Vibration [mm/s]", min_value=0.5, max_value=8.0, value=float(p_vib), step=0.1,
                              help="Root-mean-square machine vibration amplitude")

    # Run Real-Time Prediction
    telemetry_input = {
        "Type": type_code,
        "Air_Temperature_K": air_temp,
        "Process_Temperature_K": proc_temp,
        "Rotational_Speed_RPM": rpm,
        "Torque_Nm": torque,
        "Tool_Wear_min": tool_wear,
        "Vibration_mms": vibration
    }
    
    pred_res = engine.predict_single(telemetry_input)

    with col_sim_view:
        st.markdown("### 📊 Asset Health Diagnostics")
        
        # Top Row Status Indicator
        status = pred_res["status_category"]
        if status == "OPTIMAL":
            st.success(f"🟢 **STATUS: OPTIMAL OPERATION** • Health Index: {pred_res['health_index']}%")
        elif status == "WARNING":
            st.warning(f"🟡 **STATUS: WARNING / ELEVATED RISK** • Health Index: {pred_res['health_index']}%")
        else:
            st.error(f"🔴 **STATUS: CRITICAL IMMINENT FAILURE** • Health Index: {pred_res['health_index']}%")
            
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Health Index", f"{pred_res['health_index']}%")
        m2.metric("Failure Probability", f"{(pred_res['failure_probability'] * 100):.1f}%")
        m3.metric("Diagnosed Mode", pred_res['failure_mode'])
        m4.metric("Est. Remaining Life", f"{pred_res['rul_hours']:.1f} hrs")
        
        # Health Gauge & Derived Physics
        g_col, p_col = st.columns([1, 1])
        with g_col:
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_res["health_index"],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Composite Machine Health Score", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#00bc8c" if status=="OPTIMAL" else ("#f39c12" if status=="WARNING" else "#e74c3c")},
                    'steps': [
                        {'range': [0, 45], 'color': "rgba(231, 76, 60, 0.2)"},
                        {'range': [45, 75], 'color': "rgba(243, 156, 18, 0.2)"},
                        {'range': [75, 100], 'color': "rgba(0, 188, 140, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 3},
                        'thickness': 0.75,
                        'value': 45
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with p_col:
            st.markdown("#### ⚡ Physics & Boundary Envelope")
            st.markdown(f"**Mechanical Power ($P$):** `{pred_res['computed_power_w']:,} Watts` (Safe: 3,500W – 9,000W)")
            st.markdown(f"**Temperature Delta ($\Delta T$):** `{pred_res['temp_diff_k']} K` (Target: > 8.6 K)")
            st.markdown(f"**Tool Wear Strain ($\tau \times t$):** `{pred_res['tool_wear_strain']:,} min·Nm` (Limit: < 11,000)")
            st.markdown(f"**Vibration Amplitude:** `{vibration} mm/s` (Nominal: < 3.5 mm/s)")
            
            # Progress bar for RUL
            rul_val = min(1.0, max(0.0, pred_res['rul_hours'] / 300.0))
            st.write(f"RUL Remaining: **{pred_res['rul_hours']} / 300.0 hrs**")
            st.progress(rul_val)

        # Prescriptive Recommendation Card
        rec = pred_res["recommendation"]
        priority_color = "red" if rec["priority"] == "CRITICAL" else ("orange" if rec["priority"] == "HIGH" else "green")
        st.markdown(f"""
        <div style="background-color: #1a1e29; border-left: 5px solid {priority_color}; padding: 15px; border-radius: 6px; margin-top: 10px;">
            <h4 style="margin: 0 0 5px 0; color: #ffffff;">📋 Technician Prescriptive Action: {rec['action']}</h4>
            <p style="margin: 0; color: #b0b8c4; font-size: 0.95rem;">{rec['detail']}</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 2: Factory Fleet Health Overview
# ==========================================
with tab_fleet:
    st.subheader("Factory Floor Fleet Health Monitoring")
    
    # Process batch predictions on sample fleet
    fleet_sample = fleet_df.head(100).copy()
    batch_results = engine.predict_dataframe(fleet_sample)
    
    fleet_sample["Predicted_Health"] = [r["health_index"] for r in batch_results]
    fleet_sample["Failure_Risk"] = [f"{r['failure_probability']*100:.1f}%" for r in batch_results]
    fleet_sample["Predicted_Status"] = [r["status_category"] for r in batch_results]
    fleet_sample["Diagnosed_Mode"] = [r["failure_mode"] for r in batch_results]
    fleet_sample["Estimated_RUL"] = [r["rul_hours"] for r in batch_results]
    
    # Fleet Summary KPIs
    tot_machines = len(fleet_sample)
    optimal_count = (fleet_sample["Predicted_Status"] == "OPTIMAL").sum()
    warning_count = (fleet_sample["Predicted_Status"] == "WARNING").sum()
    critical_count = (fleet_sample["Predicted_Status"] == "CRITICAL").sum()
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Monitored Assets", tot_machines)
    k2.metric("Optimal Assets", optimal_count)
    k3.metric("Warning Assets", warning_count)
    k4.metric("Critical / Action Required", critical_count)
    
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        # Scatter: Rotational Speed vs Torque
        fig_scatter = px.scatter(
            fleet_sample,
            x="Rotational_Speed_RPM",
            y="Torque_Nm",
            color="Predicted_Status",
            color_discrete_map={"OPTIMAL": "#00bc8c", "WARNING": "#f39c12", "CRITICAL": "#e74c3c"},
            hover_data=["Product_ID", "Predicted_Health", "Diagnosed_Mode", "Estimated_RUL"],
            title="Operating Envelope: Rotational Speed vs Torque"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with c_chart2:
        # Failure Mode breakdown
        mode_counts = fleet_sample[fleet_sample["Diagnosed_Mode"] != "No Failure"]["Diagnosed_Mode"].value_counts().reset_index()
        if not mode_counts.empty:
            fig_bar = px.bar(
                mode_counts,
                x="Diagnosed_Mode",
                y="count",
                title="Active Failure Mode Distribution in Monitored Fleet",
                color="Diagnosed_Mode",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("All machines in this batch currently report healthy operational status.")
            
    st.markdown("#### Monitored Machines Live Roster")
    st.dataframe(
        fleet_sample[[
            "Product_ID", "Type", "Predicted_Status", "Predicted_Health", 
            "Diagnosed_Mode", "Estimated_RUL", "Rotational_Speed_RPM", "Torque_Nm", "Tool_Wear_min"
        ]],
        use_container_width=True
    )

# ==========================================
# TAB 3: Model Analytics & Explainability
# ==========================================
with tab_analytics:
    st.subheader("Explainable Machine Learning & Model Performance")
    
    metrics = engine.metrics
    
    if metrics:
        col_m1, col_m2 = st.columns([1, 1])
        
        with col_m1:
            st.markdown("### 🏆 Model Evaluation Metrics")
            
            b_metrics = metrics.get("binary_classification", {})
            r_metrics = metrics.get("rul_regression", {})
            m_metrics = metrics.get("multiclass_diagnosis", {})
            
            st.markdown(f"""
            - **Binary Failure Classifier ROC-AUC:** `{b_metrics.get('roc_auc', 'N/A')}`
            - **Binary Failure F1-Score:** `{b_metrics.get('f1_score', 'N/A')}`
            - **Failure Mode Macro F1:** `{m_metrics.get('macro_f1', 'N/A')}`
            - **RUL Regressor R² Score:** `{r_metrics.get('r2_score', 'N/A')}`
            - **RUL Mean Absolute Error (MAE):** `{r_metrics.get('mae_hours', 'N/A')} hours`
            """)
            
            # Confusion matrix
            cm = b_metrics.get("confusion_matrix", None)
            if cm:
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Healthy", "Failure"],
                    y=["Healthy", "Failure"],
                    title="Binary Failure Prediction Confusion Matrix",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                
        with col_m2:
            st.markdown("### 🔍 Global Feature Importances")
            feat_imp = metrics.get("feature_importances", {})
            if feat_imp:
                imp_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
                fig_imp = px.bar(
                    imp_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Feature Contribution to Failure Risk",
                    color="Importance",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Train the model via `python model.py` to view full evaluation metrics.")
