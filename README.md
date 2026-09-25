# Predictive Maintenance for Industrial Machinery

An end-to-end Machine Learning and Physics-Informed Predictive Maintenance (PdM) & Digital Twin system designed for industrial manufacturing assets (CNC milling machines, motors, turbines, compressors, and pumps).

---

## 🌟 Overview & Key Concepts

Predictive Maintenance moves away from reactive ("run-to-failure") and rigid preventative schedules by continuously monitoring equipment telemetry to detect degradation before breakdown occurs.

### What Features Does It Have?

1. **Input Sensor Telemetry (What We Measure)**:
   - **Air Temperature [K]**: Ambient environment temperature.
   - **Process Temperature [K]**: Internal machine fluid/spindle temperature.
   - **Rotational Speed [RPM]**: Motor spindle revolutions per minute.
   - **Torque [Nm]**: Mechanical resistance load on the shaft.
   - **Tool Wear [min]**: Cumulative run-time minutes on the cutting tool.
   - **Vibration [mm/s]**: RMS mechanical vibration velocity.

2. **Domain-Engineered Physical Features**:
   - **Temperature Differential ($\Delta T = T_{process} - T_{air}$)**: Heat dissipation efficacy.
   - **Mechanical Power ($P = \tau \times \omega = \tau \times \frac{2\pi \cdot \text{RPM}}{60}$)**: Mechanical power consumption in Watts.
   - **Tool Wear Strain ($\tau \times \text{ToolWear}$)**: Cumulative mechanical overstrain under cutting resistance.
   - **Power-to-Speed Ratio**: Electrical efficiency indicator.

3. **Multi-Task Machine Learning Engine**:
   - **Binary Failure Classifier**: Predicts failure probability and alerts operators when threshold is crossed.
   - **Failure Mode Diagnostic Classifier**: Classifies failure modes:
     - **TWF (Tool Wear Failure)**: Tool reaches physical wear limit.
     - **HDF (Heat Dissipation Failure)**: Ventilation or cooling breakdown.
     - **PWF (Power Failure)**: Power outside safe operating bounds (<3500W or >9000W).
     - **OSF (Overstrain Failure)**: Excessive product of torque and tool wear.
     - **RNF (Random/Sensor Failure)**: Uncorrelated anomalies or sensor jitter.
   - **Remaining Useful Life (RUL) Regressor**: Predicts remaining operational hours before scheduled downtime is required.
   - **Prescriptive Maintenance Engine**: Actionable guidance for shop floor technicians.

---

## 🚀 Quickstart

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Generate Industrial Benchmark Dataset
```bash
python data_generator.py
```
Outputs 10,000 realistic machine records with physical failure dynamics to `data/predictive_maintenance_data.csv`.

### 3. Train Machine Learning Models
```bash
python model.py
```
Trains binary failure classifier (ROC-AUC > 0.99, F1 > 0.96), multi-class failure mode classifier, and RUL regressor ($R^2 > 0.97$), exporting them to `models/`.

### 4. Run Automated Test Suite
```bash
python -m unittest tests/test_pipeline.py
```

### 5. Launch Interactive Streamlit Dashboard & Digital Twin
```bash
streamlit run app.py
```

---

## 🖥️ Dashboard Capabilities

- **Digital Twin & Live Asset Inspector**: Adjust telemetry sliders in real-time or pick from pre-configured fault injection presets (TWF, HDF, OSF, PWF).
- **Health Gauges & Physics Boundary Checks**: Live Plotly gauge, power envelope limits, and temperature gradient monitoring.
- **Prescriptive Action Card**: Step-by-step repair instructions for technicians based on detected root cause.
- **Factory Floor Fleet Overview**: Real-time fleet health roster, operational scatter plots, and fault distribution charts.
- **Explainable AI Analytics**: Global feature importance rankings and confusion matrix.
