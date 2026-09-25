"""
data_generator.py
Synthetic and benchmark industrial dataset generator for Predictive Maintenance.
Simulates realistic physical phenomena based on the AI4I 2020 Predictive Maintenance benchmark:
- Heat Dissipation Failure (HDF)
- Power Failure (PWF)
- Overstrain Failure (OSF)
- Tool Wear Failure (TWF)
- Random / Sensor Failure (RNF)
"""

import numpy as np
import pandas as pd
import os

def generate_predictive_maintenance_data(num_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # 1. Product Type & Variants
    # L (Low quality variant): 50%, M (Medium): 30%, H (High): 20%
    types = np.random.choice(["L", "M", "H"], size=num_samples, p=[0.50, 0.30, 0.20])
    
    # Sequential UDI and Product ID
    udis = np.arange(1, num_samples + 1)
    product_ids = [f"{t}{10000 + i}" for i, t in enumerate(types)]
    
    # 2. Environmental & Operational Sensors
    # Air temperature [K]: mean 300K (~27 C), std 2K
    air_temp = np.random.normal(loc=300.0, scale=2.0, size=num_samples)
    
    # Process temperature [K]: mean ~ Air Temp + 10K, std 1K
    process_temp = air_temp + 10.0 + np.random.normal(loc=0.0, scale=1.0, size=num_samples)
    
    # Rotational speed [rpm]: centered around 1538 rpm
    rotational_speed = np.random.normal(loc=1538.0, scale=179.0, size=num_samples)
    rotational_speed = np.clip(rotational_speed, 1100.0, 2900.0)
    
    # Torque [Nm]: normally distributed around 40 Nm, std 9.9 Nm (inversely coupled with RPM)
    # Physically, Torque ~ Power / RPM + noise
    base_torque = 60000.0 / rotational_speed
    torque = base_torque + np.random.normal(loc=0.0, scale=7.0, size=num_samples)
    torque = np.clip(torque, 3.8, 76.6)
    
    # Tool wear [min]: wear accumulated on the tool, max ~250 min
    tool_wear = np.random.uniform(0.0, 250.0, size=num_samples)
    
    # Vibration [mm/s]: typical motor vibration baseline 1.5 - 3.5 mm/s
    vibration = np.random.normal(loc=2.5, scale=0.6, size=num_samples)
    vibration = np.clip(vibration, 0.5, 6.5)
    
    # 3. Physics-Based Failure Mode Simulation
    # Physical Power in Watts: Torque [Nm] * Rotational Speed [rad/s]
    # rad/s = rpm * (2 * pi / 60)
    power_watts = torque * (rotational_speed * (2.0 * np.pi / 60.0))
    temp_diff = process_temp - air_temp
    
    # Failure condition 1: Tool Wear Failure (TWF)
    # Tool breaks when wear is between 200 and 240 mins with increasing probability
    twf_prob = np.where(tool_wear >= 200, (tool_wear - 200) / 60.0, 0.0)
    twf = (np.random.uniform(0.0, 1.0, size=num_samples) < twf_prob).astype(int)
    
    # Failure condition 2: Heat Dissipation Failure (HDF)
    # Occurs if temperature difference < 8.6 K and rotational speed < 1380 rpm
    hdf = ((temp_diff < 8.6) & (rotational_speed < 1380.0)).astype(int)
    
    # Failure condition 3: Power Failure (PWF)
    # Occurs when power drawn is outside safe operating envelope (< 3500 W or > 9000 W)
    pwf = ((power_watts < 3500.0) | (power_watts > 9000.0)).astype(int)
    
    # Failure condition 4: Overstrain Failure (OSF)
    # Occurs when tool wear * torque exceeds threshold (dependent on machine variant)
    strain = tool_wear * torque
    osf = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        thresh = 11000 if types[i] == "L" else (12000 if types[i] == "M" else 13000)
        if strain[i] > thresh:
            osf[i] = 1
            
    # Failure condition 5: Random / Unknown Failure (RNF)
    rnf = (np.random.uniform(0.0, 1.0, size=num_samples) < 0.001).astype(int)
    
    # Composite Machine Failure Target
    machine_failure = ((twf == 1) | (hdf == 1) | (pwf == 1) | (osf == 1) | (rnf == 1)).astype(int)
    
    # Multiclass Failure Mode Label
    failure_type = []
    for i in range(num_samples):
        if machine_failure[i] == 0:
            failure_type.append("No Failure")
        elif twf[i] == 1:
            failure_type.append("TWF")
        elif hdf[i] == 1:
            failure_type.append("HDF")
        elif pwf[i] == 1:
            failure_type.append("PWF")
        elif osf[i] == 1:
            failure_type.append("OSF")
        else:
            failure_type.append("RNF")
            
    # Synthetic Remaining Useful Life (RUL) in operational hours
    # Healthy machines have 100-300 hours; approaching failure scales down towards 0-20 hours
    base_rul = 300.0 - (tool_wear * 1.0)
    # Apply penalties for stress factors
    strain_penalty = (strain / 13000.0) * 80.0
    temp_penalty = np.where(temp_diff < 9.0, (9.0 - temp_diff) * 20.0, 0.0)
    power_penalty = np.where(power_watts > 8000, (power_watts - 8000) / 100.0, 0.0)
    
    rul = base_rul - strain_penalty - temp_penalty - power_penalty
    rul = np.where(machine_failure == 1, np.random.uniform(0.0, 15.0, size=num_samples), rul)
    rul = np.clip(rul, 1.0, 350.0).round(1)
    
    # Elevate vibration on impending failure
    vibration = np.where(machine_failure == 1, vibration + np.random.uniform(1.5, 3.5, size=num_samples), vibration)
    vibration = vibration.round(2)
    
    df = pd.DataFrame({
        "UDI": udis,
        "Product_ID": product_ids,
        "Type": types,
        "Air_Temperature_K": air_temp.round(2),
        "Process_Temperature_K": process_temp.round(2),
        "Rotational_Speed_RPM": rotational_speed.round(1),
        "Torque_Nm": torque.round(2),
        "Tool_Wear_min": tool_wear.round(1),
        "Vibration_mms": vibration,
        "TWF": twf,
        "HDF": hdf,
        "PWF": pwf,
        "OSF": osf,
        "RNF": rnf,
        "Machine_Failure": machine_failure,
        "Failure_Type": failure_type,
        "RUL_Hours": rul
    })
    
    return df

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    df = generate_predictive_maintenance_data(num_samples=10000)
    csv_path = os.path.join(data_dir, "predictive_maintenance_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} samples saved to {csv_path}")
    print("Failure Distribution:")
    print(df["Failure_Type"].value_counts())
