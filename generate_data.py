# ============================================================
# FILE: generate_data.py
# PURPOSE: Simulate realistic BLDC fan motor sensor readings
# AUTHOR: Rohit Yadav | BIT Mesra
# ============================================================

import numpy as np
import pandas as pd

# Set random seed so results are reproducible every time
np.random.seed(42)

# ── Number of data points to generate ──
NUM_SAMPLES = 10000

print("Generating BLDC motor dataset...")

# ── 1. RPM (Fan Speed) ──
# Real BLDC fans operate between 200 RPM (lowest) to 1400 RPM (highest)
rpm = np.random.uniform(low=200, high=1400, size=NUM_SAMPLES)

# ── 2. Ambient Temperature (°C) ──
# Indian climate — ranges from 20°C to 45°C
temperature = np.random.uniform(low=20, high=45, size=NUM_SAMPLES)

# ── 3. Load Factor (0.5 to 1.5) ──
# Represents variable mechanical load on the motor
# 1.0 = normal load, <1.0 = light load, >1.0 = heavy load
load_factor = np.random.uniform(low=0.5, high=1.5, size=NUM_SAMPLES)

# ── 4. Voltage (V) ──
# Indian household voltage with natural fluctuations
voltage = np.random.normal(loc=230, scale=5, size=NUM_SAMPLES)

# ── 5. Current Draw (Amperes) ──
# Physics: Current increases with RPM and load
# Formula based on real BLDC motor characteristics
current = (
    0.002 * rpm           # RPM contribution
    + 0.05 * load_factor  # Load contribution
    + 0.001 * temperature # Temperature contribution
    + np.random.normal(0, 0.05, NUM_SAMPLES)  # Sensor noise
)

# ── 6. Power Consumption (Watts) ──
# Physics: Power = Voltage × Current × Power Factor
# BLDC motors have power factor ~0.95
power_factor = 0.95
power = voltage * current * power_factor

# ── 7. Efficiency (%) ──
# BLDC motors are most efficient at mid-range RPM
# Efficiency drops at very low and very high speeds
efficiency = (
    85
    - 0.01 * (rpm - 800)**2 / 1000   # Peak efficiency at 800 RPM
    + np.random.normal(0, 1, NUM_SAMPLES)  # Natural variation
)
efficiency = np.clip(efficiency, 60, 95)  # Keep between 60% and 95%

# ── 8. Motor Temperature (°C) ──
# Motor heats up with load and running time
motor_temp = (
    temperature
    + 0.02 * rpm
    + 5 * load_factor
    + np.random.normal(0, 1, NUM_SAMPLES)
)

# ── 9. Anomaly Flag ──
# ~3% of readings are anomalous (motor fault, overload, etc.)
anomaly = np.zeros(NUM_SAMPLES)
anomaly_indices = np.random.choice(NUM_SAMPLES, size=int(0.03 * NUM_SAMPLES), replace=False)
anomaly[anomaly_indices] = 1
# Anomalous readings have abnormally high power
power[anomaly_indices] *= np.random.uniform(1.5, 2.5, len(anomaly_indices))

# ── Build the DataFrame ──
df = pd.DataFrame({
    'rpm':          np.round(rpm, 1),
    'temperature':  np.round(temperature, 1),
    'load_factor':  np.round(load_factor, 3),
    'voltage':      np.round(voltage, 1),
    'current':      np.round(current, 3),
    'power_watts':  np.round(power, 2),
    'efficiency':   np.round(efficiency, 1),
    'motor_temp':   np.round(motor_temp, 1),
    'anomaly':      anomaly.astype(int)
})

# ── Save to CSV ──
df.to_csv('data/bldc_motor_data.csv', index=False)

print(f"✅ Dataset generated: {NUM_SAMPLES} samples")
print(f"✅ Saved to: data/bldc_motor_data.csv")
print(f"\n📊 Dataset Preview:")
print(df.head())
print(f"\n📈 Basic Stats:")
print(df.describe().round(2))
print(f"\n⚠️  Anomalies in dataset: {int(anomaly.sum())} samples ({anomaly.mean()*100:.1f}%)")