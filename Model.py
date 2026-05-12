# ============================================================
# FILE: model.py
# PURPOSE: EDA + Train ML model to predict power consumption
# AUTHOR: Rohit Yadav | BIT Mesra
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ── Create results folder to save plots ──
os.makedirs('results', exist_ok=True)

print("=" * 60)
print("  BLDC Motor Energy Analytics — Model Training")
print("=" * 60)

# ══════════════════════════════════════════════════
# STEP 1 — Load Dataset
# ══════════════════════════════════════════════════
print("\n📂 Loading dataset...")
df = pd.read_csv('data/bldc_motor_data.csv')
print(f"✅ Loaded {len(df)} samples with {len(df.columns)} features")

# Remove anomalies for clean model training
# (Anomalies are kept for dashboard detection later)
df_clean = df[df['anomaly'] == 0].copy()
print(f"✅ Clean samples (no anomalies): {len(df_clean)}")

# ══════════════════════════════════════════════════
# STEP 2 — Exploratory Data Analysis (EDA)
# ══════════════════════════════════════════════════
print("\n📊 Running Exploratory Data Analysis...")

# ── Plot 1: Power vs RPM ──
plt.figure(figsize=(10, 6))
plt.scatter(df_clean['rpm'], df_clean['power_watts'],
            alpha=0.3, c=df_clean['temperature'],
            cmap='RdYlGn_r', s=5)
plt.colorbar(label='Temperature (°C)')
plt.xlabel('Fan Speed (RPM)', fontsize=12)
plt.ylabel('Power Consumption (Watts)', fontsize=12)
plt.title('BLDC Fan: Power Consumption vs RPM\n(Color = Ambient Temperature)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/power_vs_rpm.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: results/power_vs_rpm.png")

# ── Plot 2: Correlation Heatmap ──
plt.figure(figsize=(10, 8))
corr_matrix = df_clean[['rpm', 'temperature', 'load_factor',
                          'current', 'power_watts', 'efficiency',
                          'motor_temp', 'voltage']].corr()
sns.heatmap(corr_matrix,
            annot=True, fmt='.2f',
            cmap='coolwarm', center=0,
            square=True, linewidths=0.5,
            annot_kws={'size': 10})
plt.title('Feature Correlation Heatmap\nBLDC Motor Parameters', fontsize=14)
plt.tight_layout()
plt.savefig('results/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: results/correlation_heatmap.png")

# ── Plot 3: Efficiency Distribution ──
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_clean['efficiency'], bins=30, color='steelblue', edgecolor='white')
plt.xlabel('Efficiency (%)')
plt.ylabel('Count')
plt.title('Motor Efficiency Distribution')

plt.subplot(1, 2, 2)
# Efficiency vs RPM
rpm_bins = pd.cut(df_clean['rpm'], bins=10)
eff_by_rpm = df_clean.groupby(rpm_bins, observed=True)['efficiency'].mean()
eff_by_rpm.plot(kind='bar', color='steelblue', edgecolor='white')
plt.xlabel('RPM Range')
plt.ylabel('Average Efficiency (%)')
plt.title('Avg Efficiency by RPM Range')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/efficiency_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: results/efficiency_analysis.png")

# ── Plot 4: Power Distribution ──
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_clean['power_watts'], bins=40,
             color='coral', edgecolor='white')
plt.xlabel('Power (Watts)')
plt.title('Power Consumption Distribution')

plt.subplot(1, 2, 2)
sns.boxplot(data=df_clean, x=pd.cut(df_clean['rpm'],
            bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']),
            y='power_watts', palette='RdYlGn_r')
plt.xlabel('RPM Category')
plt.ylabel('Power (Watts)')
plt.title('Power by Speed Category')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('results/power_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: results/power_distribution.png")

# ══════════════════════════════════════════════════
# STEP 3 — Prepare Features for ML Model
# ══════════════════════════════════════════════════
print("\n🤖 Preparing ML model...")

# Features (X) — inputs to the model
# Target (y) — what we want to predict
FEATURES = ['rpm', 'temperature', 'load_factor', 'voltage', 'current']
TARGET = 'power_watts'

X = df_clean[FEATURES]
y = df_clean[TARGET]

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Testing samples:  {len(X_test)}")

# ══════════════════════════════════════════════════
# STEP 4 — Train Random Forest Model
# ══════════════════════════════════════════════════
print("\n🌲 Training Random Forest Regressor...")
model = RandomForestRegressor(
    n_estimators=100,    # 100 decision trees
    max_depth=15,        # Each tree can be 15 levels deep
    random_state=42,
    n_jobs=-1            # Use all CPU cores
)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ══════════════════════════════════════════════════
# STEP 5 — Evaluate Model
# ══════════════════════════════════════════════════
y_pred = model.predict(X_test)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n" + "=" * 40)
print("  MODEL PERFORMANCE")
print("=" * 40)
print(f"  R² Score  : {r2:.4f}  ({r2*100:.1f}% accuracy)")
print(f"  MAE       : {mae:.2f} Watts")
print(f"  RMSE      : {rmse:.2f} Watts")
print("=" * 40)

# ── Plot 5: Actual vs Predicted ──
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.3, s=5, color='steelblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect prediction')
plt.xlabel('Actual Power (Watts)', fontsize=12)
plt.ylabel('Predicted Power (Watts)', fontsize=12)
plt.title(f'Actual vs Predicted Power\nR² = {r2:.4f} | MAE = {mae:.2f}W', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: results/actual_vs_predicted.png")

# ── Plot 6: Feature Importance ──
importances = pd.Series(model.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='steelblue', edgecolor='white')
plt.xlabel('Feature Importance Score', fontsize=12)
plt.title('Which Features Predict Power Best?\n(Random Forest Feature Importance)', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: results/feature_importance.png")

# ══════════════════════════════════════════════════
# STEP 6 — Save Model
# ══════════════════════════════════════════════════
with open('data/bldc_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\n✅ Model saved: data/bldc_model.pkl")

# Save feature names for dashboard
with open('data/features.pkl', 'wb') as f:
    pickle.dump(FEATURES, f)
print("✅ Features saved: data/features.pkl")

# ══════════════════════════════════════════════════
# STEP 7 — Energy Savings Analysis
# ══════════════════════════════════════════════════
print("\n⚡ Energy Savings Analysis:")
print("-" * 40)

# Compare high speed vs optimised speed
high_rpm    = df_clean[df_clean['rpm'] > 1100]['power_watts'].mean()
optimal_rpm = df_clean[(df_clean['rpm'] > 700) &
                        (df_clean['rpm'] < 900)]['power_watts'].mean()
savings_pct = ((high_rpm - optimal_rpm) / high_rpm) * 100

print(f"  Avg power at HIGH speed (>1100 RPM) : {high_rpm:.1f}W")
print(f"  Avg power at OPTIMAL speed (700-900): {optimal_rpm:.1f}W")
print(f"  Potential energy savings             : {savings_pct:.1f}%")
print("-" * 40)
print("\n✅ All done! Run app.py next to launch the dashboard.")