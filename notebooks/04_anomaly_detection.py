"""
Anomaly Detection for Steel Industry Energy Consumption
This notebook uses Isolation Forest to detect unusual consumption patterns.
"""

import pandas as pd
import numpy as np
import json
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest

# Define paths using pathlib for cross-platform compatibility
BASE_DIR = Path(__file__).parent.parent
PROCESSED_PATH = BASE_DIR / 'data' / 'processed' / 'energy_processed.csv'
ANOMALY_RESULTS_PATH = BASE_DIR / 'data' / 'processed' / 'anomaly_results.csv'
ANOMALY_MODEL_PATH = BASE_DIR / 'models' / 'anomaly_model.pkl'
METRICS_PATH = BASE_DIR / 'models' / 'model_metrics.json'
REPORTS_PATH = BASE_DIR / 'reports'

print("="*60)
print("ANOMALY DETECTION - ISOLATION FOREST")
print("="*60)

# Load processed dataset
print(f"\nLoading processed dataset from: {PROCESSED_PATH}")
df = pd.read_csv(PROCESSED_PATH)
print(f"Dataset shape: {df.shape}")

# Convert datetime back to datetime type
df['datetime'] = pd.to_datetime(df['datetime'])

# Select numerical features for anomaly detection
print("\n" + "="*60)
print("FEATURE SELECTION FOR ANOMALY DETECTION")
print("="*60)

# Use energy-related numerical features
anomaly_features = [
    'Usage_kWh',
    'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh',
    'CO2(tCO2)',
    'Lagging_Current_Power_Factor',
    'Leading_Current_Power_Factor',
    'NSM'
]

# Filter to only existing columns
anomaly_features = [col for col in anomaly_features if col in df.columns]
print(f"Anomaly detection features ({len(anomaly_features)}): {anomaly_features}")

# Prepare feature matrix
X_anomaly = df[anomaly_features].copy()

# Handle any remaining missing values
X_anomaly = X_anomaly.fillna(X_anomaly.mean())

print(f"Feature matrix shape: {X_anomaly.shape}")

# Train Isolation Forest
print("\n" + "="*60)
print("TRAINING ISOLATION FOREST")
print("="*60)

anomaly_model = IsolationForest(
    contamination=0.05,
    random_state=42,
    n_estimators=100
)

print("Training Isolation Forest model...")
anomaly_model.fit(X_anomaly)
print("Model training completed!")

# Predict anomalies
print("\nDetecting anomalies...")
anomaly_predictions = anomaly_model.predict(X_anomaly)

# -1 indicates anomaly, 1 indicates normal
df['anomaly_prediction'] = anomaly_predictions
df['anomaly_label'] = df['anomaly_prediction'].map({1: 'Normal', -1: 'Anomaly'})

# Calculate anomaly statistics
print("\n" + "="*60)
print("ANOMALY STATISTICS")
print("="*60)

total_records = len(df)
normal_records = (df['anomaly_label'] == 'Normal').sum()
anomaly_records = (df['anomaly_label'] == 'Anomaly').sum()
anomaly_percentage = (anomaly_records / total_records) * 100

print(f"Total records: {total_records}")
print(f"Normal records: {normal_records}")
print(f"Anomaly records: {anomaly_records}")
print(f"Anomaly percentage: {anomaly_percentage:.2f}%")

# Save anomaly model
print("\n" + "="*60)
print("SAVING MODEL AND RESULTS")
print("="*60)

# Create models directory if it doesn't exist
ANOMALY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save the model
joblib.dump(anomaly_model, ANOMALY_MODEL_PATH)
print(f"Anomaly model saved to: {ANOMALY_MODEL_PATH}")

# Save anomaly results
df.to_csv(ANOMALY_RESULTS_PATH, index=False)
print(f"Anomaly results saved to: {ANOMALY_RESULTS_PATH}")

# Create visualization
print("\n" + "="*60)
print("CREATING VISUALIZATION")
print("="*60)

# Create reports directory if it doesn't exist
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

# Plot energy consumption with anomalies highlighted
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Time series with anomalies
# Sample for readability
sample_size = min(2000, len(df))
df_sample = df.iloc[:sample_size].copy()

normal_mask = df_sample['anomaly_label'] == 'Normal'
anomaly_mask = df_sample['anomaly_label'] == 'Anomaly'

ax1.plot(df_sample[normal_mask].index, df_sample[normal_mask]['Usage_kWh'], 
         'o', markersize=2, alpha=0.5, label='Normal', color='blue')
ax1.plot(df_sample[anomaly_mask].index, df_sample[anomaly_mask]['Usage_kWh'], 
         'o', markersize=4, alpha=0.8, label='Anomaly', color='red')

ax1.set_xlabel('Time Index')
ax1.set_ylabel('Energy Consumption (kWh)')
ax1.set_title('Energy Consumption with Anomalies Highlighted')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Distribution by anomaly status
normal_consumption = df[df['anomaly_label'] == 'Normal']['Usage_kWh']
anomaly_consumption = df[df['anomaly_label'] == 'Anomaly']['Usage_kWh']

ax2.hist(normal_consumption, bins=50, alpha=0.7, label='Normal', color='blue', edgecolor='black')
ax2.hist(anomaly_consumption, bins=50, alpha=0.7, label='Anomaly', color='red', edgecolor='black')
ax2.set_xlabel('Energy Consumption (kWh)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Energy Consumption by Anomaly Status')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save visualization
viz_path = REPORTS_PATH / 'anomaly_detection.png'
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {viz_path}")
plt.close()

# Print summary
print("\n" + "="*60)
print("ANOMALY DETECTION SUMMARY")
print("="*60)
print(f"Total records analyzed: {total_records}")
print(f"Normal records: {normal_records} ({100-anomaly_percentage:.2f}%)")
print(f"Anomaly records: {anomaly_records} ({anomaly_percentage:.2f}%)")
print(f"Anomaly detection method: Isolation Forest")
print(f"Contamination: 0.05 (5% configured anomaly rate)")
print(f"\nModel saved: {ANOMALY_MODEL_PATH}")
print(f"Results saved: {ANOMALY_RESULTS_PATH}")
print(f"Visualization saved: {viz_path}")

print("\n" + "="*60)
print("ANOMALY DETECTION COMPLETE")
print("="*60)
print("\nNote: The system identifies unusual consumption patterns.")
print("It does not claim specific causes for anomalies.")
print("Anomalies should be investigated by domain experts.")

# Update model metrics with anomaly information
print("\n" + "="*60)
print("UPDATING MODEL METRICS")
print("="*60)

try:
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
    
    metrics['anomaly_model'] = 'IsolationForest'
    metrics['anomaly_count'] = int(anomaly_records)
    metrics['anomaly_percentage'] = float(anomaly_percentage)
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model metrics updated with anomaly information")
    print(f"Updated metrics saved to: {METRICS_PATH}")
except FileNotFoundError:
    print("Warning: model_metrics.json not found. Creating new metrics file.")
    metrics = {
        'anomaly_model': 'IsolationForest',
        'anomaly_count': int(anomaly_records),
        'anomaly_percentage': float(anomaly_percentage)
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"New metrics file created: {METRICS_PATH}")
