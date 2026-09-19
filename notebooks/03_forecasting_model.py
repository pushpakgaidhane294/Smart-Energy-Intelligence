"""
Forecasting Model for Steel Industry Energy Consumption
This notebook trains a Random Forest model to predict energy consumption.
"""

import pandas as pd
import numpy as np
import json
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define paths using pathlib for cross-platform compatibility
BASE_DIR = Path(__file__).parent.parent
PROCESSED_PATH = BASE_DIR / 'data' / 'processed' / 'energy_processed.csv'
MODEL_PATH = BASE_DIR / 'models' / 'forecasting_model.pkl'
FEATURES_PATH = BASE_DIR / 'models' / 'forecast_features.json'
METRICS_PATH = BASE_DIR / 'models' / 'model_metrics.json'
REPORTS_PATH = BASE_DIR / 'reports'

print("="*60)
print("FORECASTING MODEL - RANDOM FOREST REGRESSION")
print("="*60)

# Load processed dataset
print(f"\nLoading processed dataset from: {PROCESSED_PATH}")
df = pd.read_csv(PROCESSED_PATH)
print(f"Dataset shape: {df.shape}")

# Convert datetime back to datetime type
df['datetime'] = pd.to_datetime(df['datetime'])

# Define target and features
target_column = 'Usage_kWh'
print(f"\nFORECASTING TARGET: {target_column}")

# Select feature columns (exclude target and non-numeric columns that shouldn't be used)
feature_columns = [
    'hour', 'day', 'month', 'day_of_week_num', 'weekend',
    'lag_1', 'lag_4', 'lag_96',
    'WeekStatus_encoded', 'Day_of_week_encoded',
    'Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh',
    'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM',
    'Load_Type_Light_Load', 'Load_Type_Medium_Load', 'Load_Type_Maximum_Load'
]

# Filter to only existing columns
feature_columns = [col for col in feature_columns if col in df.columns]
print(f"Feature columns ({len(feature_columns)}): {feature_columns}")

# Prepare X and y
X = df[feature_columns]
y = df[target_column]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Chronological train/test split (80% train, 20% test)
print("\n" + "="*60)
print("CHRONOLOGICAL TRAIN/TEST SPLIT")
print("="*60)

split_point = int(len(df) * 0.8)
X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]
y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Training time range: {df.iloc[:split_point]['datetime'].min()} to {df.iloc[:split_point]['datetime'].max()}")
print(f"Testing time range: {df.iloc[split_point:]['datetime'].min()} to {df.iloc[split_point:]['datetime'].max()}")

# Train Random Forest model
print("\n" + "="*60)
print("TRAINING RANDOM FOREST MODEL")
print("="*60)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_depth=20
)

print("Training model...")
model.fit(X_train, y_train)
print("Model training completed!")

# Make predictions
print("\nMaking predictions...")
y_pred = model.predict(X_test)

# Evaluate model
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f} kWh")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} kWh")
print(f"R² Score: {r2:.4f}")

# Feature importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print(feature_importance.head(10))

# Save model
print("\n" + "="*60)
print("SAVING MODEL AND ARTIFACTS")
print("="*60)

# Create models directory if it doesn't exist
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save the model
joblib.dump(model, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

# Save feature information
feature_info = {
    'target': target_column,
    'features': feature_columns,
    'feature_importance': feature_importance.to_dict('records')
}

with open(FEATURES_PATH, 'w') as f:
    json.dump(feature_info, f, indent=2)
print(f"Feature information saved to: {FEATURES_PATH}")

# Save metrics
metrics = {
    'model': 'RandomForestRegressor',
    'target': target_column,
    'mae': float(mae),
    'rmse': float(rmse),
    'r2': float(r2),
    'training_rows': int(len(X_train)),
    'testing_rows': int(len(X_test))
}

with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to: {METRICS_PATH}")

# Create visualization
print("\n" + "="*60)
print("CREATING VISUALIZATION")
print("="*60)

# Create reports directory if it doesn't exist
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

# Plot actual vs predicted (subset for readability)
plot_size = min(500, len(y_test))
fig, ax = plt.subplots(figsize=(12, 6))

# Plot actual values
ax.plot(range(plot_size), y_test.iloc[:plot_size].values, 
        label='Actual', alpha=0.7, linewidth=1)
# Plot predicted values
ax.plot(range(plot_size), y_pred[:plot_size], 
        label='Predicted', alpha=0.7, linewidth=1)

ax.set_xlabel('Time Steps (Test Set)')
ax.set_ylabel('Energy Consumption (kWh)')
ax.set_title('Actual vs Predicted Energy Consumption')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save visualization
viz_path = REPORTS_PATH / 'actual_vs_predicted.png'
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {viz_path}")
plt.close()

# Print summary
print("\n" + "="*60)
print("FORECASTING MODEL SUMMARY")
print("="*60)
print(f"Target column: {target_column}")
print(f"Feature columns: {len(feature_columns)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"MAE: {mae:.4f} kWh")
print(f"RMSE: {rmse:.4f} kWh")
print(f"R² Score: {r2:.4f}")
print(f"\nModel saved: {MODEL_PATH}")
print(f"Features saved: {FEATURES_PATH}")
print(f"Metrics saved: {METRICS_PATH}")
print(f"Visualization saved: {viz_path}")

print("\n" + "="*60)
print("FORECASTING MODEL TRAINING COMPLETE")
print("="*60)
