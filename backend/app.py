"""
FastAPI Backend for Smart Energy Intelligence
This API serves energy consumption forecasts and anomaly detection results.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Any

# Initialize FastAPI app
app = FastAPI(title="Smart Energy Intelligence API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths using pathlib for cross-platform compatibility
BASE_DIR = Path(__file__).parent.parent
PROCESSED_PATH = BASE_DIR / 'data' / 'processed' / 'energy_processed.csv'
ANOMALY_RESULTS_PATH = BASE_DIR / 'data' / 'processed' / 'anomaly_results.csv'
FORECAST_MODEL_PATH = BASE_DIR / 'models' / 'forecasting_model.pkl'
ANOMALY_MODEL_PATH = BASE_DIR / 'models' / 'anomaly_model.pkl'
FEATURES_PATH = BASE_DIR / 'models' / 'forecast_features.json'
METRICS_PATH = BASE_DIR / 'models' / 'model_metrics.json'

# Global variables to store loaded models and data
forecast_model = None
anomaly_model = None
feature_info = None
model_metrics = None
processed_data = None
anomaly_data = None

def load_models_and_data():
    """Load all models and data at startup."""
    global forecast_model, anomaly_model, feature_info, model_metrics, processed_data, anomaly_data
    
    try:
        # Load forecasting model
        if FORECAST_MODEL_PATH.exists():
            forecast_model = joblib.load(FORECAST_MODEL_PATH)
            print("Forecasting model loaded successfully")
        else:
            print("Warning: Forecasting model not found")
        
        # Load anomaly model
        if ANOMALY_MODEL_PATH.exists():
            anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
            print("Anomaly model loaded successfully")
        else:
            print("Warning: Anomaly model not found")
        
        # Load feature information
        if FEATURES_PATH.exists():
            with open(FEATURES_PATH, 'r') as f:
                feature_info = json.load(f)
            print("Feature information loaded successfully")
        else:
            print("Warning: Feature information not found")
        
        # Load model metrics
        if METRICS_PATH.exists():
            with open(METRICS_PATH, 'r') as f:
                model_metrics = json.load(f)
            print("Model metrics loaded successfully")
        else:
            print("Warning: Model metrics not found")
        
        # Load processed data
        if PROCESSED_PATH.exists():
            processed_data = pd.read_csv(PROCESSED_PATH)
            processed_data['datetime'] = pd.to_datetime(processed_data['datetime'])
            print(f"Processed data loaded: {len(processed_data)} records")
        else:
            print("Warning: Processed data not found")
        
        # Load anomaly results
        if ANOMALY_RESULTS_PATH.exists():
            anomaly_data = pd.read_csv(ANOMALY_RESULTS_PATH)
            anomaly_data['datetime'] = pd.to_datetime(anomaly_data['datetime'])
            print(f"Anomaly results loaded: {len(anomaly_data)} records")
        else:
            print("Warning: Anomaly results not found")
            
    except Exception as e:
        print(f"Error loading models and data: {e}")

# Load models and data on startup
@app.on_event("startup")
def startup_event():
    load_models_and_data()

# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint to check API status."""
    return {
        "status": "online",
        "project": "Smart Energy Intelligence",
        "description": "Industrial Electricity Demand Forecasting & Anomaly Detection System"
    }

# Summary endpoint
@app.get("/api/summary")
def get_summary():
    """Get summary statistics and the latest model estimate."""
    if processed_data is None:
        raise HTTPException(status_code=503, detail="Processed data not available")
    
    try:
        target_column = 'Usage_kWh'
        
        # Calculate summary statistics
        total_records = len(processed_data)
        average_consumption = float(processed_data[target_column].mean())
        minimum_consumption = float(processed_data[target_column].min())
        maximum_consumption = float(processed_data[target_column].max())
        
        # Estimate demand from the latest available feature vector.
        predicted_demand = None
        if forecast_model is not None and feature_info is not None:
            feature_columns = feature_info.get('features', [])
            missing_features = [
                column for column in feature_columns
                if column not in processed_data.columns
            ]
            if not missing_features:
                latest_features = processed_data[feature_columns].iloc[[-1]]
                predicted_demand = float(forecast_model.predict(latest_features)[0])

        # Keep the summary available if the model artifact is unavailable.
        if predicted_demand is None:
            predicted_demand = average_consumption
        
        # Get anomaly statistics
        if anomaly_data is not None:
            anomaly_count = int((anomaly_data['anomaly_label'] == 'Anomaly').sum())
            anomaly_percentage = float((anomaly_count / len(anomaly_data)) * 100)
        else:
            anomaly_count = 0
            anomaly_percentage = 0.0
        
        return {
            "total_records": total_records,
            "average_consumption": round(average_consumption, 2),
            "minimum_consumption": round(minimum_consumption, 2),
            "maximum_consumption": round(maximum_consumption, 2),
            "predicted_demand": round(predicted_demand, 2),
            "anomaly_count": anomaly_count,
            "anomaly_percentage": round(anomaly_percentage, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating summary: {str(e)}")

# Consumption endpoint
@app.get("/api/consumption")
def get_consumption(limit: int = 1000):
    """Get historical consumption data."""
    if processed_data is None:
        raise HTTPException(status_code=503, detail="Processed data not available")
    
    try:
        # Limit the number of records for performance
        data_subset = processed_data.head(limit)
        
        # Prepare response data
        consumption_data = []
        for _, row in data_subset.iterrows():
            consumption_data.append({
                "datetime": row['datetime'].isoformat(),
                "usage_kwh": float(row['Usage_kWh']),
                "load_type": str(row.get('Load_Type', 'Unknown'))
            })
        
        return {
            "count": len(consumption_data),
            "data": consumption_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching consumption data: {str(e)}")

# Forecast endpoint
@app.get("/api/forecast")
def get_forecast():
    """Get forecast information."""
    if model_metrics is None:
        raise HTTPException(status_code=503, detail="Model metrics not available")
    
    try:
        return {
            "model": model_metrics.get('model', 'Unknown'),
            "target": model_metrics.get('target', 'Unknown'),
            "mae": model_metrics.get('mae', 0),
            "rmse": model_metrics.get('rmse', 0),
            "r2": model_metrics.get('r2', 0),
            "training_rows": model_metrics.get('training_rows', 0),
            "testing_rows": model_metrics.get('testing_rows', 0),
            "note": "Predictions are estimates based on historical consumption patterns."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching forecast info: {str(e)}")

# Anomalies endpoint
@app.get("/api/anomalies")
def get_anomalies(limit: int = 100):
    """Get detected anomaly records."""
    if anomaly_data is None:
        raise HTTPException(status_code=503, detail="Anomaly data not available")
    
    try:
        # Filter only anomalies
        anomalies = anomaly_data[anomaly_data['anomaly_label'] == 'Anomaly'].head(limit)
        
        # Prepare response data
        anomaly_list = []
        for _, row in anomalies.iterrows():
            anomaly_list.append({
                "datetime": row['datetime'].isoformat(),
                "usage_kwh": float(row['Usage_kWh']),
                "load_type": str(row.get('Load_Type', 'Unknown')),
                "anomaly_label": str(row['anomaly_label'])
            })
        
        return {
            "count": len(anomaly_list),
            "data": anomaly_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching anomaly data: {str(e)}")

# Alerts endpoint
@app.get("/api/alerts")
def get_alerts(limit: int = 50):
    """Generate severity-based alerts from detected anomalies."""
    if anomaly_data is None:
        raise HTTPException(status_code=503, detail="Anomaly data not available")

    if processed_data is None:
        raise HTTPException(status_code=503, detail="Processed data not available")

    try:
        anomalies = anomaly_data[
            anomaly_data['anomaly_label'] == 'Anomaly'
        ].copy()
        critical_threshold = float(
            processed_data['Usage_kWh'].quantile(0.90)
        )
        anomalies = anomalies.sort_values(
            by='datetime',
            ascending=False
        ).head(limit)

        alert_list = []
        for _, row in anomalies.iterrows():
            usage = float(row['Usage_kWh'])
            if usage >= critical_threshold:
                severity = "Critical"
                message = (
                    f"Critical energy anomaly detected: "
                    f"{usage:.2f} kWh consumption."
                )
            else:
                severity = "Warning"
                message = (
                    f"Unusual energy consumption detected: "
                    f"{usage:.2f} kWh."
                )

            alert_list.append({
                "datetime": row['datetime'].isoformat(),
                "usage_kwh": round(usage, 2),
                "load_type": str(row.get('Load_Type', 'Unknown')),
                "severity": severity,
                "status": "Active",
                "message": message
            })

        critical_count = sum(
            1 for alert in alert_list
            if alert["severity"] == "Critical"
        )
        warning_count = sum(
            1 for alert in alert_list
            if alert["severity"] == "Warning"
        )

        return {
            "total_alerts": len(alert_list),
            "critical_alerts": critical_count,
            "warning_alerts": warning_count,
            "critical_threshold_kwh": round(critical_threshold, 2),
            "data": alert_list
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating alerts: {str(e)}"
        )

# Metrics endpoint
@app.get("/api/metrics")
def get_metrics():
    """Get model performance metrics."""
    if model_metrics is None:
        raise HTTPException(status_code=503, detail="Model metrics not available")
    
    try:
        return {
            "model": model_metrics.get('model', 'Unknown'),
            "target": model_metrics.get('target', 'Unknown'),
            "mae": model_metrics.get('mae', 0),
            "rmse": model_metrics.get('rmse', 0),
            "r2": model_metrics.get('r2', 0),
            "anomaly_model": model_metrics.get('anomaly_model', 'Unknown'),
            "anomaly_count": model_metrics.get('anomaly_count', 0),
            "anomaly_percentage": model_metrics.get('anomaly_percentage', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
