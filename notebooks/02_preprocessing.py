"""
Data Preprocessing for Steel Industry Energy Consumption Dataset
This notebook preprocesses the data for machine learning model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define paths using pathlib for cross-platform compatibility
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'Steel_industry_data.csv'
PROCESSED_PATH = BASE_DIR / 'data' / 'processed' / 'energy_processed.csv'

print("="*60)
print("DATA PREPROCESSING - STEEL INDUSTRY DATASET")
print("="*60)

# Load the original dataset
print(f"\nLoading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

print(f"\nOriginal dataset shape: {df.shape}")
print(f"Original columns: {df.columns.tolist()}")

# Create a copy to avoid modifying the original
df_processed = df.copy()

# Convert date column to datetime
print("\n" + "="*60)
print("DATETIME CONVERSION")
print("="*60)
df_processed['datetime'] = pd.to_datetime(df_processed['date'], format='%d/%m/%Y %H:%M')
print("Date column converted to datetime format")

# Sort chronologically
df_processed = df_processed.sort_values('datetime').reset_index(drop=True)
print("Dataset sorted chronologically")

# Create time-based features
print("\n" + "="*60)
print("FEATURE ENGINEERING - TIME FEATURES")
print("="*60)

df_processed['hour'] = df_processed['datetime'].dt.hour
df_processed['day'] = df_processed['datetime'].dt.day
df_processed['month'] = df_processed['datetime'].dt.month
df_processed['day_of_week_num'] = df_processed['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
df_processed['weekend'] = (df_processed['day_of_week_num'] >= 5).astype(int)

print("Created time features:")
print("- hour: Hour of the day (0-23)")
print("- day: Day of the month (1-31)")
print("- month: Month (1-12)")
print("- day_of_week_num: Day of week as number (0-6)")
print("- weekend: Binary indicator (1 for weekend, 0 for weekday)")

# Create lag features (based on 15-minute intervals)
print("\n" + "="*60)
print("FEATURE ENGINEERING - LAG FEATURES")
print("="*60)

# Since data is at 15-minute intervals:
# lag_1 = 15 minutes ago
# lag_4 = 1 hour ago (4 * 15 min)
# lag_96 = 1 day ago (96 * 15 min)
# lag_672 = 1 week ago (672 * 15 min)

target_column = 'Usage_kWh'
df_processed['lag_1'] = df_processed[target_column].shift(1)
df_processed['lag_4'] = df_processed[target_column].shift(4)  # 1 hour
df_processed['lag_96'] = df_processed[target_column].shift(96)  # 1 day

print("Created lag features:")
print("- lag_1: Usage 15 minutes ago")
print("- lag_4: Usage 1 hour ago")
print("- lag_96: Usage 1 day ago")

# Encode categorical variables
print("\n" + "="*60)
print("CATEGORICAL ENCODING")
print("="*60)

# Encode WeekStatus
df_processed['WeekStatus_encoded'] = df_processed['WeekStatus'].map({'Weekday': 0, 'Weekend': 1})
print("Encoded WeekStatus: Weekday=0, Weekend=1")

# Encode Load_Type using one-hot encoding
load_type_dummies = pd.get_dummies(df_processed['Load_Type'], prefix='Load_Type')
df_processed = pd.concat([df_processed, load_type_dummies], axis=1)
print(f"Created one-hot encoding for Load_Type: {load_type_dummies.columns.tolist()}")

# Encode Day_of_week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_mapping = {day: i for i, day in enumerate(day_order)}
df_processed['Day_of_week_encoded'] = df_processed['Day_of_week'].map(day_mapping)
print("Encoded Day_of_week: Monday=0, Tuesday=1, ..., Sunday=6")

# Handle missing values created by lag features
print("\n" + "="*60)
print("HANDLING MISSING VALUES")
print("="*60)
missing_before = df_processed.isnull().sum().sum()
print(f"Missing values before handling: {missing_before}")

# Drop rows with missing values (from lag features)
df_processed = df_processed.dropna()
missing_after = df_processed.isnull().sum().sum()
print(f"Missing values after handling: {missing_after}")

# Select final columns for modeling
print("\n" + "="*60)
print("FINAL COLUMN SELECTION")
print("="*60)

# Keep original columns plus engineered features
columns_to_keep = [
    'datetime', 'date', target_column,
    'Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh',
    'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM',
    'WeekStatus', 'Day_of_week', 'Load_Type',
    'hour', 'day', 'month', 'day_of_week_num', 'weekend',
    'lag_1', 'lag_4', 'lag_96',
    'WeekStatus_encoded', 'Day_of_week_encoded'
] + load_type_dummies.columns.tolist()

# Filter to only existing columns
columns_to_keep = [col for col in columns_to_keep if col in df_processed.columns]
df_processed = df_processed[columns_to_keep]

print(f"Final columns ({len(df_processed.columns)}): {df_processed.columns.tolist()}")

# Save processed dataset
print("\n" + "="*60)
print("SAVING PROCESSED DATASET")
print("="*60)

# Create processed directory if it doesn't exist
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

df_processed.to_csv(PROCESSED_PATH, index=False)
print(f"Processed dataset saved to: {PROCESSED_PATH}")

# Print summary
print("\n" + "="*60)
print("PREPROCESSING SUMMARY")
print("="*60)
print(f"Original shape: {df.shape}")
print(f"Processed shape: {df_processed.shape}")
print(f"Rows removed: {df.shape[0] - df_processed.shape[0]}")
print(f"Rows remaining: {df_processed.shape[0]}")
print(f"Final columns: {len(df_processed.columns)}")
print(f"Missing values: {df_processed.isnull().sum().sum()}")
print(f"\nTime range: {df_processed['datetime'].min()} to {df_processed['datetime'].max()}")
print(f"Target column: {target_column}")
print(f"Target range: {df_processed[target_column].min():.2f} to {df_processed[target_column].max():.2f} kWh")
print(f"Target mean: {df_processed[target_column].mean():.2f} kWh")

print("\n" + "="*60)
print("PREPROCESSING COMPLETE")
print("="*60)
