"""
Data Exploration for Steel Industry Energy Consumption Dataset
This notebook explores the structure and characteristics of the energy consumption data.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Define paths using pathlib for cross-platform compatibility
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'Steel_industry_data.csv'
REPORTS_PATH = BASE_DIR / 'reports'

print("="*60)
print("DATA EXPLORATION - STEEL INDUSTRY DATASET")
print("="*60)

# Verify dataset exists
print(f"\nDataset path: {DATA_PATH}")
if not DATA_PATH.exists():
    print(f"ERROR: Dataset not found at {DATA_PATH}")
    exit(1)
else:
    print("Dataset found successfully!")

# Load the dataset
print("\nLoading the Steel Industry dataset...")
df = pd.read_csv(DATA_PATH)

# Display basic information about the dataset
print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)

# Number of rows and columns
print(f"\nNumber of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Column names
print("\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

# First 5 rows
print("\n" + "="*60)
print("FIRST 5 ROWS")
print("="*60)
print(df.head())

# Data types
print("\n" + "="*60)
print("DATA TYPES")
print("="*60)
print(df.dtypes)

# Missing values
print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)
missing_values = df.isnull().sum()
print(missing_values)
if missing_values.sum() == 0:
    print("\nNo missing values found in the dataset!")
else:
    print(f"\nTotal missing values: {missing_values.sum()}")

# Duplicate count
print("\n" + "="*60)
print("DUPLICATE ROWS")
print("="*60)
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")
if duplicate_count == 0:
    print("No duplicate rows found!")

# Descriptive statistics
print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)
print(df.describe())

# Identify column types
print("\n" + "="*60)
print("COLUMN CLASSIFICATION")
print("="*60)

# Identify datetime-related columns
datetime_cols = []
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        datetime_cols.append(col)
print(f"\nDatetime-related columns: {datetime_cols}")

# Identify numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Numerical columns: {numerical_cols}")

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

# Identify energy consumption target
print("\n" + "="*60)
print("ENERGY CONSUMPTION COLUMN IDENTIFICATION")
print("="*60)
energy_column = 'Usage_kWh'
if energy_column in df.columns:
    print(f"Main energy consumption column: '{energy_column}'")
    print(f"Unit: Kilowatt-hours (kWh)")
    print(f"Range: {df[energy_column].min():.2f} to {df[energy_column].max():.2f} kWh")
    print(f"Mean consumption: {df[energy_column].mean():.2f} kWh")
else:
    print(f"ERROR: Expected column '{energy_column}' not found!")

# Explain target selection
print("\n" + "="*60)
print("TARGET VARIABLE SELECTION")
print("="*60)
print("FORECASTING TARGET: Usage_kWh")
print("\nReasoning:")
print("- 'Usage_kWh' represents actual energy consumption in kilowatt-hours")
print("- This is the most direct measure of electricity usage")
print("- It's a continuous numerical variable suitable for regression")
print("- Other columns like reactive power and CO2 are derived from usage")

# Print unique values for categorical columns
print("\n" + "="*60)
print("CATEGORICAL COLUMNS ANALYSIS")
print("="*60)

for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col}:")
        unique_values = df[col].unique()
        print(f"  Number of unique values: {len(unique_values)}")
        print(f"  Unique values: {unique_values.tolist()}")

# Analyze time information
print("\n" + "="*60)
print("TIME INFORMATION ANALYSIS")
print("="*60)
if 'date' in df.columns:
    print(f"Date column format: {df['date'].iloc[0]}")
    df['datetime'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
    df_sorted = df.sort_values('datetime')
    print(f"Time range: {df_sorted['datetime'].min()} to {df_sorted['datetime'].max()}")
    print(f"Total time span: {df_sorted['datetime'].max() - df_sorted['datetime'].min()}")
    
    # Check data frequency
    if len(df_sorted) > 1:
        time_diffs = df_sorted['datetime'].diff().dropna()
        print(f"Most common time interval: {time_diffs.mode()[0]}")
    
# Create visualizations
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Create reports directory if it doesn't exist
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Energy Consumption Analysis', fontsize=16, fontweight='bold')

# 1. Time series plot of energy consumption
print("Creating time series plot...")
if 'datetime' in df.columns:
    subset_size = min(1000, len(df_sorted))
    axes[0, 0].plot(df_sorted['datetime'].head(subset_size), 
                    df_sorted[energy_column].head(subset_size))
    axes[0, 0].set_title('Energy Consumption Over Time (First 1000 Records)')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Usage (kWh)')
    axes[0, 0].tick_params(axis='x', rotation=45)
else:
    axes[0, 0].text(0.5, 0.5, 'Time series not available', 
                   ha='center', va='center', transform=axes[0, 0].transAxes)

# 2. Distribution of energy consumption
print("Creating distribution plot...")
axes[0, 1].hist(df[energy_column], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Distribution of Energy Consumption')
axes[0, 1].set_xlabel('Usage (kWh)')
axes[0, 1].set_ylabel('Frequency')

# 3. Energy consumption by load type
print("Creating load type comparison...")
if 'Load_Type' in df.columns:
    load_type_consumption = df.groupby('Load_Type')[energy_column].mean()
    axes[1, 0].bar(load_type_consumption.index, load_type_consumption.values)
    axes[1, 0].set_title('Average Energy Consumption by Load Type')
    axes[1, 0].set_xlabel('Load Type')
    axes[1, 0].set_ylabel('Average Usage (kWh)')
    axes[1, 0].tick_params(axis='x', rotation=45)
else:
    axes[1, 0].text(0.5, 0.5, 'Load Type column not available', 
                   ha='center', va='center', transform=axes[1, 0].transAxes)

# 4. Energy consumption by day of week
print("Creating day of week comparison...")
if 'Day_of_week' in df.columns:
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_consumption = df.groupby('Day_of_week')[energy_column].mean()
    day_consumption = day_consumption.reindex(day_order)
    axes[1, 1].bar(day_consumption.index, day_consumption.values)
    axes[1, 1].set_title('Average Energy Consumption by Day of Week')
    axes[1, 1].set_xlabel('Day of Week')
    axes[1, 1].set_ylabel('Average Usage (kWh)')
    axes[1, 1].tick_params(axis='x', rotation=45)
else:
    axes[1, 1].text(0.5, 0.5, 'Day of week column not available', 
                   ha='center', va='center', transform=axes[1, 1].transAxes)

plt.tight_layout()

# Save the visualization
output_path = REPORTS_PATH / 'energy_consumption_overview.png'
print(f"\nSaving visualization to {output_path}...")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("Visualization saved successfully!")
plt.close()

print("\n" + "="*60)
print("DATA EXPLORATION COMPLETE")
print("="*60)
print("\nKey findings:")
print(f"- Dataset contains {df.shape[0]} records")
print(f"- Dataset contains {df.shape[1]} columns")
print(f"- Main target variable: {energy_column} (energy consumption in kWh)")
if 'datetime' in df.columns:
    print(f"- Time range: From {df_sorted['datetime'].min()} to {df_sorted['datetime'].max()}")
if 'Load_Type' in df.columns:
    print(f"- Load types: {df['Load_Type'].unique().tolist()}")
if 'Day_of_week' in df.columns:
    print(f"- Days covered: {df['Day_of_week'].unique().tolist()}")
print(f"- Duplicate rows: {duplicate_count}")
print(f"- Missing values: {missing_values.sum()}")
