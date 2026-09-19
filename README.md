# Smart Energy Intelligence

## Industrial Electricity Demand Forecasting & Anomaly Detection System

A comprehensive machine learning web application for analyzing industrial electricity consumption patterns, forecasting future demand, and detecting unusual consumption patterns.

## 🌟 Project Overview

Smart Energy Intelligence is a real-world application that leverages machine learning to analyze industrial electricity consumption data from steel industry operations. The system provides accurate demand forecasting and anomaly detection capabilities through a professional web dashboard.

## 🎯 Problem Statement

Industrial facilities consume significant amounts of electricity, and understanding consumption patterns is crucial for:

- **Cost Optimization**: Reducing energy costs through better demand planning
- **Operational Efficiency**: Identifying inefficiencies in energy usage
- **Predictive Maintenance**: Detecting unusual patterns that may indicate equipment issues
- **Grid Management**: Helping utilities balance supply and demand

## 🚀 Objectives

1. **Analyze** historical electricity consumption patterns
2. **Forecast** future energy demand using machine learning
3. **Detect** anomalous consumption patterns automatically
4. **Visualize** data through an interactive web dashboard
5. **Provide** real-time insights through REST API endpoints

## 📊 Dataset

The project uses the **Steel Industry Energy Consumption Dataset** containing:

- **35,040 records** of 15-minute interval measurements
- **Time period**: January 1, 2018 to December 31, 2018
- **11 columns** including energy consumption, reactive power, CO2 emissions, and load types

### Key Features:
- Usage_kWh: Actual energy consumption (target variable)
- Lagging/Leading reactive power measurements
- Power factor metrics
- CO2 emissions
- Load types: Light_Load, Medium_Load, Maximum_Load
- Temporal features: Time, day of week, weekend status

## 🛠 Technology Stack

### Backend
- **Python**: Core programming language
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **joblib**: Model serialization

### Machine Learning
- **RandomForestRegressor**: Energy consumption forecasting
- **IsolationForest**: Anomaly detection

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling with modern design
- **JavaScript**: Client-side logic
- **Chart.js**: Data visualization

### Data Processing
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization

## 🏗 System Architecture

```
Smart-Energy-Intelligence/
├── data/
│   ├── raw/
│   │   └── Steel_industry_data.csv          # Original dataset
│   └── processed/
│       ├── energy_processed.csv             # Preprocessed data
│       └── anomaly_results.csv              # Anomaly detection results
├── models/
│   ├── forecasting_model.pkl                # Trained forecasting model
│   ├── anomaly_model.pkl                    # Trained anomaly detection model
│   ├── forecast_features.json               # Feature information
│   └── model_metrics.json                   # Model performance metrics
├── notebooks/
│   ├── 01_data_exploration.py               # EDA and visualization
│   ├── 02_preprocessing.py                  # Data preprocessing
│   ├── 03_forecasting_model.py              # Model training
│   └── 04_anomaly_detection.py              # Anomaly detection
├── backend/
│   └── app.py                               # FastAPI application
├── frontend/
│   ├── index.html                           # Dashboard interface
│   ├── style.css                            # Styling
│   └── app.js                               # Frontend logic
├── reports/
│   ├── energy_consumption_overview.png      # EDA visualization
│   ├── actual_vs_predicted.png              # Forecast visualization
│   ├── anomaly_detection.png                # Anomaly visualization
│   └── project_development_report.md        # Detailed project report
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
└── .gitignore                               # Git ignore rules
```

## 🔍 Data Preprocessing

### Steps Performed:

1. **Datetime Conversion**: Parse date strings to datetime objects
2. **Chronological Sorting**: Order data by timestamp
3. **Feature Engineering**:
   - Time features: hour, day, month, day of week, weekend indicator
   - Lag features: 15-minute, 1-hour, and 1-day lags
4. **Categorical Encoding**:
   - One-hot encoding for load types
   - Label encoding for week status and day of week
5. **Missing Value Handling**: Remove rows with missing lag values

### Feature Set:
- **Temporal features**: hour, day, month, day_of_week_num, weekend
- **Lag features**: lag_1 (15 min), lag_4 (1 hour), lag_96 (1 day)
- **Electrical features**: reactive power, power factor, CO2 emissions
- **Load type indicators**: Light_Load, Medium_Load, Maximum_Load

## 🤖 Machine Learning Approach

### Forecasting Model

**Algorithm**: Random Forest Regressor

**Target Variable**: Usage_kWh (energy consumption in kWh)

**Features**: 18 engineered features including temporal, lag, and electrical measurements

**Training Strategy**: Chronological train/test split (80/20)

**Model Parameters**:
- n_estimators: 200
- max_depth: 20
- random_state: 42
- n_jobs: -1 (parallel processing)

### Model Evaluation

**Metrics** (to be calculated after execution):
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error  
- **R²**: Coefficient of Determination

### Anomaly Detection

**Algorithm**: Isolation Forest

**Features**: Energy-related numerical columns
- Usage_kWh
- Reactive power measurements
- CO2 emissions
- Power factors
- NSM (Number of Seconds from Midnight)

**Parameters**:
- contamination: 'auto'
- random_state: 42
- n_estimators: 100

**Output**: Binary classification (Normal/Anomaly)

## 📡 API Endpoints

### Base URL: `http://127.0.0.1:8000`

### Endpoints:

#### `GET /`
- **Description**: API health check
- **Response**: 
  ```json
  {
    "status": "online",
    "project": "Smart Energy Intelligence"
  }
  ```

#### `GET /api/summary`
- **Description**: Summary statistics
- **Response**: Total records, average/min/max consumption, predicted demand, anomaly count

#### `GET /api/consumption`
- **Description**: Historical consumption data
- **Parameters**: `limit` (default: 1000)
- **Response**: Array of consumption records with datetime and usage

#### `GET /api/forecast`
- **Description**: Forecast model information
- **Response**: Model type, target variable, performance metrics

#### `GET /api/anomalies`
- **Description**: Detected anomaly records
- **Parameters**: `limit` (default: 100)
- **Response**: Array of anomaly records with consumption details

#### `GET /api/metrics`
- **Description**: Model performance metrics
- **Response**: MAE, RMSE, R², anomaly statistics

### API Documentation
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

## 🎨 Dashboard Features

### Summary Cards
- Average Consumption
- Maximum Consumption  
- Predicted Demand
- Detected Anomalies
- R² Score
- RMSE

### Interactive Charts
1. **Historical Electricity Consumption**: Time series of energy usage
2. **Actual vs Predicted**: Model performance visualization
3. **Anomaly Detection**: Highlighted unusual patterns
4. **Consumption by Hour**: Hourly consumption patterns

### Anomaly Table
- Date/Time of anomalies
- Consumption values
- Load type classification
- Anomaly status badges

### Forecast Section
- Model information
- Performance metrics
- Predictive accuracy indicators

### System Status
- Real-time API status indicator
- Error handling and fallback messages

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository** (if applicable)
   ```bash
   git clone <repository-url>
   cd Smart-Energy-Intelligence
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix/MacOS:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify dataset**
   - Ensure `data/raw/Steel_industry_data.csv` exists

## 🚀 Running the Project

### Step 1: Data Exploration
```bash
python notebooks/01_data_exploration.py
```
- Inspects dataset structure
- Generates visualizations
- Saves to `reports/energy_consumption_overview.png`

### Step 2: Data Preprocessing
```bash
python notebooks/02_preprocessing.py
```
- Preprocesses the dataset
- Creates engineered features
- Saves to `data/processed/energy_processed.csv`

### Step 3: Model Training
```bash
python notebooks/03_forecasting_model.py
```
- Trains Random Forest model
- Evaluates performance
- Saves model and metrics

### Step 4: Anomaly Detection
```bash
python notebooks/04_anomaly_detection.py
```
- Trains Isolation Forest
- Detects anomalies
- Saves results and visualizations

### Step 5: Start FastAPI Backend
```bash
python -m uvicorn backend.app:app --reload
```
- API will be available at `http://127.0.0.1:8000`
- Access docs at `http://127.0.0.1:8000/docs`

### Step 6: Open Frontend
- Open `frontend/index.html` in a web browser
- Or use a local server:
  ```bash
  # Using Python
  python -m http.server 8001 --directory frontend
  # Then navigate to http://localhost:8001
  ```

## 📈 Model Evaluation

### Forecasting Model Performance
*Metrics will be populated after model execution*

- **MAE**: *To be calculated*
- **RMSE**: *To be calculated*
- **R² Score**: *To be calculated*
- **Training Rows**: *To be calculated*
- **Testing Rows**: *To be calculated*

### Anomaly Detection Results
*Results will be populated after execution*

- **Total Records**: *To be calculated*
- **Normal Records**: *To be calculated*
- **Anomaly Records**: *To be calculated*
- **Anomaly Percentage**: *To be calculated*

## ⚠️ Limitations

1. **Data Dependency**: Model accuracy depends on data quality and quantity
2. **Temporal Scope**: Predictions based on historical patterns may not account for sudden changes
3. **Feature Scope**: Limited to available features in the dataset
4. **Real-time Updates**: Current implementation uses batch processing
5. **Single Facility**: Trained on data from one industrial facility

## 🔮 Future Scope

1. **Real-time Processing**: Implement streaming data processing
2. **Multi-facility Support**: Extend to handle multiple industrial facilities
3. **Advanced Models**: Experiment with deep learning models (LSTM, GRU)
4. **Weather Integration**: Incorporate weather data for improved forecasting
5. **Alert System**: Implement automated anomaly alerts
6. **User Authentication**: Add user management and access control
7. **Database Integration**: Replace CSV files with proper database
8. **Mobile App**: Develop mobile application for on-the-go monitoring
9. **Export Features**: Add data export capabilities
10. **Custom Alerts**: Allow users to set custom thresholds

## 📝 Notes

- **No Fake Data**: All results are based on actual dataset analysis
- **No Database**: Uses file-based storage for simplicity
- **No Authentication**: Open API for demonstration purposes
- **Beginner-Friendly**: Code includes comprehensive comments
- **Cross-Platform**: Uses pathlib for Windows/Mac/Linux compatibility

## 🤝 Contributing

This is a demonstration project. For production use, consider:
- Adding comprehensive error handling
- Implementing logging
- Adding unit tests
- Setting up CI/CD pipeline
- Implementing security best practices

## 📄 License

This project is for educational and demonstration purposes.

## 👥 Support

For issues or questions, please refer to the project documentation or create an issue in the repository.

---

**Note**: Model metrics and results will be populated after executing the training scripts. The current README provides a framework for the complete system.
