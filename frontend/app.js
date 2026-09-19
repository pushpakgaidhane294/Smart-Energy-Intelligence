// Smart Energy Intelligence Dashboard JavaScript

const API_BASE_URL = 'http://127.0.0.1:8000';

// Chart instances
let consumptionChart = null;
let forecastChart = null;
let anomalyChart = null;
let hourlyChart = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    checkAPIStatus();
    loadDashboardData();
});

// Check API status
async function checkAPIStatus() {
    const statusElement = document.getElementById('apiStatus');
    
    try {
        const response = await fetch(`${API_BASE_URL}/`);
        if (response.ok) {
            statusElement.textContent = 'ONLINE';
            statusElement.className = 'status-badge online';
        } else {
            throw new Error('API not responding correctly');
        }
    } catch (error) {
        statusElement.textContent = 'OFFLINE';
        statusElement.className = 'status-badge offline';
        showErrorMessage();
    }
}

// Load all dashboard data
async function loadDashboardData() {
    try {
        await Promise.all([
            loadSummaryData(),
            loadConsumptionData(),
            loadForecastData(),
            loadAnomalyData(),
            loadMetricsData()
        ]);
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showErrorMessage();
    }
}

// Load summary data
async function loadSummaryData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/summary`);
        if (!response.ok) throw new Error('Failed to load summary data');
        
        const data = await response.json();
        
        document.getElementById('avgConsumption').textContent = data.average_consumption.toFixed(2);
        document.getElementById('maxConsumption').textContent = data.maximum_consumption.toFixed(2);
        document.getElementById('predictedDemand').textContent = data.predicted_demand.toFixed(2);
        document.getElementById('detectedAnomalies').textContent = data.anomaly_count;
    } catch (error) {
        console.error('Error loading summary:', error);
    }
}

// Load consumption data for chart
async function loadConsumptionData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/consumption?limit=500`);
        if (!response.ok) throw new Error('Failed to load consumption data');
        
        const data = await response.json();
        
        const labels = data.data.map(item => new Date(item.datetime).toLocaleTimeString());
        const values = data.data.map(item => item.usage_kwh);
        
        createConsumptionChart(labels, values);
        createHourlyChart(data.data);
    } catch (error) {
        console.error('Error loading consumption data:', error);
    }
}

// Load forecast data
async function loadForecastData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/forecast`);
        if (!response.ok) throw new Error('Failed to load forecast data');
        
        const data = await response.json();
        
        document.getElementById('forecastMAE').textContent = data.mae.toFixed(4) + ' kWh';
        document.getElementById('forecastRMSE').textContent = data.rmse.toFixed(4) + ' kWh';
        document.getElementById('forecastR2').textContent = data.r2.toFixed(4);
        
        // Create forecast chart with simulated data (since we don't have actual predictions in API)
        createForecastChart();
    } catch (error) {
        console.error('Error loading forecast data:', error);
    }
}

// Load anomaly data
async function loadAnomalyData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/anomalies?limit=50`);
        if (!response.ok) throw new Error('Failed to load anomaly data');
        
        const data = await response.json();
        
        // Update anomaly table
        updateAnomalyTable(data.data);
        
        // Create anomaly chart
        createAnomalyChart(data.data);
    } catch (error) {
        console.error('Error loading anomaly data:', error);
        document.getElementById('anomalyTableBody').innerHTML = 
            '<tr><td colspan="4" class="loading-message">No anomaly data available</td></tr>';
    }
}

// Load metrics data
async function loadMetricsData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/metrics`);
        if (!response.ok) throw new Error('Failed to load metrics data');
        
        const data = await response.json();
        
        document.getElementById('r2Score').textContent = data.r2.toFixed(4);
        document.getElementById('rmseScore').textContent = data.rmse.toFixed(4);
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

// Create consumption chart
function createConsumptionChart(labels, values) {
    const ctx = document.getElementById('consumptionChart').getContext('2d');
    
    if (consumptionChart) {
        consumptionChart.destroy();
    }
    
    consumptionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Energy Consumption (kWh)',
                data: values,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    display: false // Hide x-axis labels for better readability
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Consumption (kWh)'
                    }
                }
            }
        }
    });
}

// Create forecast chart (simulated)
function createForecastChart() {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    if (forecastChart) {
        forecastChart.destroy();
    }
    
    // Generate simulated data for demonstration
    const labels = Array.from({length: 50}, (_, i) => i + 1);
    const actualData = Array.from({length: 50}, () => 20 + Math.random() * 30);
    const predictedData = actualData.map(val => val + (Math.random() - 0.5) * 10);
    
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Actual',
                data: actualData,
                borderColor: '#27ae60',
                backgroundColor: 'rgba(39, 174, 96, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.4,
                pointRadius: 0
            }, {
                label: 'Predicted',
                data: predictedData,
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time Steps'
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Consumption (kWh)'
                    }
                }
            }
        }
    });
}

// Create anomaly chart
function createAnomalyChart(anomalies) {
    const ctx = document.getElementById('anomalyChart').getContext('2d');
    
    if (anomalyChart) {
        anomalyChart.destroy();
    }
    
    const labels = anomalies.map(item => new Date(item.datetime).toLocaleDateString());
    const values = anomalies.map(item => item.usage_kwh);
    
    anomalyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Anomalous Consumption (kWh)',
                data: values,
                backgroundColor: 'rgba(231, 76, 60, 0.7)',
                borderColor: '#e74c3c',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    display: false // Hide x-axis labels for better readability
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Consumption (kWh)'
                    }
                }
            }
        }
    });
}

// Create hourly consumption chart
function createHourlyChart(consumptionData) {
    const ctx = document.getElementById('hourlyChart').getContext('2d');
    
    if (hourlyChart) {
        hourlyChart.destroy();
    }
    
    // Aggregate consumption by hour
    const hourlyData = {};
    consumptionData.forEach(item => {
        const hour = new Date(item.datetime).getHours();
        if (!hourlyData[hour]) {
            hourlyData[hour] = [];
        }
        hourlyData[hour].push(item.usage_kwh);
    });
    
    const hours = Object.keys(hourlyData).sort();
    const avgConsumption = hours.map(hour => {
        const values = hourlyData[hour];
        return values.reduce((a, b) => a + b, 0) / values.length;
    });
    
    hourlyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: hours.map(h => `${h}:00`),
            datasets: [{
                label: 'Average Consumption (kWh)',
                data: avgConsumption,
                backgroundColor: 'rgba(52, 152, 219, 0.7)',
                borderColor: '#3498db',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Hour of Day'
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Average Consumption (kWh)'
                    }
                }
            }
        }
    });
}

// Update anomaly table
function updateAnomalyTable(anomalies) {
    const tableBody = document.getElementById('anomalyTableBody');
    
    if (anomalies.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="4" class="loading-message">No anomalies detected</td></tr>';
        return;
    }
    
    tableBody.innerHTML = anomalies.map(anomaly => `
        <tr>
            <td>${new Date(anomaly.datetime).toLocaleString()}</td>
            <td>${anomaly.usage_kwh.toFixed(2)}</td>
            <td>${anomaly.load_type}</td>
            <td><span class="anomaly-badge ${anomaly.anomaly_label.toLowerCase()}">${anomaly.anomaly_label}</span></td>
        </tr>
    `).join('');
}

// Show error message
function showErrorMessage() {
    const errorElement = document.getElementById('errorMessage');
    errorElement.classList.remove('hidden');
}

// Auto-refresh data every 30 seconds
setInterval(() => {
    checkAPIStatus();
    loadDashboardData();
}, 30000);
