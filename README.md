# SURVEX - Urban Mobility and Smart City Prediction System

🏙️ **AI-Powered Urban Intelligence Platform for Smart City Management**

A comprehensive machine learning system that predicts urban challenges including air quality, traffic accidents, parking availability, and citizen activity patterns to enable proactive city management.

## 🎯 Project Overview

This system integrates **4 specialized ML models** to provide real-time predictions for critical urban challenges:
- **Air Quality Prediction** - AQI forecasting using pollutant and weather data
- **Accident Risk Analysis** - Traffic safety risk assessment
- **Smart Parking System** - Parking availability prediction
- **Citizen Activity Monitoring** - Urban activity level forecasting

## 🚀 Key Features

- **99.5% Peak Accuracy** across prediction modules
- **Real-time Web Dashboard** with modern UI
- **Ensemble Learning** using stacking methodology
- **4,000+ Data Points** for robust training
- **Professional Interface** with Bootstrap 5
- **Responsive Design** for all devices

## 📊 Performance Results

| Module | Best Algorithm | Performance |
|--------|---------------|-------------|
| Air Quality | Linear Regression | R² = 0.958 |
| Accident Risk | Stacking Ensemble | 99.5% Accuracy |
| Citizen Activity | Stacking Ensemble | 99.0% Accuracy |
| Smart Parking | Stacking Ensemble | 93.0% Accuracy |

## 🛠️ Technology Stack

**Backend:**
- Python 3.8+
- Flask (Web Framework)
- Scikit-learn (Machine Learning)
- Pandas & NumPy (Data Processing)

**Frontend:**
- HTML5, CSS3, JavaScript
- Bootstrap 5 (UI Framework)
- FontAwesome (Icons)
- Google Fonts (Typography)

**Machine Learning:**
- Random Forest Classifier/Regressor
- Logistic/Linear Regression
- Support Vector Machine (SVM)
- Stacking Ensemble Methods

## 📁 Project Structure

```
smart-city-system/
├── app.py                    # Flask web application
├── templates/
│   └── index.html           # Main web interface
├── smart_city_system.py     # ML model training
├── generate_datasets.py     # Synthetic data generation
├── predict_interface.py     # Prediction demo
├── *.csv                    # Generated datasets
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8 or higher
# pip package manager
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd smart-city-system

# Install required packages
pip install flask pandas numpy scikit-learn matplotlib seaborn
```

### Generate Datasets
```bash
python generate_datasets.py
```
**Output:** Creates 4 CSV files with 1000 samples each
- `accident_risk.csv`
- `air_quality.csv` 
- `citizen_activity.csv`
- `smart_parking.csv`

### Train Models
```bash
python smart_city_system.py
```
**Output:** Trains all ML models and displays performance results

### Run Web Application
```bash
python app.py
```
**Access:** Open http://127.0.0.1:5000 in your browser

## 🎮 Usage Guide

### Web Interface
1. **Select Module** - Choose from 4 prediction modules
2. **Input Parameters** - Fill in the required features
3. **Get Prediction** - Click predict button for results
4. **View Results** - Color-coded predictions (🟢🟡🔴)

### API Usage
```python
# Example API call
import requests

data = {
    'module': 'air_quality',
    'pm25': 85,
    'pm10': 120,
    'no2': 45,
    'co': 1.2,
    'so2': 25,
    'temperature': 28,
    'humidity': 65,
    'wind_speed': 8
}

response = requests.post('http://localhost:5000/predict', json=data)
result = response.json()
print(f"Predicted AQI: {result['prediction']}")
```

## 📈 Model Details

### Air Quality Prediction
- **Input:** PM2.5, PM10, NO2, CO, SO2, Temperature, Humidity, Wind Speed
- **Output:** AQI value (0-500)
- **Best Model:** Linear Regression (R² = 0.958)
- **Use Case:** Environmental monitoring, health alerts

### Accident Risk Analysis
- **Input:** Vehicle density, speed, road/weather conditions, visibility, time
- **Output:** Risk level (Low/Medium/High)
- **Best Model:** Stacking Ensemble (99.5% accuracy)
- **Use Case:** Traffic safety, resource deployment

### Citizen Activity Monitoring
- **Input:** Population density, age, workplaces, events, temperature, day
- **Output:** Activity level (Low/Moderate/High)
- **Best Model:** Stacking Ensemble (99.0% accuracy)
- **Use Case:** Urban planning, resource allocation

### Smart Parking System
- **Input:** Capacity, occupancy, entry/exit rates, time, events
- **Output:** Availability (Available/Full)
- **Best Model:** Stacking Ensemble (93.0% accuracy)
- **Use Case:** Traffic optimization, parking management

## 🔬 Machine Learning Methodology

### Data Generation
- **Synthetic datasets** with realistic urban patterns
- **Domain expertise** for feature relationships
- **Statistical distributions** matching real-world data
- **1000 samples** per module for robust training

### Feature Engineering
- **StandardScaler** for feature normalization
- **PCA** for dimensionality reduction (95% variance)
- **Label encoding** for categorical variables
- **Missing value imputation** using statistical methods

### Model Training
- **Train-test split** (80-20 ratio)
- **Cross-validation** (5-fold) for robust evaluation
- **Hyperparameter tuning** for optimal performance
- **Ensemble methods** for superior accuracy

## 🌟 Real-World Applications

### For City Administrators
- **Proactive Policy Making** based on predictions
- **Resource Optimization** using activity forecasts
- **Emergency Preparedness** with risk assessments
- **Environmental Management** through AQI monitoring

### For Citizens
- **Health Alerts** for air quality conditions
- **Safety Warnings** for high-risk areas
- **Parking Information** for convenient travel
- **Activity Planning** based on urban conditions

## 📊 Performance Metrics

### Classification Metrics
- **Accuracy:** Overall prediction correctness
- **Precision:** True positive rate
- **Recall:** Sensitivity to positive cases
- **F1-Score:** Harmonic mean of precision and recall

### Regression Metrics
- **R² Score:** Variance explained by the model
- **Mean Squared Error:** Average squared prediction error
- **Mean Absolute Error:** Average absolute prediction error

## 🔧 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Install missing packages
pip install <package-name>
```

**Port Already in Use:**
```bash
# Change port in app.py
app.run(debug=True, port=5001)
```

**Dataset Not Found:**
```bash
# Generate datasets first
python generate_datasets.py
```

## 🚀 Future Enhancements

- **Real-time Data Integration** with IoT sensors
- **Mobile Application** for citizen access
- **Advanced Visualization** with charts and maps
- **Multi-city Support** for scalable deployment
- **Deep Learning Models** for complex patterns
- **Cloud Deployment** on AWS/Azure platforms

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Suryaprakash S - SURVEX - Group of tech solutions
- **Sri Eshwar College of Engineering** - *Academic Institution*

## 🙏 Acknowledgments

- Faculty advisors for guidance and support
- Open source community for tools and libraries
- Urban planning research for domain knowledge
- Bootstrap and FontAwesome for UI components

---

**⭐ Star this repository if you found it helpful!**

<<<<<<< HEAD
**🔗 Share with others interested in Smart City AI solutions!**
=======
**🔗 Share SURVEX with others interested in Smart City AI solutions!**
>>>>>>> 233269e (brand update)
