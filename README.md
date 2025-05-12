# FuelFlowDynamics
FuelFlow Dynamics is a data-driven web dashboard and machine learning model built using Python, Streamlit, and Random Forests to analyze, visualize, and predict fuel consumption patterns across Indian states based on vehicle type, population, fuel prices, and economic factors.

# Project Structure
India_Statewise_Fuel_Data_.csv: Dataset containing fuel consumption, price, population, and income data by state and vehicle type.
ML model.py: Python script that processes data, trains a Random Forest Regressor, and saves the model.
fuel_consumption_model.pkl: Trained machine learning model for predicting future fuel consumption.
website.py: Streamlit-powered web dashboard for data exploration and prediction.

# Features

**Interactive Dashboard**: Filter fuel data by state, year, and vehicle type.
**Visual Insights**: Line charts and bar graphs for fuel trends, pricing, and population.
**ML-Powered Predictions**: Predict future fuel needs using a trained regression model.
**Downloadable Filtered Data**: Export your customized views for offline analysis.

# Getting Started

# Prerequisites
```bash
Make sure you have the following installed:
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install joblib
pip installstreamlit

# Run the dashboard
```bash
streamlit run website.py

## Model Training

The `ML model.py` script:

- Loads and preprocesses the dataset.
- Visualizes trends and correlations.
- Trains a Random Forest Regressor.
- Evaluates model performance using MAE and R².
- Saves the model as `fuel_consumption_model.pkl`.

You can retrain the model with:

```bash
python "ML model.py"

