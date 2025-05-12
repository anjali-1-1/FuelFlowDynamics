import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load the Dataset
data_url = "India_Statewise_Fuel_Data_.csv"
data = pd.read_csv(data_url)

# Streamlit Dashboard Layout
st.title("FuelFlow Dynamics Dashboard")
st.subheader("Explore Fuel Consumption Patterns")

# User Inputs
states = sorted(data['State'].unique())
state = st.selectbox("Select State:", states)

# Filter dataset by selected state
state_data = data[data['State'] == state]

# Select Year Range
year_min, year_max = st.slider(
    "Select Year Range:",
    min_value=int(state_data['Year'].min()),
    max_value=int(state_data['Year'].max()),
    value=(1950, 2025)
)

# Filter dataset by selected year range
year_data = state_data[(state_data['Year'] >= year_min) & (state_data['Year'] <= year_max)]

# Select Vehicle Type
vehicle_types = sorted(year_data['Vehicle_Type'].unique())
vehicle_type = st.selectbox("Select Vehicle Type:", vehicle_types)

# Filter dataset by selected vehicle type
final_data = year_data[year_data['Vehicle_Type'] == vehicle_type]

# Display Filtered Data and Charts
st.subheader("Filtered Data")
if not final_data.empty:
    st.write(final_data)
    st.download_button("Download Data", final_data.to_csv(), "filtered_data.csv")

    # Chart: Fuel Consumption by Year
    st.subheader("Fuel Consumption Trends")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if year >= 2025 else 'blue' for year in final_data['Year']]
    sns.barplot(data=final_data, x='Year', y='Fuel_Consumption (Million Liters)', ax=ax, palette=colors)
    ax.set_title(f"Fuel Consumption in {state} ({vehicle_type})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fuel Consumption (Million Liters)")
    st.pyplot(fig)

    # Fuel Price Trends
    st.subheader("Fuel Price Trends")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=final_data, x='Year', y='Fuel_Price (INR per Liter)', marker='o', ax=ax2, color="blue", label="1950-2024")
    sns.scatterplot(data=final_data[final_data['Year'] >= 2025], x='Year', y='Fuel_Price (INR per Liter)', color="red", s=100, label="2025-2050")
    ax2.set_title(f"Fuel Price Trends in {state} ({vehicle_type})")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Fuel Price (INR per Liter)")
    ax2.legend()
    st.pyplot(fig2)

    # Population Trends
    st.subheader("Population Trends")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=final_data, x='Year', y='Population', marker='o', ax=ax3, color="blue", label="1950-2024")
    sns.scatterplot(data=final_data[final_data['Year'] >= 2025], x='Year', y='Population', color="red", s=100, label="2025-2050")
    ax3.set_title(f"Population Trends in {state} ({vehicle_type})")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Population")
    ax3.legend()
    st.pyplot(fig3)
else:
    st.write("No data available for the selected filters.")

# Summary Insights
st.subheader("Summary Insights")
if not final_data.empty:
    total_consumption = final_data['Fuel_Consumption (Million Liters)'].sum()
    avg_price = final_data['Fuel_Price (INR per Liter)'].mean()
    population = final_data['Population'].mean()

    st.write(f"Total Fuel Consumption: {total_consumption:.2f} Million Liters")
    st.write(f"Average Fuel Price: {avg_price:.2f} INR per Liter")
    st.write(f"Average Population: {population:.0f}")
else:
    st.write("No insights available for the selected filters.")

# --- Fuel Consumption Prediction Section ---
st.title("🔮 Predict Fuel Consumption")
st.subheader("Use Machine Learning to Predict Future Fuel Needs")

# Inputs for prediction
selected_state = st.selectbox("Select State for Prediction", states, key="pred_state")
selected_vehicle_type = st.selectbox("Select Vehicle Type", vehicle_types, key="pred_vehicle")
selected_year = st.number_input("Enter Year for Prediction", min_value=1950, max_value=2050, value=2025)

# Dynamic user inputs for additional features
user_income = st.number_input("Enter Estimated Income (Crore INR)", min_value=0.0, value=1000.0)
user_price = st.number_input("Enter Estimated Fuel Price (INR per Liter)", min_value=0.0, value=100.0)
user_population = st.number_input("Enter Estimated Population", min_value=0.0, value=1000000.0)

# Load and preprocess data again for structure
data_for_model = pd.read_csv("India_Statewise_Fuel_Data_.csv")
data_for_model['Year'] = data_for_model['Year'].astype(int)
data_for_model['Fuel_Consumption (Million Liters)'] = data_for_model['Fuel_Consumption (Million Liters)'].astype(float)
data_for_model['Fuel_Price (INR per Liter)'] = data_for_model['Fuel_Price (INR per Liter)'].astype(float)
data_encoded = pd.get_dummies(data_for_model, columns=['State', 'Vehicle_Type'], drop_first=True)

X = data_encoded.drop(['Fuel_Consumption (Million Liters)'], axis=1)
model = joblib.load('fuel_consumption_model.pkl')

# Prediction input
def construct_input(state, year, vehicle_type, income, price, population):
    input_data = {
        'Year': [year],
        'Income (Crore INR)': [income],
        'Fuel_Price (INR per Liter)': [price],
        'Population': [population]
    }
    for col in X.columns:
        if col.startswith('State_'):
            input_data[col] = [1 if f'State_{state}' == col else 0]
        elif col.startswith('Vehicle_Type_'):
            input_data[col] = [1 if f'Vehicle_Type_{vehicle_type}' == col else 0]
    for col in X.columns:
        if col not in input_data:
            input_data[col] = 0
    return pd.DataFrame(input_data)

if st.button("Predict Fuel Consumption"):
    input_df = construct_input(selected_state, selected_year, selected_vehicle_type, user_income, user_price, user_population)
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Fuel Consumption: **{prediction:.2f} Million Liters**")

    fig, ax = plt.subplots()
    ax.bar(f"{selected_state} - {selected_year} ({selected_vehicle_type})", prediction, color='green')
    ax.set_ylabel("Fuel Consumption (Million Liters)")
    ax.set_title("Predicted Fuel Consumption")
    st.pyplot(fig)
