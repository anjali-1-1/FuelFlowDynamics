import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
data_url = "India_Statewise_Fuel_Data_.csv"
data = pd.read_csv(data_url)
print("Basic Information about the Dataset:")
data.info()
print("\nPreview of the Dataset:")
print(data.head())
print("\nMissing Values per Column:")
print(data.isnull().sum())
#Data Preprocessing
data['Year'] = data['Year'].astype(int)
data['Fuel_Consumption (Million Liters)'] = data['Fuel_Consumption (Million Liters)'].astype(float)
data['Fuel_Price (INR per Liter)'] = data['Fuel_Price (INR per Liter)'].astype(float)
data = pd.get_dummies(data, columns=['State', 'Vehicle_Type'], drop_first=True)
# Plot fuel consumption trends over years
print("\nVisualizing Fuel Consumption Trends Over Time:")
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Year', y='Fuel_Consumption (Million Liters)', ci=None)
plt.title("Fuel Consumption Over Time")
plt.xlabel("Year")
plt.ylabel("Fuel Consumption (Million Liters)")
plt.show()
# Plot correlation heatmap
print("\nCorrelation Heatmap:")
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
X = data.drop(['Fuel_Consumption (Million Liters)'], axis=1)
y = data['Fuel_Consumption (Million Liters)']

# Split the data into training and testing sets
print("\nSplitting Data into Training and Testing Sets:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)
# Initialize the Random Forest Regressor
print("\nTraining the Random Forest Regressor:")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model Training Complete.")
# Save the trained model
import joblib
joblib.dump(model, 'fuel_consumption_model.pkl')
print("✅ Model saved as fuel_consumption_model.pkl")

# Make predictions on the test set
print("\nEvaluating the Model:")
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)
#  Define Prediction Function
def predict_fuel(state, year, vehicle_type):
    # Create a dictionary for prediction
    input_data = {
        'Year': [year],
        'Income (Crore INR)': [X['Income (Crore INR)'].mean()],  # Using mean income for prediction
        'Fuel_Price (INR per Liter)': [X['Fuel_Price (INR per Liter)'].mean()],  # Using mean fuel price
        'Population': [X['Population'].mean()]  # Using mean population
    }

    # Add state and vehicle type columns
    for col in X.columns:
        if col.startswith('State_'):
            input_data[col] = [1 if f'State_{state}' == col else 0]
        elif col.startswith('Vehicle_Type_'):
            input_data[col] = [1 if f'Vehicle_Type_{vehicle_type}' == col else 0]

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame(input_data)

    # Ensure all columns in the model are present in input_df
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Predict fuel consumption
    prediction = model.predict(input_df)[0]

    # Print and visualize the result
    print(f"\nPredicted Fuel Consumption for {state} in {year} for {vehicle_type}: {prediction:.2f} Million Liters")

    plt.bar([f'{state} - {year} ({vehicle_type})'], [prediction], color='skyblue')
    plt.title("Predicted Fuel Consumption")
    plt.ylabel("Fuel Consumption (Million Liters)")
    plt.show()



