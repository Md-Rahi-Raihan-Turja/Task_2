# Task_2

# Biomass Conversion to Biofuel Simulation

This project simulates the process of converting biomass into biofuels. The goal is to analyze how different operating conditions, such as temperature and catalyst concentration, affect the biofuel production rate using a linear regression model.

## Features

1. **Data Import and Preprocessing**
   - Reads biomass conversion data from a CSV file.
   - Cleans the data by handling missing or inconsistent entries.

2. **Process Simulation Model**
   - Utilizes a linear regression model to estimate the biofuel production rate based on temperature and catalyst concentration.
   - Model equation: 
     `Biofuel Production Rate = a * Temperature + b * Catalyst Concentration + c`

3. **Production Rate Analysis**
   - Computes the average biofuel production rate for different combinations of temperature and catalyst concentration.
   - Identifies the optimal operating conditions for maximum production.

4. **Visualization**
   - Scatter plots visualize the relationship between the operating conditions and biofuel production rate.
   - Line charts show the predicted biofuel production rate for varying conditions.

## Prerequisites

- Python 3.6+
- Google Colab
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn

## How to Use

1. **Set Up Google Colab**
   - Upload your CSV file to Google Drive.
   - Update the `file_path` variable in the code to point to your file in Google Drive.

2. **Run the Script**
   - Execute the code to:
     - Load and clean the data.
     - Train the model and analyze results.
     - Visualize relationships and predictions.

3. **Interpret the Results**
   - View scatter plots and line charts to understand the impact of temperature and catalyst concentration.
   - Check the model equation and optimal conditions for insights.

## Code Example

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Specify the path to the CSV file in Google Drive
file_path = '/content/drive/My Drive/extra_1.csv'  # Update with the correct path

# Load the CSV file into a pandas DataFrame
data = pd.read_csv(file_path)

# Data Cleaning
# Check for missing values
data.dropna(inplace=True)

# Ensure all columns are numeric (convert if necessary)
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values after conversion
data.dropna(inplace=True)

# Display the first few rows of the dataset
print("Cleaned Data:")
print(data.head())

# Define features and target variable
X = data[['Temperature', 'Catalyst_Concentration']]
y = data['Biofuel_Production_Rate']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the simulation model using a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients of the model
a, b = model.coef_
c = model.intercept_
print(f"Model equation: Biofuel Production Rate = {a:.2f} * Temperature + {b:.2f} * Catalyst_Concentration + {c:.2f}")

# Predict and calculate errors
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Analyze average biofuel production rate
average_rates = data.groupby(['Temperature', 'Catalyst_Concentration'])['Biofuel_Production_Rate'].mean()
print("\nAverage Biofuel Production Rates:")
print(average_rates)

# Find optimal conditions for maximum production
optimal_conditions = data.loc[data['Biofuel_Production_Rate'].idxmax()]
print("\nOptimal Conditions for Maximum Biofuel Production:")
print(optimal_conditions)

# Visualization
# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Temperature'], data['Biofuel_Production_Rate'], label='Temperature', alpha=0.6)
plt.scatter(data['Catalyst_Concentration'], data['Biofuel_Production_Rate'], label='Catalyst Concentration', alpha=0.6)
plt.title('Biofuel Production Rate vs. Temperature and Catalyst Concentration')
plt.xlabel('Operating Conditions')
plt.ylabel('Biofuel Production Rate')
plt.legend()
plt.show()

# Line chart for predictions
temp_range = np.linspace(data['Temperature'].min(), data['Temperature'].max(), 100)
catalyst_range = np.linspace(data['Catalyst_Concentration'].min(), data['Catalyst_Concentration'].max(), 100)

temp_pred = model.predict(np.column_stack((temp_range, np.full_like(temp_range, catalyst_range.mean()))))
catalyst_pred = model.predict(np.column_stack((np.full_like(catalyst_range, temp_range.mean()), catalyst_range)))

plt.figure(figsize=(10, 6))
plt.plot(temp_range, temp_pred, label='Varying Temperature', color='blue')
plt.plot(catalyst_range, catalyst_pred, label='Varying Catalyst Concentration', color='green')
plt.title('Predicted Biofuel Production Rate')
plt.xlabel('Operating Condition Value')
plt.ylabel('Predicted Biofuel Production Rate')
plt.legend()
plt.show()
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
