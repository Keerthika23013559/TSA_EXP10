# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 01-11-2025
# Reg no:212223240071

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the laptop price dataset
data = pd.read_csv('/content/laptop_price.csv', encoding='latin1')

# Display numeric columns to choose one for forecasting
print("Numeric columns available:\n", data.select_dtypes(include=[np.number]).columns)

# 🔹 Choose a numeric column for time-series analysis
target_variable = 'Price_euros'   # Change to another numeric column if you prefer (e.g. 'price_euros')

# Create a pseudo 'Date' index since dataset isn’t time-based
data['Index'] = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
data.set_index('Index', inplace=True)

# Plot the selected feature
plt.plot(data.index, data[target_variable])
plt.xlabel('Date')
plt.ylabel(target_variable)
plt.title(f'{target_variable} Time Series')
plt.show()

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Check stationarity
print("\nChecking stationarity for:", target_variable)
check_stationarity(data[target_variable])

# Plot ACF and PACF
plot_acf(data[target_variable])
plt.show()

plot_pacf(data[target_variable])
plt.show()

# Train-test split
train_size = int(len(data) * 0.8)
train, test = data[target_variable][:train_size], data[target_variable][train_size:]

# SARIMA model (you can tune these values)
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit(disp=False)

# Forecast
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('Root Mean Squared Error (RMSE):', rmse)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel(target_variable)
plt.title(f'SARIMA Predictions for {target_variable}')
plt.legend()
plt.show()
```

### OUTPUT:
<img width="702" height="440" alt="image" src="https://github.com/user-attachments/assets/86b60d40-2930-40a4-abbc-71aedea726f8" />
<img width="662" height="420" alt="image" src="https://github.com/user-attachments/assets/905067b8-908a-4f01-b802-bb51ce86b104" />
<img width="619" height="423" alt="image" src="https://github.com/user-attachments/assets/f559c8c9-0c33-4e86-8916-898e8234532f" />
<img width="984" height="536" alt="image" src="https://github.com/user-attachments/assets/ab819c2f-5e95-4226-8ff1-15c470efb686" />


### RESULT:
Thus the program run successfully based on the SARIMA model.
