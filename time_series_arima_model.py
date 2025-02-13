# We will use ARIMA model to forecast the electricity load demand in the UK. 
# ARIMA is a time series forecasting model that uses past data to predict future values.  
# The model has three main parameters: p, d, and q.
# p: The number of lag observations included in the model (lag order).
# d: The number of times that the raw observations are differenced (degree of differencing).
# q: The size of the moving average window (order of moving average).
# We will use the auto_arima function from the pmdarima library to automatically find the best parameters for the ARIMA model.
# We will then fit the ARIMA model to the data and forecast the electricity demand for the next 30 days.
# Finally, we will plot the forecasted demand along with the observed data to visualize the results.

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima, StepwiseContext
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt

import warnings
warnings.filterwarnings("ignore")

# Load dataset and combine all years data into one dataset
#df = pd.read_csv('data/UK_data/demanddata_2011_2025.csv', parse_dates=['SETTLEMENT_DATE'], index_col='SETTLEMENT_DATE')

# Load dataset and combine all years data into one dataset
#year = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
year = [2022, 2023, 2024, 2025]
df = pd.read_csv('data/UK_data/demanddata_2022.csv', parse_dates=['SETTLEMENT_DATE'], index_col='SETTLEMENT_DATE')

#Use only 2018 to 2025 data to reduce the size of the dataset for ARIMA model
for i in range(len(year)):
    data_path = f'data/UK_data/demanddata_{year[i]}.csv'    
    df_year = pd.read_csv(data_path, parse_dates=['SETTLEMENT_DATE'], index_col='SETTLEMENT_DATE') 
    df = pd.concat([df, df_year], ignore_index=False)

# Save the combined dataset
df.to_csv('data/UK_data/demanddata_2018_2025.csv')

#Exploratory data analysis
# Check the first few rows of the data
print(df.head())

# Check the last few rows of the data
print(df.tail())

# Check the data types of the columns
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Check the summary statistics of the data
print(df.describe())   

# Select the demand column for analysis
df['EMBEDDED_SOLAR_GENERATION'] = df['EMBEDDED_SOLAR_GENERATION'].astype('float32')

df_solar = df['EMBEDDED_SOLAR_GENERATION']  # Embedded solar is the column for demand
print(df_solar.head())

#df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'])
#df_d = df.sort_values(by=['SETTLEMENT_DATE'])
#df_ = df.set_index('SETTLEMENT_DATE').resample('60min').mean()

# Resample to average values for hourly data
data = df_solar.resample('60min').mean()

# Visualise acf and pacf
plot_acf(data)
plot_pacf(data)
plt.show()

# Plot the solar generation data
plt.figure(figsize=(12, 6))
plt.plot(pd.date_range(data.index[-1], freq= 'H', periods=data.shape[0]), data, label="Solar Generation")
plt.title("UK Solar Generation Over Time")
plt.xlabel("Date")
plt.ylabel(" Solar Generation (MW)")
plt.legend()
#plt.savefig('plots/uk_electricity_demand_daily.png')
plt.show()

# Perform Augmented Dickey-Fuller (ADF)test to check for stationarity
def adf_test(series):
	is_stationary = False
	result = adfuller(series.dropna())
	print(f'ADF Statistic: {result[0]}')
	print(f'p-value: {result[1]}')
	if result[1] <= 0.05:
		is_stationary = True
		print("The series is stationary.")		
	else:		
		print("The series is NOT stationary, differencing is required.")
	return is_stationary 
   
# Use StepwiseContext to estimate the arima model order   
def evaluate_models(dataset, max_p=None, max_d=None, max_q=None, 
					is_seasonal=False, seasonal_p=None, stepwise=False):				
	with StepwiseContext():
		best_model = auto_arima(
			dataset,
			start_p=0, max_p=max_p,   # AR terms
			start_q=0, max_q=max_q,   # MA terms
			start_d=0, max_d=max_d,    # Auto-detect differencing
			simple_differencing=True,  # Use simple differencing
			seasonal=is_seasonal,       # Seasonal ARIMA
       		m=seasonal_p,           # Seasonal period, energy demand has weekly seasonality
			stepwise=stepwise,        # Stepwise search 
			suppress_warnings=True,
			error_action="ignore",
			cache_size=1,
			trace=True
   		)
		return best_model
		

# Define function to fit and evaluate the model
def fit_and_evaluate_arima_model(X_train, X_test, arima_order, seasonal_order=None):
	history = [x for x in X_train]
	predictions = list()
	for t in range(len(X_test)):
		model = ARIMA(history, order= arima_order, seasonal_order=seasonal_order) 
		model_fit = model.fit()
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = X_test[t]
		history.append(obs) # Update the data for the next model fit
		print('predicted=%f, expected=%f' % (yhat, obs))
	return model_fit, predictions

# Perform ADF test to check station
is_stationary = adf_test(data)
max_d = 0
if not is_stationary:
	data = data.diff().dropna()
	max_d = max_d + 1
	is_stationary = adf_test(data)

# Specify if the data is seasonal and seasonal period
is_seasonal = True
seasonal_p = None
if is_seasonal:
	seasonal_p = 48

# Evaluate arima model to determine the order
best_model = evaluate_models(data, max_p=5, max_d=max_d, max_q=5, 
							is_seasonal=is_seasonal,  seasonal_p=seasonal_p, stepwise=True)

# Summary of best ARIMA model
print(best_model.summary())

# Split the dataset into train
X = data.values
size = int(len(X) * 0.8)
X_train, X_test = X[0:size], X[size:len(X)]
"""
# Split data into train and test
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Fit ARIMA model
model = ARIMA(X, order=(1,1,1))
model_fit = model.fit()
# Forecast
forecast = model_fit.forecast(steps=len(X_test))

# Plot the results with specified colors
plt.figure(figsize=(14,7))
plt.plot(train.index, X_train, label='Train', color='#203147')
plt.plot(test.index, X_test, label='Test', color='#01ef63')
plt.plot(test.index, forecast, label='Forecast', color='orange')
plt.title('Solar Generation Forecast')
plt.xlabel('Date')
plt.ylabel('Solar Generation')
plt.legend()
plt.show()
"""
# Extract best parameters
# Print the parameters of the best model
print("The order of best model is:", best_model.order, best_model.seasonal_order)
model_fit, predictions = fit_and_evaluate_arima_model(X_train, X_test, best_model.order , best_model.seasonal_order)

# Evaluate forecasts
rmse = sqrt(mean_squared_error(X_test, predictions))
print('Test RMSE: %.3f' % rmse)

# Line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()

# Density plot of residuals
residuals.plot(kind='kde')
plt.show()

# Summary stats of residuals
print(residuals.describe())

# Print model summary
print(model_fit.summary())

# Forecast next 30 days demand
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)
print(forecast)


# Plot forecasts against actual outcomes
x = np.arange(X_test.shape[0])
plt.figure(figsize=(12, 6))
plt.plot(pd.date_range(data[size:len(X)].index[-1], freq= 'H', periods=(len(X) - size)), X_test, label="Actual", marker='x')
plt.plot(pd.date_range(data[size:len(X)].index[-1], freq= 'H', periods=(len(X) - size)), predictions, label="Prediction")

#plt.plot(predictions, label="Forecast", color='red')
plt.title("UK Embedded Solar Generation Forecast")
plt.xlabel("Date ")
plt.ylabel("Solar Generation (MW)")
plt.legend()
plt.show()

from sklearn.metrics import r2_score,  mean_squared_error, mean_absolute_error
r2 = r2_score(X_test, predictions)
mse = mean_squared_error(X_test, predictions)
rmse = np.sqrt(mse)
rmse = float("{:.4f}".format(rmse))         
mae = mean_absolute_error(X_test, predictions)

print(f'R2: {r2:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}')
