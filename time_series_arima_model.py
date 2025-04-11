# We will use ARIMA model to forecast the solar power generation in the UK. 
# ARIMA is a time series forecasting model that uses past data to predict future values.  
# The model has three main parameters: p, d, and q.
# p: The number of lag observations included in the model (lag order).
# d: The number of times that the raw observations are differenced (degree of differencing).
# q: The size of the moving average window (order of moving average).
# We will use the auto_arima function from the pmdarima library to automatically find the best parameters for the ARIMA model.
# We will then fit the ARIMA model to the data and forecast the solar generation and test the accuracy of the model.
# Finally, we will plot the forecasted generation along with the test data to visualize the results.

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima, StepwiseContext
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import r2_score,  mean_squared_error, mean_absolute_error
from pandas.plotting import autocorrelation_plot
from math import sqrt
from pmdarima.arima.utils import nsdiffs
import pickle
import joblib

import warnings
warnings.filterwarnings("ignore")

# Load dataset and combine all years data into one dataset
year = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
df = pd.read_csv('data/UK_data/demanddata_2018.csv', parse_dates=['SETTLEMENT_DATE'], index_col='SETTLEMENT_DATE')

#Use only 2018 to 2025 data to reduce the size of the dataset for ARIMA model
for i in range(len(year)):
    data_path = f'data/UK_data/demanddata_{year[i]}.csv'    
    df_year = pd.read_csv(data_path, parse_dates=['SETTLEMENT_DATE'], index_col='SETTLEMENT_DATE') 
    df = pd.concat([df, df_year], ignore_index=False)

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

# Select the solar generation column for analysis
df['EMBEDDED_SOLAR_GENERATION'] = df['EMBEDDED_SOLAR_GENERATION'].astype('float32')
df_solar = df['EMBEDDED_SOLAR_GENERATION']  # Embedded solar is the column for solar generation in the UK
print(df_solar.head())

"""
# Select the wind generation column for analysis
df['EEMBEDDED_WIND_GENERATION'] = df['EMBEDDED_WIND_GENERATION'].astype('float32')
df_wind = df['EMBEDDED_WIND_GENERATION']  # Embedded solar is the column for solar generation in the UK
print(df_wind.head())
"""
# Resample to average values to get data at weekly data to reduce the size of the dataset
data = df_solar.resample('W').mean()

# Perform autocorrelation plot
autocorrelation_plot(data)
plt.show()

# Visualise acf and pacf
plot_acf(data)
plot_pacf(data)
plt.show()

# Visualise the seasonal decomposition of the data
decomposition = seasonal_decompose(data, model='additive') 
decomposition.plot()
plt.show()

seasonal_p = 52 # Seasonal period is 52 weeks
D = 1  # Seasonal differencing term

# Estimate number of seasonal differences using a Canova-Hansen test
#D = nsdiffs(data, m= seasonal_p, max_D=3, test='ch')  
print(f"Estimated seasonal differencing term (D): {D}")

# Split the dataset into train and test set
X = data.values
size = int(len(X) * 0.5)
X_train, X_test = X[0:size], X[size:len(X)]
data_train, data_test = data.iloc[0:size], data.iloc[size:len(X)]
data.to_csv('data/UK_data/solar_data.csv')

# Plot the solar generation data
plt.figure(figsize=(12, 6))
plt.plot(pd.date_range(data.index[-1], freq= 'D', periods=(data.shape[0])), data, label="Solar Generation", marker='x')
plt.title("UK Solar Generation Over Time")
plt.xlabel("Date")
plt.ylabel(" Solar Generation (MW)")
plt.legend()
plt.show()
"""
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

is_stationary = adf_test(data)

# Perform differencing if data is Not stationary to make it stationary and determine max_d
max_d = 0
if not is_stationary:
	max_d = max_d + 1	
	data_diff = data.diff().dropna()
	seasonal_data_diff = data_diff.diff(seasonal_p).dropna()	
	plot_acf(seasonal_data_diff, lags=seasonal_p)
	plot_pacf(seasonal_data_diff, lags=seasonal_p)
	plt.show()	
	
# Evaluate arima model to determine the order
auto_model = auto_arima(data, start_p=0, start_q=0,
    max_p=3, d=max_d, max_d=2, max_q=3,
    start_P=1, D=D, start_Q=0, max_P=3, max_D=3,
    max_Q=3, m = seasonal_p, seasonal=True, 
    stationary=False,
    error_action='warn', trace=True,
    suppress_warnings=True, stepwise=True,
    random_state=20, n_fits=50)

# Summary of best ARIMA model
print(auto_model.summary())
arima_order = auto_model.order
seasonal_order = auto_model.seasonal_order
"""
#r2 = 0.83
#arima_order = (2,2,2)
#seasonal_order = (2,1,2,seasonal_p)	

#r2 = 0.83
#arima_order = (1,1,1)
#seasonal_order = (1,1,1,seasonal_p)	

arima_order = (1,1,1)
seasonal_order = (2,1,0,seasonal_p)

# Fit ARIMA model
model = SARIMAX(X_test,  order=arima_order, seasonal_order=seasonal_order) 
#model = SARIMAX(data,  order=arima_order, seasonal_order=seasonal_order) 
model_fit = model.fit()

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

# Forecast solar generation using the test data
forecast_steps = len(X) - size 
forecast = model_fit.forecast(steps=forecast_steps)
print(forecast)

forecast_steps = 184
last_date = pd.to_datetime("2025-01-19")  # The last date in the dataset
future_dates = pd.date_range(last_date, periods=forecast_steps + 1, freq='W')[1:]

forecast_result = model_fit.get_forecast(steps=forecast_steps)
forecast = forecast_result.predicted_mean

# Plot the results with specified colors
plt.figure(figsize=(14,7))
#plt.plot(data.iloc[:size].index, X_train, label='Train Solar Generation Data', color='#203147')
plt.plot(data.iloc[size:].index, X_test, label='Test Solar Generation Data', color='#01ef63')
#plt.plot(data.iloc[size:].index, forecast, label='Forecast Solar Generation Data', color='orange')

plt.plot(future_dates, forecast, label="Solar Generation Forecast", color="red")
#plt.plot(data.index, data.values, label='Forecast Solar Generation Data', color='orange')
#plt.plot(pd.date_range(data.index[-1], freq= 'D', periods=(data.shape[0])), data, label="Solar Generation", marker='x')

plt.title("UK Embedded Solar Generation Forecast")
plt.xlabel("Date")
plt.ylabel("Solar Generation (MW)")
plt.legend()
plt.savefig('plots/uk_solar_generation.png')
plt.show()

# Evaluate forecasts
rmse = sqrt(mean_squared_error(X_test, forecast))
print('Test RMSE: %.3f' % rmse)

r2 = r2_score(X_test, forecast)
mse = mean_squared_error(X_test, forecast)
rmse = np.sqrt(mse)
rmse = float("{:.4f}".format(rmse))         
mae = mean_absolute_error(X_test, forecast)

# Print evaluation metrics
print(f'R2: {r2:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}')

# Save model to disk
joblib.dump(model_fit, "results/solar_arima_model.pkl")

"""
# Save model to disk
import pickle
with open('results/solar_generation_arima_model1.pkl', 'wb') as f:
   pickle.dump(model_fit, f)
"""