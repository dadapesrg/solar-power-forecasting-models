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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima, StepwiseContext
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import r2_score,  mean_squared_error, mean_absolute_error
from pandas.plotting import autocorrelation_plot
from math import sqrt

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

# Resample to average values for weekly data
data = df_solar.resample('W').mean()

# Visualise the seasonal decomposition
seasonal_p = 52 #24
decomposition=seasonal_decompose(data, model='additive', period=seasonal_p)
decomposition.plot()
plt.show()

autocorrelation_plot(data)
plt.show()

# Visualise acf and pacf
plot_acf(data)
plot_pacf(data)
plt.show()

# Split the dataset into train and test set
X = data.values
size = int(len(X) * 0.8)
X_train, X_test = X[0:size], X[size:len(X)]

# Plot the solar generation data
plt.figure(figsize=(12, 6))
plt.plot(pd.date_range(data.index[-1], freq= 'H', periods=(data.shape[0])), data, label="Solar Generation", marker='x')
plt.title("UK Solar Generation Over Time")
plt.xlabel("Date")
plt.ylabel(" Solar Generation (MW)")
plt.legend()
#plt.savefig('plots/uk_electricity_demand_daily.png')
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

# Perform differencing if data is Not stationary
max_d = 0
max_D = 1
if not is_stationary:
	max_d = max_d + 1
	max_D = max_D + 1
	data_diff = data.diff().dropna()
	seasonal_data_diff = data_diff.diff(seasonal_p).dropna()	
	plot_acf(seasonal_data_diff, lags=seasonal_p)
	plot_pacf(seasonal_data_diff, lags=seasonal_p)
	plt.show()	
	data = seasonal_data_diff
	
# Evaluate arima model to determine the order
auto_model = auto_arima(data,start_p=1,start_q=1, d=max_d, test='adf', n_jobs=-1, m=seasonal_p,D=max_D, seasonal_test='ocsb', stepwise=True, seasonal=True,trace=True)

# Summary of best ARIMA model
print(auto_model.summary())
arima_order = auto_model.order
seasonal_order = auto_model.seasonal_order
"""
#arima_order = (3,1,4) (81)
#seasonal_order = (3,1,4,seasonal_p)
#arima_order = (0,1,1) (.82)

arima_order = (0,1,1) 
seasonal_order = (0,1,1,seasonal_p)

#arima_order = (1,0,1)
#seasonal_order = (0,1,1,seasonal_p)

# Fit ARIMA model
model = SARIMAX(X_train, order= arima_order, seasonal_order=seasonal_order) 
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
predictions = model_fit.forecast(steps=forecast_steps)
print(predictions)

forecast_steps = len(X) - size + 48
forecast = model_fit.forecast(steps=forecast_steps)
print(forecast)

# Plot forecasts against actual outcomes
plt.figure(figsize=(12, 6))
plt.plot(pd.date_range(data[size:len(X)].index[-1], freq= 'H', periods=(len(X) - size)), X_test, label="Actual", marker='x')
plt.plot(pd.date_range(data[size:len(X)].index[-1], freq= 'H', periods=forecast_steps), forecast, label="Prediction")
plt.title("UK Embedded Solar Generation Forecast")
plt.xlabel("Date ")
plt.ylabel("Solar Generation (MW)")
plt.legend()
plt.show()

# Evaluate forecasts
rmse = sqrt(mean_squared_error(X_test, predictions))
print('Test RMSE: %.3f' % rmse)

r2 = r2_score(X_test, predictions)
mse = mean_squared_error(X_test, predictions)
rmse = np.sqrt(mse)
rmse = float("{:.4f}".format(rmse))         
mae = mean_absolute_error(X_test, predictions)

print(f'R2: {r2:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}')
