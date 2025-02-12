#import the libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D
import keras.optimizers

from tensorflow.keras.layers import LSTM, Dropout

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,  mean_squared_error
from data_preprocessor import read_database_data, create_sequences, invert_transform

import warnings
warnings.filterwarnings("ignore")
generation_data = pd.read_csv('data/Plant_2_Generation_Data.csv')
weather_data = pd.read_csv('data/Plant_2_Weather_Sensor_Data.csv')

print(generation_data.head())
generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'],format = 'mixed') # '%Y-%m-%d %H:%M')
weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'],format = 'mixed') #'%Y-%m-%d %H:%M:%S')

#merge the generator and weather data
df_solar = pd.merge(generation_data.drop(columns = ['PLANT_ID']), weather_data.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')
df_solar.sample(5).style.background_gradient(cmap='cool')

# adding separate date, time, day, month ans weekday columns
df_solar["DATE"] = pd.to_datetime(df_solar["DATE_TIME"]).dt.date
df_solar["TIME"] = pd.to_datetime(df_solar["DATE_TIME"]).dt.time
df_solar['DAY'] = pd.to_datetime(df_solar['DATE_TIME']).dt.day
df_solar['MONTH'] = pd.to_datetime(df_solar['DATE_TIME']).dt.month
df_solar['WEEK'] = pd.to_datetime(df_solar['DATE_TIME']).dt.weekday

df_solar.info()

# Add hours and minutes for ml models
df_solar['HOURS'] = pd.to_datetime(df_solar['TIME'],format='%H:%M:%S').dt.hour
df_solar['MINUTES'] = pd.to_datetime(df_solar['TIME'],format='%H:%M:%S').dt.minute
df_solar['TOTAL MINUTES PASS'] = df_solar['MINUTES'] + df_solar['HOURS']*60

# Add date as string column
df_solar["DATE_STRING"] = df_solar["DATE"].astype(str) # add column with date as string
df_solar["HOURS"] = df_solar["HOURS"].astype(str)
df_solar["TIME"] = df_solar["TIME"].astype(str)

#Check the first few rows of the data
print(df_solar.head())

# Check the data types and missing values
print(df_solar.info())

# Visualize the DC power, the target data
plt.plot(df_solar.index, df_solar["DC_POWER"], label="Time Series Data for Stock Price Close Selected as Target")
plt.legend()
#plt.show()

# Select the relevant features for prediction
features = ['DAILY_YIELD','TOTAL_YIELD','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION','DC_POWER','AC_POWER']

df_solar = df_solar[features]
corr = df_solar.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()
df_solar.describe()

# Drop DAILY_YIELD and TOTAL_YIELD columns because they are highly correlated with DC_POWER
df_solar = df_solar.drop(columns=["DAILY_YIELD", "TOTAL_YIELD"])

# Normalize the data
data = df_solar.values
scaler = MinMaxScaler()
#scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define the sequence length
seq_length = 60

# Create sequences
table_column_index = 3  # Index of the 'DC_POWER' column as the target
X, y = create_sequences(scaled_data, seq_length, table_column_index)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_train_cnn_model(X_train, y_train, epochs=100, batch=24, lr=0.0003):
    # Define the CNN model    
    adam = keras.optimizers.Adam(lr)
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=adam)
    #model.compile(optimizer='adam', loss='mean_squared_error')

    # Model summary
    model.summary()   

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch, verbose=2)

    return model

# Build and train the lstm model
def build_train_lstm_model(X_train, y_train, epochs=100, batch=24):
    # Initialize the model
    model = Sequential()

    # Add LSTM layers with Dropout regularization
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    # Add Dense layer
    model.add(Dense(units=25))
    model.add(Dense(units=1))  # Output layer, predicting the 'solar power generation'
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=batch, epochs=epochs)

    return model

def build_train_lstm_pytorch_model(X_train, y_train, epochs=100):    
    import torch.nn as nn
    import torch.optim as optim

    # Convert data to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)      
    
    # Define LSTM model
    class LSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            out = self.fc(hn[-1])
            return out

    # Model, loss function, and optimizer
    input_dim = X_train.shape[2]
    hidden_dim = 64
    output_dim = 1

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    train_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train).squeeze()
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model

def invert_transform(data, shape, column_index, scaler):
    dummy_array = np.zeros((len(data), shape))
    dummy_array[:, column_index] = data.flatten()
    return scaler.inverse_transform(dummy_array)[:, column_index]

models = {
    'LSTM': build_train_lstm_model(X_train, y_train, 120, 24),
    'CNN': build_train_cnn_model(X_train, y_train, 120, 24),
  #  'PTLSTM': build_train_lstm_pytorch_model(X_train, y_train),    
}

import torch
rmse_scores = dict()
predictions = dict()
for name, model in models.items():
    if name == 'PTLSTM':        
        model.eval()
        X_test = torch.FloatTensor(X_test)
        with torch.no_grad():            
            y_pred = model(X_test).squeeze()   
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))       
        predictions[name] = y_pred       
        rmse = float("{:.4f}".format(rmse))
        rmse_scores[name] = rmse
        print(f"{name} RMSE: {rmse}") 
    else:        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))           
        predictions[name] = y_pred        
        rmse = float("{:.4f}".format(rmse))
        rmse_scores[name] = rmse
        print(f"{name} RMSE: {rmse}")

 # Plot the RMSE scores       
def add_comparison_plots(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')

plt.bar(list(rmse_scores.keys()), list(rmse_scores.values()), color ='red')

# Add labels
add_comparison_plots(list(rmse_scores.keys()), list(rmse_scores.values()))
plt.xlabel('') 
plt.ylabel('RMSE') 
plt.title('Models') 
plt.show()

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True,figsize=(22,12))
fig.suptitle('Solar DC Power Predictions')
fig.supxlabel('Time')
fig.supylabel('Solar DC Power')
#ax1, ax2 = axes[0]
#ax3, ax4 = axes[1]
ax1 = axes[0]
ax2 = axes[1]
ax3 = axes[2]


y_test = invert_transform(y_test, X_train.shape[2], 0, scaler)

# Compare the predictions wtih the actual Dc power
def add_plots(x,y):
    for i in range(len(x)):            
        if x[i] == 'LSTM':
            y[i] = invert_transform(y[i], X_train.shape[2], 0, scaler)               
            ax1.plot(y[i], label=x[i])                       
        elif x[i] == 'CNN':
           y[i] = invert_transform(y[i], X_train.shape[2], 0, scaler)
           ax2.plot(y[i], label=x[i])                                
        else:   
            y[i] = scaler.inverse_transform(np.hstack([np.zeros((y[i].shape[0], scaled_data.shape[1] - 1)), y[i].numpy().reshape(-1, 1)]))[:, -1]
            ax3.plot(y[i], label=x[i]) 
              
    ax1.plot(y_test, label='Actual DC Power')
    ax2.plot(y_test, label='Actual DC Power')
    ax3.plot(y_test, label='Actual DC Power')   
    
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')   
   
    ax1.set_title('LSTM Predictions')
    ax2.set_title('CNN Predictions')
    ax3.set_title('PTLSTM Predictions')   

add_plots(list(predictions.keys()), list(predictions.values()))  

plt.tight_layout()
plt.show()