import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
import pmdarima as pm

def optimize_arima(series, seasonal=False, m=1):
    """
    Uses auto_arima to find the best ARIMA/SARIMA parameters.
    """
    model = pm.auto_arima(series, seasonal=seasonal, m=m,
                          suppress_warnings=True, stepwise=True,
                          error_action='ignore')
    return model

def train_arima_model(train_data, order, seasonal_order=None):
    """
    Trains an ARIMA or SARIMA model.
    """
    model = ARIMA(train_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    return model_fit

def prepare_lstm_data(data, look_back=60):
    """
    Prepares data for LSTM (X=past look_back days, y=next day).
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """
    Builds a simple LSTM model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_model_metrics(y_true, y_pred):
    """
    Calculates MAE, RMSE, MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
