# More imports
from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import os
import sys
sys.path.append("../")
from utils.utils import get_data, make_dir


def lstm(T):
    ### LSTM model
    i = Input(shape=(T, 1))
    x = LSTM(5)(i)
    x = Dense(1)(x)
    model = Model(i, x)
    model.compile(
            loss='mse',
            optimizer=Adam(lr=0.1)
    )

    return model


def plot_loss(model, ticker):
    plt.plot(model.history['loss'], label='loss')
    plt.plot(model.history['val_loss'], label='val_loss')
    plt.title(f"{ticker} Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Episodes")
    plt.legend()
    plt.show()


def plot_predictions(predictions, Y, ticker):
    plt.plot(Y, label='targets')
    plt.plot(predictions, label='predictions')
    plt.title(f"{ticker} Model Forecast")
    plt.ylabel("Stock Price (Normalized)")
    plt.xlabel("Episodes")
    plt.legend()
    plt.show()


def onestep_forecast(model, T, X, Y, N, ticker):
    # One-step forecast using true targets
    outputs = model.predict(X)
    predictions = outputs[:,0]
    plot_predictions(predictions, Y, ticker)
    return predictions


def multistep_forecast(model, T, X, Y, N, ticker):
    # Multi-step forecast
    validation_target = Y[-N//2:]
    validation_predictions = []

    # first validation input
    last_x = X[-N//2] # 1-D array of length T

    while len(validation_predictions) < len(validation_target):
        p = model.predict(last_x.reshape(1, T, 1))[0,0] # 1x1 array -> scalar
        
        # update the predictions list
        validation_predictions.append(p)
        
        # make the new input
        last_x = np.roll(last_x, -1)
        last_x[-1] = p

    plot_predictions(validation_predictions, validation_target, ticker)
    return validation_predictions


def lstm_forecast(models_folder, ticker, data, forecast_window, mode):

    #data = get_data("../data")
    series = data[ticker].values.reshape(-1, 1)

    scaler = StandardScaler()
    scaler.fit(series[:len(series) // 2])
    series = scaler.transform(series).flatten()

    ### Build the dataset
    # Use T past values to predict the next value
    T = forecast_window  # Sliding window
    D = 1   # Dimension
    X = []  # Data
    Y = []  # Target label
    for t in range(len(series) - T):
        x = series[t:t+T]
        X.append(x)
        y = series[t+T]
        Y.append(y)

    X = np.array(X).reshape(-1, T, 1) # Now the data should be N x T x D
    Y = np.array(Y)
    N = len(X)
    # print("X.shape", X.shape, "Y.shape", Y.shape)
    
    ### LSTM model
    model = lstm(T)

    # If there is existing saved weights
    if os.path.exists(f'{models_folder}/lstm/{ticker}.h5'):
        print(f"\nLoading {ticker} Saved Weights")

        # Load the saved model weights
        model.load_weights(f'{models_folder}/lstm/{ticker}.h5')

        if mode.lower() == "one":
            predictions = scaler.inverse_transform(onestep_forecast(model, T, X, Y, N, ticker))
            print(f"Successfully retrieved {ticker} Onestep predictions!")
            return predictions
        
        predictions = scaler.inverse_transform(multistep_forecast(model, T, X, Y, N, ticker))
        print(f"Successfully retrieved {ticker} Multistep predictions!")
        return predictions

    # Else train new model
    else:
        print("\n{} Stock Price Forecast Training\n".format(ticker))
        # Train the RNN (nn -> neural network)
        nn = model.fit(
                X[:-N//2], Y[:-N//2],
                epochs=100,
                validation_data=(X[-N//2:], Y[-N//2:])
        )
        make_dir(f'{models_folder}/lstm')
        model.save(f'{models_folder}/lstm/{ticker}.h5')
        
        #plot_loss(nn, ticker)

        # If mode is one, then perform onestep forecast
        if mode.lower() == "one":
            # scaler.inverse_transform transforms the scaled output back to unscaled value
            predictions = scaler.inverse_transform(onestep_forecast(model, T, X, Y, N, ticker))
            print(f"Successfully trained {ticker} models & saved!")
            return predictions
    
        # Else, perform multistep forecast
        # scaler.inverse_transform transforms the scaled output back to unscaled value
        predictions = scaler.inverse_transform(multistep_forecast(model, T, X, Y, N, ticker))
        print(f"Successfully trained {ticker} models & saved!")
        return predictions
