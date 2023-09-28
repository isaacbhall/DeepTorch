import numpy as np
import optuna
from easyesn import PredictionESN
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yfinance as yf
import functools
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

@functools.lru_cache(maxsize=None)  # Unlimited cache size    

# Assume fetch_data(start, end) is a function that fetches the required data
# between the specified start and end times.

def fetch_btc_data(start_hours_ago, end_hours_ago):
    btc = yf.Ticker("BTC-USD")
    end_date = btc.history(period="1d").index[-1]  # Get the last date available
    start_date = end_date - pd.Timedelta(hours=start_hours_ago)  # Calculate the start date
    end_date = end_date - pd.Timedelta(hours=end_hours_ago)  # Calculate the end date
    
    # Ensure start_date is before end_date
    if start_date > end_date:
        start_date, end_date = end_date, start_date  # Swap the dates if they are in the wrong order
    
    data = btc.history(start=start_date, end=end_date, interval="1h")['Close'].values.reshape(-1, 1)
    return data

def objective(trial, data):
    # Hyperparameters to be optimized
    reservoir_size = trial.suggest_int('reservoir_size', 50, 500)
    leaking_rate = trial.suggest_float('leaking_rate', 0.1, 1.0)
    regression_parameter = trial.suggest_float('regression_parameter', 0.1, 1.0)
    
    # Create a Time Series Split object
    tscv = TimeSeriesSplit(n_splits=5)  # Adjust the number of splits as needed
    
    validation_losses = []  # to store the validation loss of each split
    
    for train_index, val_index in tscv.split(data):
        train_data, val_data = data[train_index], data[val_index]
        
        # Build the ESN
        esn = PredictionESN(n_input=1,
                            n_output=1,
                            n_reservoir=reservoir_size,
                            leakingRate=leaking_rate,
                            regressionParameters=regression_parameter)
        
        # Train the ESN
        esn.fit(train_data, train_data, transientTime=0, verbose=0)
        
        # Validate the ESN
        val_output = esn.predict(val_data)
        val_loss = np.sqrt(mean_squared_error(val_output, val_data))
        validation_losses.append(val_loss)
        
    best_val_loss = np.inf
    best_esn = None

    # Split your data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)

    for train_size in np.linspace(0.1, 1.0, 10):  # Vary the amount of training data from 10% to 100%
        current_train_data = train_data[:int(len(train_data) * train_size)]
        
        # Build the ESN
        esn = PredictionESN(n_input=1,
                            n_output=1,
                            n_reservoir=reservoir_size,
                            leakingRate=leaking_rate,
                            regressionParameters=regression_parameter)
        
        # Train the ESN
        esn.fit(current_train_data, current_train_data, transientTime=0, verbose=0)
        
        # Validate the ESN
        val_output = esn.predict(val_data)
        val_loss = np.sqrt(mean_squared_error(val_output, val_data))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_esn = esn  # Assume esn can be copied like this
        else:
            # If the validation loss didn't improve, stop training
            break

    return best_val_loss  # return the best validation loss found
    
    return np.mean(validation_losses)  # return the average validation loss across all splits


def train_and_test_esn(train_hours, test_hours_ago, pdf_pages):
    # Fetch training data
    data = fetch_btc_data(train_hours, 0)  # Assume 0 is the current time
    
    # Optimize hyperparameters using the modified objective function
    study = optuna.create_study(direction='minimize', sampler=TPESampler(), pruner=HyperbandPruner())
    study.optimize(lambda trial: objective(trial, data), n_trials=100, n_jobs=-1)
    
    # Build and train the ESN with optimized hyperparameters
    best_params = study.best_trial.params
    esn = PredictionESN(n_input=1,
                        n_output=1,
                        n_reservoir=best_params['reservoir_size'],
                        leakingRate=best_params['leaking_rate'],
                        regressionParameters=best_params['regression_parameter'])
    esn.fit(data, data, transientTime="Auto", verbose=0)
    
    for start_hours_ago in range(test_hours_ago, 100, -100):  # Adjust as needed
        end_hours_ago = start_hours_ago - 100
        test_data = fetch_btc_data(start_hours_ago, end_hours_ago)
        test_output = esn.predict(test_data)
        test_loss = sqrt(mean_squared_error(test_output, test_data))
        print(f'Train Hours: {train_hours}, Test Hours Ago: {start_hours_ago}, Test Loss: {test_loss}')
        
        # Plot the results
        plt.figure()
        plt.plot(test_data, label='Actual Data')
        plt.plot(test_output, label='ESN Output')
        plt.legend()
        plt.title(f'Train Hours: {train_hours}, Test Hours Ago: {start_hours_ago}')

        # Create a text annotation with the hyperparameters and loss
        annotation_text = (
            f"Test Loss: {test_loss}\n"
            f"Reservoir Size: {best_params['reservoir_size']}\n"
            f"Leaking Rate: {best_params['leaking_rate']}\n"
            f"Regression Parameter: {best_params['regression_parameter']}"
        )
        plt.annotate(
            annotation_text,
            xy=(0.05, 0.05),
            xycoords='axes fraction',
            fontsize=10,
            ha='left',
            va='bottom'
        )

        # Save the plot to the PDF
        pdf_pages.savefig()
        plt.close()

# Create a PDF to save the results
with PdfPages('esn_results.pdf') as pdf_pages:
    # Loop over the specified training set sizes and testing periods
    for train_hours in range(1000, 0, -100):
        train_and_test_esn(train_hours, 5000, pdf_pages)