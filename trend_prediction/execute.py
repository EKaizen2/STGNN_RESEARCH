from os import path, remove, makedirs
import numpy as np
from run.execute import run as executer
from pandas import DataFrame
from specs import parameters as generic_parameters
from copy import deepcopy
import sys
import click
from re import sub as regex_replace
from datetime import datetime
import os
import pandas as pd

def _setup_experiment(parameters, data, feature):
    params = deepcopy(parameters)
    params['feature_type'] = feature
    if data == 'nyse':
        params['raw_data_filename'] = 'datasets/NYSE.csv'
        params['trend_filename'] = 'processed_data/nyse.pkl'
        params['main_time_series'] = 'Close'
        params['test_size'] = 1001
        params['train_size'] = 4008
        params['window_size'] = 1
        params['target_name'] = 'trend'
        params['local_window'] = 21
    elif data == 'jse':
        # Set up base parameters
        params['main_time_series'] = 'ASPEN'
        params['test_size'] = 1
        params['train_size'] = 899
        params['window_size'] = 1
        params['target_name'] = 'trend'
        params['local_window'] = 21
        
        # For GWN model, use both raw data and processed files
        estimator = params.get('estimator', '').lower() if params.get('estimator') else ''
        if estimator == 'gwn':
            params['raw_data_filename'] = 'datasets/JSE.csv'
            if feature == 'pointdata':
                params['trend_filename'] = 'processed_data/jse_raw.pkl'
            else:  # trend or trend_local_data
                params['trend_filename'] = 'processed_data/jse_trends.pkl'
        else:
            # For LSTM and other models, use the original jse.pkl
            params['raw_data_filename'] = 'datasets/JSE.csv'
            params['trend_filename'] = 'processed_data/jse.pkl'
    return params

def execute(params, algorithm, n_runs, feature, verbose):
    algorithm = algorithm.lower()
    if algorithm == 'lstm':
        algorithm = f"bohb_lstm_regressor"
    elif algorithm == 'gwn':
        algorithm = algorithm  # Keep as is
    else:
        raise ValueError(f"Only 'lstm' and 'gwn' algorithms are supported")

    params['estimator'] = algorithm
    test_scores, val_scores, train_scores, train_times = [], [], [], []
    
    for run in range(n_runs):
        results = executer(**params)
        
        test_score = np.sqrt(results['test_score'])
        val_score = np.sqrt(results['train_val_info']['test_score'])
        train_score = np.sqrt(-1*results['train_val_info']['train_score'])
        train_time = np.sum(results['train_val_info']['fit_time'])
        
        test_scores.append(test_score)
        val_scores.append(val_score)
        train_scores.append(train_score)
        train_times.append(train_time)
        
        if verbose:
            print('Run ', run + 1)
            print('Test RMSE: ', test_score)
            print('Validation RMSE: ', val_score)
            print("Training RMSE: ", train_score)
            print('Training time: ', train_time)
        
    test_scores = np.array(test_scores).reshape(n_runs, -1)
    train_scores = np.array(train_scores)
    val_scores = np.array(val_scores)
    train_times = np.array(train_times)
    return DataFrame({'Mean Test RMSE': np.mean(test_scores, axis=0),
                      'Std Test RMSE': np.std(test_scores, axis=0),
                      'Mean Validation RMSE': np.mean(val_scores),
                      'Std Validation RMSE': np.std(val_scores),
                      'Mean Train RMSE': np.mean(train_scores),
                      'Std Train RMSE': np.std(train_scores),
                      'Training Time (Sec)': np.mean(train_times),
                      'Std Training Time': np.std(train_times)}, index=["slope", "duration"])


@click.command()
@click.option('--dataset', default="voltage", help='The dataset to be used from the set {voltage, methane, nyse, jse}')
@click.option('--feature', default="pointdata", help='The feature type to be used from the set '
                                                '{pointdata, trend, trend_local_data}.')
@click.option('--algorithm', default="rf", help='The algorithm to be used from the set '
                                                '{trenet, lstm, cnn, mlp, rf, gbm, svr}.')
@click.option('--nruns', default=1, help='The number of runs.', type=int)
@click.option('--verbose/--silent', default=True, help='Whether or not to print scores and training time per run')
@click.option('--save/--temporary', default=True, help='Whether or not to save the results to disk as CSV.')
@click.option('--output', default=None, help='The directory of the CSV result if save it True.')
def main(dataset, feature, algorithm, nruns, verbose, save, output):
    start_time = regex_replace('\.|:', '-', str(datetime.now()))

    experiment_params = _setup_experiment(parameters=generic_parameters, data=dataset, feature=feature)
    result = execute(params=experiment_params, algorithm=algorithm, n_runs=nruns, feature=feature, verbose=verbose)
    print(result)

    if save:
        filename = f"experiment_result_dataset_{dataset}_feature_{feature}_" \
                   f"algorithm_{algorithm}_nruns_{nruns}_started_at_{start_time}.csv"
        if output is not None:
            # Create output directory if it doesn't exist
            makedirs(output, exist_ok=True)
            result.to_csv(path.join(output, filename))
        else:
            # Create default results directory if it doesn't exist
            default_dir = path.join("trend_prediction", "results", "manual")
            makedirs(default_dir, exist_ok=True)
            result.to_csv(path.join(default_dir, filename))


if __name__ == '__main__':
    main()


