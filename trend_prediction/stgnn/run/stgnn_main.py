import click
from os import path, makedirs
from datetime import datetime
from re import sub as regex_replace
from pandas import DataFrame
import numpy as np
from stgnn.run.stgnn_execute import run_stgnn
from specs import parameters as generic_parameters
from copy import deepcopy
import json
import os

def load_hyperparameters(model_type='gwn'):
    """Load hyperparameters from the JSON file."""
    config_path = os.path.join('hyperparameters', 'hyperparameters.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config[model_type]

def validate_parameters(params, dataset):
    """Validate that all required parameters are present."""
    required_model_params = {
        'input_dim', 'output_dim', 'num_nodes', 'dropout', 'gcn_bool',
        'adapt_adj', 'hidden_dim', 'residual_channels', 'dilation_channels',
        'skip_channels', 'end_channels', 'kernel_size', 'blocks', 'layers'
    }
    required_training_params = {
        'device', 'learning_rate', 'prediction_horizon', 'batch_size',
        'n_epochs', 'validation_fraction', 'test_fraction'
    }
    
    missing_model = required_model_params - set(params.get('model_params', {}).keys())
    missing_training = required_training_params - set(params.get('training_params', {}).keys())
    
    if missing_model or missing_training:
        raise ValueError(
            f"Missing required parameters:\n"
            f"Model parameters: {missing_model}\n"
            f"Training parameters: {missing_training}"
        )
    
    if dataset not in params.get('dataset_params', {}):
        raise ValueError(f"No configuration found for dataset: {dataset}")

def _setup_stgnn_experiment(parameters, data, feature):
    """Setup experiment parameters specifically for STGNN models."""
    # Load hyperparameters
    config = load_hyperparameters('gwn')
    
    # Validate parameters
    validate_parameters(config, data)
    
    # Start with dataset-specific parameters
    params = deepcopy(config['dataset_params'][data])
    
    # Add model parameters
    params.update(config['model_params'])
    
    # Add training parameters
    params.update(config['training_params'])
    
    # Override feature type if specified
    if feature:
        params['feature_type'] = feature
    
    # Print configuration for debugging
    print("\nExperiment Configuration:")
    print("------------------------")
    print("Dataset:", data)
    print("Feature type:", params['feature_type'])
    print("Model parameters:", {k: v for k, v in params.items() if k in config['model_params']})
    print("Training parameters:", {k: v for k, v in params.items() if k in config['training_params']})
    print("Dataset parameters:", {k: v for k, v in params.items() if k not in config['model_params'] and k not in config['training_params']})
    print("------------------------\n")
    
    return params

def execute_stgnn(params, algorithm, n_runs, feature, verbose):
    """Execute STGNN model training and evaluation."""
    algorithm = algorithm.lower()
    if algorithm not in ['gwn']:
        raise ValueError(f"Only 'gwn' algorithm is supported for STGNN models")

    params['estimator'] = algorithm
    test_scores = {'rmse': [], 'mae': [], 'mape': []}
    val_scores = {'rmse': [], 'mae': [], 'mape': []}
    train_scores = {'rmse': [], 'mae': [], 'mape': []}
    train_times = []
    
    for run in range(n_runs):
        results = run_stgnn(**params)
        
        # Extract metrics from results
        test_rmse = np.mean([score['rmse'] for score in results['test_scores']])
        test_mae = np.mean([score['mae'] for score in results['test_scores']])
        test_mape = np.mean([score['mape'] for score in results['test_scores']])
        
        val_rmse = np.mean([score['rmse'] for score in results['val_scores']])
        val_mae = np.mean([score['mae'] for score in results['val_scores']])
        val_mape = np.mean([score['mape'] for score in results['val_scores']])
        
        train_rmse = np.mean([score['rmse'] for score in results['train_scores']])
        train_mae = np.mean([score['mae'] for score in results['train_scores']])
        train_mape = np.mean([score['mape'] for score in results['train_scores']])
        
        train_time = np.sum(results['fit_time'])
        
        # Store metrics
        test_scores['rmse'].append(test_rmse)
        test_scores['mae'].append(test_mae)
        test_scores['mape'].append(test_mape)
        
        val_scores['rmse'].append(val_rmse)
        val_scores['mae'].append(val_mae)
        val_scores['mape'].append(val_mape)
        
        train_scores['rmse'].append(train_rmse)
        train_scores['mae'].append(train_mae)
        train_scores['mape'].append(train_mape)
        
        train_times.append(train_time)
        
        if verbose:
            print('Run ', run + 1)
            print('Test  - RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.2f}%'.format(test_rmse, test_mae, test_mape))
            print('Val   - RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.2f}%'.format(val_rmse, val_mae, val_mape))
            print('Train - RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.2f}%'.format(train_rmse, train_mae, train_mape))
            print('Training time: {:.2f}s'.format(train_time))
    
    # Calculate mean and std for each metric
    results = {}
    for metric in ['rmse', 'mae', 'mape']:
        results[f'Mean Test {metric.upper()}'] = np.mean(test_scores[metric])
        results[f'Std Test {metric.upper()}'] = np.std(test_scores[metric])
        results[f'Mean Val {metric.upper()}'] = np.mean(val_scores[metric])
        results[f'Std Val {metric.upper()}'] = np.std(val_scores[metric])
        results[f'Mean Train {metric.upper()}'] = np.mean(train_scores[metric])
        results[f'Std Train {metric.upper()}'] = np.std(train_scores[metric])
    
    results['Mean Training Time (Sec)'] = np.mean(train_times)
    results['Std Training Time'] = np.std(train_times)
    
    return DataFrame(results, index=["slope", "duration"])

@click.command()
@click.option('--dataset', default="jse", help='The dataset to be used (currently only jse is supported for STGNN)')
@click.option('--feature', default="trend", help='The feature type to be used (trend for slope and duration)')
@click.option('--algorithm', default="gwn", help='The STGNN algorithm to be used (currently only gwn is supported)')
@click.option('--nruns', default=1, help='The number of runs.', type=int)
@click.option('--verbose/--silent', default=True, help='Whether or not to print scores and training time per run')
@click.option('--save/--temporary', default=True, help='Whether or not to save the results to disk as CSV')
@click.option('--output', default=None, help='The directory of the CSV result if save is True')
def main(dataset, feature, algorithm, nruns, verbose, save, output):
    """Main entry point for STGNN models."""
    start_time = regex_replace('\.|:', '-', str(datetime.now()))

    # Setup experiment parameters
    experiment_params = _setup_stgnn_experiment(parameters=generic_parameters, data=dataset, feature=feature)
    
    # Execute experiment
    result = execute_stgnn(params=experiment_params, algorithm=algorithm, n_runs=nruns, feature=feature, verbose=verbose)
    print(result)

    if save:
        filename = f"results{dataset}_feature_{feature}_algorithm_{algorithm}_nruns_{nruns}_started_at_{start_time}.csv"
        if output is not None:
            makedirs(output, exist_ok=True)
            result.to_csv(path.join(output, filename))
        else:
            default_dir = path.join("results", "stgnn")
            makedirs(default_dir, exist_ok=True)
            result.to_csv(path.join(default_dir, filename))

if __name__ == '__main__':
    main() 