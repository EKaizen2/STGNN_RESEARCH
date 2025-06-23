import json
from copy import deepcopy
import os
from stgnn.preprocessing.stgnn_preprocessor import to_stgnn_supervise
from stgnn.evaluation.stgnn_validation import stgnn_time_series_validation, stgnn_testing_with_recalibration
from datasets.loader import load_data
from stgnn.models.gwn_trend import GWNTrendPredictor
import pandas as pd
import pickle

def run_stgnn(X=None, Y=None, read_data=True, raw_data=None, trends=None, raw_data_filename=None,
             trend_filename=None, main_time_series=None, na_values=None, header=0,
             main_only=False, separator=',', fill_na_method='pad',
             # preprocessing
             last_n_days=False, strength_metric='angle', angle_metric='degree',
             duration=None, return_segment=False, overlap=False, overlap_fraction=0.0,
             pla_algorithm='bottom', error=False, max_error=0.0, n_segments=10000,
             feature_type='trend', window_size=12, target_name='trend', n_classes=3,
             line_fitter='regression', lower=-0.5, upper=0.55, local_window=21, lag=30,
             n_trend=True, max_feature_size=None,
             # model parameters (from hyperparameters.json)
             input_dim=None, output_dim=None, num_nodes=None, dropout=0.3,
             gcn_bool=True, adapt_adj=True, hidden_dim=32, residual_channels=32,
             dilation_channels=32, skip_channels=256, end_channels=512,
             kernel_size=2, blocks=4, layers=2,
             # training parameters (from hyperparameters.json)
             device='cuda', learning_rate=0.001, prediction_horizon=5,
             batch_size=64, n_epochs=100, validation_fraction=0.1, test_fraction=0.2,
             # other parameters
             estimator='gwn', verbose=1):
    """
    Run STGNN models with their specific preprocessing and evaluation procedures.
    Handles multivariate time series where each column represents a node in the graph.
    """
    if X is None and Y is None:
        if read_data:
            # Load raw data - now loading all columns for multivariate analysis
            raw_data = load_data(raw_data_filename, is_pickle=False,
                               main_time_series=main_time_series, na_values=na_values,
                               header=header, main_only=main_only, separator=separator,
                               fill_na_method=fill_na_method)
            
            if not isinstance(raw_data, pd.DataFrame):
                raise ValueError("Raw data must be a pandas DataFrame with multiple columns")

            # Check if preprocessed data exists
            processed_data_dir = 'stgnn/processed_data'
            processed_data_file = os.path.join(processed_data_dir, f'stgnn_processed_data_{feature_type}_{n_segments}_segments.pkl')
            
            if os.path.exists(processed_data_file):
                if verbose:
                    print(f"Loading preprocessed data from {processed_data_file}")
                with open(processed_data_file, 'rb') as f:
                    processed_data = pickle.load(f)
                    X = processed_data['X']
                    Y = processed_data['Y']
                    if verbose:
                        print(f"Loaded data shapes - X: {X.shape}, Y: {Y.shape}")
            else:
                if verbose:
                    print("Preprocessed data not found. Processing data...")
                # Process data using STGNN-specific preprocessing
                X, Y = to_stgnn_supervise(raw_data=raw_data,
                                         feature_type=feature_type,
                                         window_size=window_size,
                                         n_segments=n_segments,
                                         n_classes=n_classes,
                                         lower=lower,
                                         upper=upper,
                                         local_window=local_window)

    # Print shapes for debugging
    if verbose:
        print("\nData shapes before model initialization:")
        print(f"X shape: {X.shape}")  # Should be [batch_size, num_nodes, num_timesteps, num_features]
        print(f"Y shape: {Y.shape}")  # Should be [batch_size, num_nodes, output_dim]

    # Only override data-dependent parameters if not explicitly set
    if input_dim is None:
        input_dim = X.shape[3]  # Number of features
    if output_dim is None:
        output_dim = 2  # Always predict [slope, duration]
    if num_nodes is None:
        num_nodes = X.shape[1]  # Number of nodes

    # Verify data shapes match model parameters
    if X.shape[3] != input_dim:
        raise ValueError(f"Input features dimension {X.shape[3]} does not match model input_dim {input_dim}")
    if X.shape[1] != num_nodes:
        raise ValueError(f"Number of nodes {X.shape[1]} does not match model num_nodes {num_nodes}")
    if Y.shape[2] != output_dim:
        raise ValueError(f"Output dimension {Y.shape[2]} does not match model output_dim {output_dim}")

    # Split data into train, validation, and test sets
    n_samples = len(X)
    train_end = int((1 - validation_fraction - test_fraction) * n_samples)
    val_end = int((1 - test_fraction) * n_samples)

    X_train = X[:train_end]
    Y_train = Y[:train_end]
    X_val = X[train_end:val_end]
    Y_val = Y[train_end:val_end]
    X_test = X[val_end:]
    Y_test = Y[val_end:]

    if verbose:
        print("\nData split sizes:")
        print(f"Train: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")

    # Create model parameters dictionary
    model_params = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'num_nodes': num_nodes,
        'dropout': dropout,
        'gcn_bool': gcn_bool,
        'adapt_adj': adapt_adj,
        'hidden_dim': hidden_dim,
        'residual_channels': residual_channels,
        'dilation_channels': dilation_channels,
        'skip_channels': skip_channels,
        'end_channels': end_channels,
        'kernel_size': kernel_size,
        'blocks': blocks,
        'layers': layers,
        'device': device,
        'learning_rate': learning_rate,
        'prediction_horizon': prediction_horizon,
        'batch_size': batch_size,
        'n_epochs': n_epochs
    }

    if verbose:
        print("\nModel Configuration:")
        print("-------------------")
        for k, v in model_params.items():
            print(f"{k}: {v}")
        print("-------------------\n")

    # Initialize the model
    if estimator == 'gwn':
        model = GWNTrendPredictor(**model_params)
    else:
        raise ValueError(f"Unknown STGNN model: {estimator}")

    if verbose:
        print("\nInitial model training...")
        print(f"Training on full dataset with {n_epochs} epochs")
    
    # Initial training on full dataset
    model.fit(X_train, Y_train)
    fit_time = getattr(model, '_fit_time', None)
    
    if verbose:
        print("\nMaking predictions for evaluation...")

    # Make predictions on all sets
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Calculate metrics for each set
    from stgnn.evaluation.stgnn_metrics import calculate_rmse, calculate_mae, calculate_mape

    train_scores = [{
        'rmse': calculate_rmse(Y_train, train_pred),
        'mae': calculate_mae(Y_train, train_pred),
        'mape': calculate_mape(Y_train, train_pred)
    }]

    val_scores = [{
        'rmse': calculate_rmse(Y_val, val_pred),
        'mae': calculate_mae(Y_val, val_pred),
        'mape': calculate_mape(Y_val, val_pred)
    }]

    test_scores = [{
        'rmse': calculate_rmse(Y_test, test_pred),
        'mae': calculate_mae(Y_test, test_pred),
        'mape': calculate_mape(Y_test, test_pred)
    }]

    if verbose:
        print("\nEvaluation Results:")
        print("------------------")
        print(f"Train - RMSE: {train_scores[0]['rmse']:.4f}, MAE: {train_scores[0]['mae']:.4f}, MAPE: {train_scores[0]['mape']:.2f}%")
        print(f"Val   - RMSE: {val_scores[0]['rmse']:.4f}, MAE: {val_scores[0]['mae']:.4f}, MAPE: {val_scores[0]['mape']:.2f}%")
        print(f"Test  - RMSE: {test_scores[0]['rmse']:.4f}, MAE: {test_scores[0]['mae']:.4f}, MAPE: {test_scores[0]['mape']:.2f}%")

    # Return results
    results = {
        'train_scores': train_scores,
        'val_scores': val_scores,
        'test_scores': test_scores,
        'fit_time': [fit_time] if fit_time is not None else [0.0],
        'predictions': {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        }
    }
    
    return results

def normalise_parameters(default_params, config):
    """Normalize parameters for STGNN models."""
    _config = deepcopy(config)
    if 'algorithm' in _config:
        del _config['algorithm']
    for param in _config:
        default_params[param] = _config[param]
    return default_params 