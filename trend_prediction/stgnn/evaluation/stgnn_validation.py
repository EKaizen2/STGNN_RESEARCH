import numpy as np
from sklearn.metrics import mean_squared_error
import time
from stgnn.evaluation.stgnn_metrics import calculate_rmse, calculate_mae, calculate_mape, calculate_node_wise_metrics
from evaluation.partition import TimeSeriesPartition
import torch

def stgnn_time_series_validation(model, X, Y, n_splits=5, val_fraction=0.1, test_fraction=0.2,
                               return_train_score=False, scoring='neg_mean_squared_error',
                               verbose=1, return_estimator=False, n_jobs=1):
    """
    Walk-forward validation for STGNN models with 3-way split (train, val, test).
    """
    # Convert data to numpy if needed
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()
    n_samples = X.shape[0]
    train_size = int((1 - val_fraction - test_fraction) * n_samples)
    val_size = int(val_fraction * n_samples)
    test_size = int(test_fraction * n_samples)
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError("Invalid split sizes: train, val, and test sizes must all be > 0.")
    # Use TimeSeriesPartition for 3-way split
    tscv = TimeSeriesPartition(X, train_size, val_size, shift_train=True, validation=True)
    train_scores, val_scores, test_scores = [], [], []
    if verbose:
        print(f"\nStarting walk-forward validation with 3-way split: {n_splits} splits")
        print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")
    for i, (train_idx, val_idx, test_idx) in enumerate(tscv.split()):
        if verbose:
            print(f"\nSplit {i+1}")
            print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        # Get splits
        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = Y[train_idx], Y[val_idx], Y[test_idx]
        # Predict
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        # Print sample labels and predictions
        print("Sample y_train[0, 0, :] (label):", y_train[0, 0, :])
        print("Sample y_train_pred[0, 0, :] (pred):", y_train_pred[0, 0, :])
        print("Sample y_val[0, 0, :] (label):", y_val[0, 0, :])
        print("Sample y_val_pred[0, 0, :] (pred):", y_val_pred[0, 0, :])
        print("Sample y_test[0, 0, :] (label):", y_test[0, 0, :])
        print("Sample y_test_pred[0, 0, :] (pred):", y_test_pred[0, 0, :])
        # Metrics (validation uses only the next immediate prediction)
        train_score = {
            'rmse': calculate_rmse(y_train, y_train_pred),
            'mae': calculate_mae(y_train, y_train_pred),
            'mape': calculate_mape(y_train, y_train_pred)
        }
        val_score = {
            'rmse': calculate_rmse(y_val, y_val_pred),
            'mae': calculate_mae(y_val, y_val_pred),
            'mape': calculate_mape(y_val, y_val_pred)
        }
        test_score = {
            'rmse': calculate_rmse(y_test, y_test_pred),
            'mae': calculate_mae(y_test, y_test_pred),
            'mape': calculate_mape(y_test, y_test_pred)
        }
        train_scores.append(train_score)
        val_scores.append(val_score)
        test_scores.append(test_score)
        if verbose:
            print(f"Train RMSE: {train_score['rmse']:.4f}, MAE: {train_score['mae']:.4f}, MAPE: {train_score['mape']:.2f}%")
            print(f"Val   RMSE: {val_score['rmse']:.4f}, MAE: {val_score['mae']:.4f}, MAPE: {val_score['mape']:.2f}%")
            print(f"Test  RMSE: {test_score['rmse']:.4f}, MAE: {test_score['mae']:.4f}, MAPE: {test_score['mape']:.2f}%")
    # Aggregate results
    def agg(scores, metric):
        vals = [s[metric] for s in scores]
        return np.mean(vals), np.std(vals)
    results = {
        'train_scores': train_scores,
        'val_scores': val_scores,
        'test_scores': test_scores,
        'mean_train_rmse': agg(train_scores, 'rmse')[0],
        'std_train_rmse': agg(train_scores, 'rmse')[1],
        'mean_val_rmse': agg(val_scores, 'rmse')[0],
        'std_val_rmse': agg(val_scores, 'rmse')[1],
        'mean_test_rmse': agg(test_scores, 'rmse')[0],
        'std_test_rmse': agg(test_scores, 'rmse')[1],
        'mean_train_mae': agg(train_scores, 'mae')[0],
        'std_train_mae': agg(train_scores, 'mae')[1],
        'mean_val_mae': agg(val_scores, 'mae')[0],
        'std_val_mae': agg(val_scores, 'mae')[1],
        'mean_test_mae': agg(test_scores, 'mae')[0],
        'std_test_mae': agg(test_scores, 'mae')[1],
        'mean_train_mape': agg(train_scores, 'mape')[0],
        'std_train_mape': agg(train_scores, 'mape')[1],
        'mean_val_mape': agg(val_scores, 'mape')[0],
        'std_val_mape': agg(val_scores, 'mape')[1],
        'mean_test_mape': agg(test_scores, 'mape')[0],
        'std_test_mape': agg(test_scores, 'mape')[1],
    }
    if verbose:
        print("\nAggregated Results:")
        print(f"Train RMSE: {results['mean_train_rmse']:.4f} ± {results['std_train_rmse']:.4f}")
        print(f"Val   RMSE: {results['mean_val_rmse']:.4f} ± {results['std_val_rmse']:.4f}")
        print(f"Test  RMSE: {results['mean_test_rmse']:.4f} ± {results['std_test_rmse']:.4f}")
    return results

def stgnn_testing_with_recalibration(model, X, Y, train_size=None, test_size=None,
                                    shift_train=True, scoring='neg_mean_squared_error',
                                    new_validation_score=True, return_train_score=False,
                                    verbose=1, n_jobs=1, y_transformer=None):
    """
    Testing procedure for STGNN models with recalibration.
    Uses the pre-trained model for predictions without retraining.
    """
    try:
        # Convert data to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.numpy()
        
        # Split data into train and test sets
        if train_size is None and test_size is None:
            train_size = int(0.7 * len(X))
            test_size = len(X) - train_size
        
        X_train, X_test = X[:train_size], X[train_size:train_size+test_size]
        y_train, y_test = Y[:train_size], Y[train_size:train_size+test_size]
        
        if verbose:
            print(f"\nTest set evaluation")
            print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Make predictions using the pre-trained model
        if verbose:
            print("Making predictions on test set...")
        y_pred = model.predict(X_test)
        # Print sample test labels and predictions for all horizons if present
        if len(y_pred.shape) == 4:
            print("Sample y_test[0, 0, :] (label):", y_test[0, 0, :])
            print("Sample y_pred[0, 0, :, :] (pred, all horizons):", y_pred[0, 0, :, :])
        else:
            print("Sample y_test[0, 0, :] (label):", y_test[0, 0, :])
            print("Sample y_pred[0, 0, :] (pred):", y_pred[0, 0, :])
        # Calculate all metrics over the full prediction horizon if present
        if len(y_pred.shape) == 4:
            # Average metrics over all horizons
            horizon = y_pred.shape[2]
            rmse_list, mae_list, mape_list = [], [], []
            for h in range(horizon):
                rmse_list.append(calculate_rmse(y_test, y_pred[:, :, h, :]))
                mae_list.append(calculate_mae(y_test, y_pred[:, :, h, :]))
                mape_list.append(calculate_mape(y_test, y_pred[:, :, h, :]))
            test_rmse = np.mean(rmse_list)
            test_mae = np.mean(mae_list)
            test_mape = np.mean(mape_list)
        else:
            test_rmse = calculate_rmse(y_test, y_pred)
            test_mae = calculate_mae(y_test, y_pred)
            test_mape = calculate_mape(y_test, y_pred)
        
        # Calculate node-wise metrics if multiple nodes
        if len(y_test.shape) > 2 and y_test.shape[1] > 1:
            node_metrics = calculate_node_wise_metrics(y_test, y_pred)
        else:
            node_metrics = None
        
        results = {
            'test_scores': [{
                'rmse': test_rmse,
                'mae': test_mae,
                'mape': test_mape,
                'node_metrics': node_metrics
            }],
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        if return_train_score:
            if verbose:
                print("Making predictions on training set...")
            y_train_pred = model.predict(X_train)
            print("Sample y_train[0, 0, :] (label):", y_train[0, 0, :])
            if len(y_train_pred.shape) == 4:
                print("Sample y_train_pred[0, 0, :, :] (pred, all horizons):", y_train_pred[0, 0, :, :])
            else:
                print("Sample y_train_pred[0, 0, :] (pred):", y_train_pred[0, 0, :])
            train_rmse = calculate_rmse(y_train, y_train_pred)
            train_mae = calculate_mae(y_train, y_train_pred)
            train_mape = calculate_mape(y_train, y_train_pred)
            
            results['train_scores'] = [{
                'rmse': train_rmse,
                'mae': train_mae,
                'mape': train_mape
            }]
        
        if verbose:
            print("\nTest Results:")
            print(f"RMSE: {test_rmse:.4f}")
            print(f"MAE: {test_mae:.4f}")
            print(f"MAPE: {test_mape:.2f}%")
            
            if node_metrics:
                print("\nNode-wise metrics:")
                for node, metrics in node_metrics.items():
                    print(f"Node {node}:")
                    print(f"  RMSE: {metrics['rmse']:.4f}")
                    print(f"  MAE: {metrics['mae']:.4f}")
                    print(f"  MAPE: {metrics['mape']:.2f}%")
            
            if return_train_score:
                print("\nTrain Results:")
                print(f"RMSE: {train_rmse:.4f}")
                print(f"MAE: {train_mae:.4f}")
                print(f"MAPE: {train_mape:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        print("Attempting to print partial results...")
        
        # Try to print any results we have
        if 'test_rmse' in locals():
            print("\nPartial Test Results:")
            print(f"RMSE: {test_rmse:.4f}")
            if 'test_mae' in locals():
                print(f"MAE: {test_mae:.4f}")
            if 'test_mape' in locals():
                print(f"MAPE: {test_mape:.2f}%")
        
        raise  # Re-raise the exception after printing partial results 