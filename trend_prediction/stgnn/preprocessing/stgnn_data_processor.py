import numpy as np
import pandas as pd
from stgnn.utils.stgnn_utils import create_stgnn_windows, create_stgnn_targets
import os
import pickle

def process_stgnn_data(raw_data, feature_type='trend', window_size=21, n_segments=10015, n_classes=3, lower=-0.5, upper=0.55, local_window=21, save_dir=None):
    """
    Process multivariate time series for STGNN with flexible feature extraction.
    Args:
        raw_data: DataFrame with multiple columns (nodes)
        feature_type: 'trend', 'pointdata', 'strength', 'direction'
        window_size: sliding window size
        n_segments: for trend segmentation
        n_classes: for direction
        lower, upper: for direction
        local_window: for local data
        save_dir: optional, for saving processed data
    Returns:
        X: [num_samples, num_nodes, window_size, num_features]
        Y: [num_samples, num_nodes, 2] (slope, duration)
    """
    from preprocessing.features import trend, movement_direction
    all_trend_features = []  # For Y (always slope, duration)
    all_X_features = []      # For X (depends on feature_type)
    for column in raw_data.columns:
        node_data = raw_data[column].values
        # Always compute trends for Y
        node_trends = trend(node_data, strength_metric='angle', angle_metric='degree',
                            duration=None, return_segment=False, overlap=False,
                            overlap_fraction=0.0, pla_algorithm='bottom', slope_estimator='regression',
                            error=False, max_error=False, n_segments=n_segments)
        trend_features = node_trends[['strength', 'duration']].values  # [timesteps, 2]
        all_trend_features.append(trend_features)
        # X feature extraction
        if feature_type == 'trend':
            all_X_features.append(trend_features)  # [timesteps, 2]
        elif feature_type == 'pointdata':
            all_X_features.append(node_data.reshape(-1, 1))  # [timesteps, 1]
        elif feature_type == 'strength':
            all_X_features.append(trend_features[:, [0]])  # [timesteps, 1]
        elif feature_type == 'direction':
            directions = movement_direction(node_trends['strength'], n_classes=n_classes, lower=lower, upper=upper)
            all_X_features.append(directions.reshape(-1, 1))  # [timesteps, 1]
        else:
            raise ValueError(f"Unsupported feature_type: {feature_type}")
    # Stack all nodes: [num_nodes, timesteps, features] -> [timesteps, num_nodes, features]
    X_features = np.stack(all_X_features, axis=1)  # [timesteps, num_nodes, features]
    trend_features_all = np.stack(all_trend_features, axis=1)  # [timesteps, num_nodes, 2]
    # Debug: print shapes before windowing
    print("X_features shape before windowing:", X_features.shape)
    print("trend_features_all shape before windowing:", trend_features_all.shape)
    # Ensure same number of timesteps before windowing
    min_timesteps = min(X_features.shape[0], trend_features_all.shape[0])
    if X_features.shape[0] != trend_features_all.shape[0]:
        print(f"Trimming base arrays to {min_timesteps} timesteps for alignment.")
    X_features = X_features[:min_timesteps]
    trend_features_all = trend_features_all[:min_timesteps]
    # Create windows and targets using STGNN utils
    X = create_stgnn_windows(X_features, window_size)  # [samples, num_nodes, window_size, features]
    Y = create_stgnn_targets(trend_features_all, window_size)  # [samples, num_nodes, 2]
    # Ensure X and Y have the same number of samples
    min_samples = min(X.shape[0], Y.shape[0])
    if X.shape[0] != Y.shape[0]:
        print(f"Trimming windowed arrays to {min_samples} samples for alignment.")
    X = X[:min_samples]
    Y = Y[:min_samples]
    # Save if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        processed_data = {'X': X, 'Y': Y, 'metadata': {'num_nodes': X.shape[1], 'num_features': X.shape[3], 'window_size': window_size, 'num_samples': X.shape[0]}}
        save_path = os.path.join(save_dir, f'stgnn_processed_data_{feature_type}_{n_segments}_segments.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(processed_data, f)
        print(f"Processed data saved to {save_path}")
    return X, Y

def load_stgnn_data(data_path='stgnn/processed_data/stgnn_processed_data.pkl'):
    """
    Load processed STGNN data.
    
    Parameters
    ----------
    data_path : str, optional
        Path to the processed data file, by default 'stgnn/processed_data/stgnn_processed_data.pkl'
        
    Returns
    -------
    X : ndarray
        Processed features
    Y : ndarray
        Target values
    metadata : dict
        Dictionary containing data metadata
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['X'], data['Y'], data['metadata'] 