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
    from preprocessing.util import sliding_window
    all_X = []
    all_Y = []
    for column in raw_data.columns:
        node_data = raw_data[column].values
        # Always compute trends for Y
        node_trends = trend(node_data, strength_metric='angle', angle_metric='degree',
                            duration=None, return_segment=False, overlap=False,
                            overlap_fraction=0.0, pla_algorithm='bottom', slope_estimator='regression',
                            error=False, max_error=False, n_segments=n_segments)
        trend_features = node_trends[['strength', 'duration']].values
        # X feature extraction
        if feature_type == 'trend':
            node_X = sliding_window(trend_features, window_size)  # [samples, window, 2]
        elif feature_type == 'pointdata':
            node_X = sliding_window(node_data.reshape(-1, 1), window_size)  # [samples, window, 1]
        elif feature_type == 'strength':
            node_X = sliding_window(trend_features[:, [0]], window_size)  # [samples, window, 1]
        elif feature_type == 'direction':
            directions = movement_direction(node_trends['strength'], n_classes=n_classes, lower=lower, upper=upper)
            node_X = sliding_window(directions.reshape(-1, 1), window_size)  # [samples, window, 1]
        else:
            raise ValueError(f"Unsupported feature_type: {feature_type}")
        # Y is always next [slope, duration] after window
        node_Y = trend_features[window_size-1:]  # [samples, 2]
        all_X.append(node_X)
        all_Y.append(node_Y)
    # Stack nodes: [num_nodes, samples, window, features] -> [samples, num_nodes, window, features]
    X = np.stack(all_X, axis=1)
    Y = np.stack(all_Y, axis=1)
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