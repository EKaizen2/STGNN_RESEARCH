import numpy as np
from preprocessing.features import trend
from preprocessing.util import sliding_window
from stgnn.preprocessing.stgnn_data_processor import process_stgnn_data, load_stgnn_data
from stgnn.utils.stgnn_utils import normalize_4d_array, normalize_3d_array
import pandas as pd

def to_stgnn_supervise(raw_data=None, feature_type='trend', window_size=21, n_segments=10015, n_classes=3, lower=-0.5, upper=0.55, local_window=21, save_dir='stgnn/processed_data'):
    """
    Preprocess data for STGNN models, supporting multiple feature types.
    Args:
        raw_data: DataFrame with multiple columns (nodes)
        feature_type: 'trend', 'pointdata', 'strength', or 'direction'
        window_size: sliding window size
        n_segments: number of segments for trend analysis
        n_classes: for direction discretization
        lower, upper: for direction binning
        local_window: for local data if needed
        save_dir: directory to save processed data (default: 'stgnn/processed_data')
    Returns:
        X: [num_samples, num_nodes, window_size, num_features]
        Y: [num_samples, num_nodes, 2] (slope, duration)
    """
    if raw_data is None:
        raise ValueError("raw_data must be provided")
    X, Y = process_stgnn_data(
        raw_data,
        feature_type=feature_type,
        window_size=window_size,
        n_segments=n_segments,
        n_classes=n_classes,
        lower=lower,
        upper=upper,
        local_window=local_window,
        save_dir=save_dir
    )
    X = normalize_4d_array(X)
    Y = normalize_3d_array(Y)
    return X, Y

def prepare_stgnn_data(raw_data, feature_type='trend', window_size=21):
    """
    Prepare data specifically for STGNN models with proper windowing and reshaping.
    
    Parameters
    ----------
    raw_data : DataFrame
        Input data (DataFrame with multiple columns)
    feature_type : str, optional
        Type of features to use ('trend' or 'pointdata'), by default 'trend'
    window_size : int, optional
        Size of the sliding window, by default 21
        
    Returns
    -------
    X : ndarray
        Processed features of shape [num_samples, num_nodes, window_size, num_features]
    Y : ndarray
        Target values of shape [num_samples, num_nodes, 2]
    """
    if not isinstance(raw_data, pd.DataFrame):
        raise ValueError("raw_data must be a pandas DataFrame with multiple columns")
    
    # Process data using the STGNN-specific processor
    X, Y = process_stgnn_data(raw_data, 
                            window_size=window_size)
    
    # Normalize the data using z-score normalization
    X = normalize_4d_array(X)
    Y = normalize_3d_array(Y)
    
    return X, Y 