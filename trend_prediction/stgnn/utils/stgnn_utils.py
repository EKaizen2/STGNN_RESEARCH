import numpy as np

def normalize_4d_array(X):
    """
    Normalize a 4D array using z-score normalization.
    
    Parameters
    ----------
    X : ndarray
        Array of shape [num_samples, num_nodes, window_size, num_features]
        to be normalized.
        
    Returns
    -------
    normalized_X : ndarray
        Normalized array of same shape as input.
        
    Notes
    -----
    This function performs z-score normalization on the last dimension (features)
    while preserving the structure of the first three dimensions (samples, nodes, window).
    """
    # Reshape to 2D for normalization
    orig_shape = X.shape
    X_2d = X.reshape(-1, X.shape[-1])
    
    # Calculate mean and std
    mean = np.mean(X_2d, axis=0)
    std = np.std(X_2d, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    # Normalize
    X_norm = (X_2d - mean) / std
    
    # Reshape back to original shape
    return X_norm.reshape(orig_shape)

def normalize_3d_array(X):
    """
    Normalize a 3D array using z-score normalization.
    
    Parameters
    ----------
    X : ndarray
        Array of shape [num_samples, num_nodes, num_features]
        to be normalized.
        
    Returns
    -------
    normalized_X : ndarray
        Normalized array of same shape as input.
        
    Notes
    -----
    This function performs z-score normalization on the last dimension (features)
    while preserving the structure of the first two dimensions (samples, nodes).
    """
    # Reshape to 2D for normalization
    orig_shape = X.shape
    X_2d = X.reshape(-1, X.shape[-1])
    
    # Calculate mean and std
    mean = np.mean(X_2d, axis=0)
    std = np.std(X_2d, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    # Normalize
    X_norm = (X_2d - mean) / std
    
    # Reshape back to original shape
    return X_norm.reshape(orig_shape)

def create_stgnn_windows(X, window_size):
    """
    Create sliding windows for STGNN input data.
    
    Parameters
    ----------
    X : ndarray
        Input array of shape [num_timesteps, num_nodes, num_features]
    window_size : int
        Size of the sliding window
        
    Returns
    -------
    windows : ndarray
        Array of shape [num_samples, num_nodes, window_size, num_features]
        where num_samples = num_timesteps - window_size + 1
    """
    num_timesteps, num_nodes, num_features = X.shape
    num_samples = num_timesteps - window_size + 1
    
    # Initialize output array
    windows = np.zeros((num_samples, num_nodes, window_size, num_features))
    
    # Create windows for each node
    for node in range(num_nodes):
        # Get data for this node
        node_data = X[:, node, :]  # [num_timesteps, num_features]
        
        # Create windows
        for i in range(num_samples):
            windows[i, node] = node_data[i:i + window_size]
    
    return windows

def create_stgnn_targets(X, window_size):
    """
    Create target values for STGNN prediction.
    
    Parameters
    ----------
    X : ndarray
        Input array of shape [num_timesteps, num_nodes, num_features]
    window_size : int
        Size of the sliding window
        
    Returns
    -------
    targets : ndarray
        Array of shape [num_samples, num_nodes, num_features]
        where num_samples = num_timesteps - window_size + 1
    """
    num_timesteps, num_nodes, num_features = X.shape
    num_samples = num_timesteps - window_size + 1
    
    # Initialize output array
    targets = np.zeros((num_samples, num_nodes, num_features))
    
    # Create targets for each node
    for node in range(num_nodes):
        # Get data for this node
        node_data = X[:, node, :]  # [num_timesteps, num_features]
        
        # Create targets (next value after each window)
        for i in range(num_samples):
            targets[i, node] = node_data[i + window_size - 1]
    
    return targets 