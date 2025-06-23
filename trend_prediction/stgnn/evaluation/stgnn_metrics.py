import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Square Error between true and predicted values.
    Handles both 3D and 4D predictions (with prediction horizon).
    
    Args:
        y_true: True values of shape [batch_size, num_nodes, num_features]
        y_pred: Predicted values of shape [batch_size, num_nodes, prediction_horizon, num_features]
    """
    # If predictions have prediction horizon, take the first horizon
    if len(y_pred.shape) == 4:
        y_pred = y_pred[:, :, 0, :]  # Take first prediction horizon
    
    # Reshape to 2D for RMSE calculation
    y_true = y_true.reshape(-1, y_true.shape[-1])
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error between true and predicted values.
    Handles both 3D and 4D predictions (with prediction horizon).
    
    Args:
        y_true: True values of shape [batch_size, num_nodes, num_features]
        y_pred: Predicted values of shape [batch_size, num_nodes, prediction_horizon, num_features]
    """
    # If predictions have prediction horizon, take the first horizon
    if len(y_pred.shape) == 4:
        y_pred = y_pred[:, :, 0, :]  # Take first prediction horizon
    
    # Reshape to 2D for MAE calculation
    y_true = y_true.reshape(-1, y_true.shape[-1])
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    
    return np.mean(np.abs(y_true - y_pred))

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error between true and predicted values.
    Handles both 3D and 4D predictions (with prediction horizon).
    
    Args:
        y_true: True values of shape [batch_size, num_nodes, num_features]
        y_pred: Predicted values of shape [batch_size, num_nodes, prediction_horizon, num_features]
    """
    # If predictions have prediction horizon, take the first horizon
    if len(y_pred.shape) == 4:
        y_pred = y_pred[:, :, 0, :]  # Take first prediction horizon
    
    # Reshape to 2D for MAPE calculation
    y_true = y_true.reshape(-1, y_true.shape[-1])
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    
    # Avoid division by zero
    mask = y_true != 0
    mape = np.zeros_like(y_true)
    mape[mask] = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    
    return np.mean(mape) * 100  # Convert to percentage

def calculate_node_wise_metrics(y_true, y_pred):
    """
    Calculate metrics for each node separately.
    Handles both 3D and 4D predictions (with prediction horizon).
    
    Args:
        y_true: True values of shape [batch_size, num_nodes, num_features]
        y_pred: Predicted values of shape [batch_size, num_nodes, prediction_horizon, num_features]
    """
    # If predictions have prediction horizon, take the first horizon
    if len(y_pred.shape) == 4:
        y_pred = y_pred[:, :, 0, :]  # Take first prediction horizon
    
    num_nodes = y_true.shape[1]
    node_metrics = {}
    
    for node in range(num_nodes):
        node_true = y_true[:, node, :]
        node_pred = y_pred[:, node, :]
        
        node_metrics[f'node_{node}'] = {
            'rmse': calculate_rmse(node_true, node_pred),
            'mae': calculate_mae(node_true, node_pred),
            'mape': calculate_mape(node_true, node_pred)
        }
    
    return node_metrics 