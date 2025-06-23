import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gwn import GraphWaveNet
from common._base_estimator import _Estimator
from common._pytorch_utility import create_tensor_from_numpy, detach_tensor
from preprocessing.features import trend, local_data
import numpy as np
import pandas as pd


class GWNTrendPredictor(torch.nn.Module, _Estimator):
    def __init__(self, input_dim, output_dim=2, num_nodes=1, dropout=0.3, supports=None, 
                 gcn_bool=True, adapt_adj=True, hidden_dim=12, residual_channels=32, 
                 dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2, 
                 blocks=4, layers=2, learning_rate=1e-3, device='cuda', prediction_horizon=5,
                 batch_size=32, n_epochs=100):
        """
        Graph WaveNet model for trend prediction
        Args:
            input_dim: Input dimension (number of features per node)
            output_dim: Output dimension (2 for slope and duration)
            num_nodes: Number of nodes in the graph
            dropout: Dropout rate
            supports: Support matrices for graph convolution
            gcn_bool: Whether to use graph convolution
            adapt_adj: Whether to use adaptive adjacency matrix
            hidden_dim: Hidden dimension per node
            residual_channels: Number of residual channels
            dilation_channels: Number of dilation channels
            skip_channels: Number of skip channels
            end_channels: Number of end channels
            kernel_size: Kernel size for temporal convolution
            blocks: Number of blocks
            layers: Number of layers per block
            device: Device to run the model on
            learning_rate: Learning rate for optimization
            prediction_horizon: Prediction horizon
            batch_size: Batch size for training
            n_epochs: Number of training epochs
        """
        super(GWNTrendPredictor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.prediction_horizon = prediction_horizon
        self.device = device  # Store as string for scikit-learn compatibility
        self.torch_device = torch.device(device if torch.cuda.is_available() else "cpu")  # Actual torch device
        self.dropout = dropout
        self.supports = supports
        self.gcn_bool = gcn_bool
        self.adapt_adj = adapt_adj
        self.hidden_dim = hidden_dim
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.layers = layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Initialize the GraphWaveNet model
        self.model = GraphWaveNet(
            device=self.torch_device,
            node_cnt=num_nodes,
            dropout=dropout,
            supports=supports,
            gcn_bool=gcn_bool,
            adapt_adj=adapt_adj,
            in_dim=input_dim,
            out_dim=hidden_dim,
            residual_channels=residual_channels,
            dilation_channels=dilation_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            kernel_size=kernel_size,
            blocks=blocks,
            layers=layers
        )
        
        # Add a final layer to predict slope and duration for each timestep in the horizon
        self.final_layer = nn.Linear(hidden_dim, output_dim * prediction_horizon)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Move model to device
        self.model.to(self.torch_device)
        self.final_layer.to(self.torch_device)
        
    def eval(self):
        """Set the model to evaluation mode"""
        super(GWNTrendPredictor, self).eval()
        self.model.eval()
        return self

    def train(self, mode=True):
        """Set the model to training mode"""
        super(GWNTrendPredictor, self).train(mode)
        self.model.train(mode)
        return self

    def _prepare_data(self, X):
        """
        Prepare input data for the model
        Args:
            X: Input data (numpy array or pandas DataFrame)
        Returns:
            Processed X tensor
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        elif isinstance(X, pd.DataFrame):
            X = torch.from_numpy(X.values).float()
            
        # For pointdata, we need to reshape the input to match the model's expectations
        # Input shape should be [batch_size, num_nodes, num_timesteps, num_features]
        if len(X.shape) == 2:  # [batch_size, num_features]
            X = X.reshape(-1, 1, 1, X.shape[1])
        elif len(X.shape) == 3:  # [batch_size, num_nodes, num_features]
            X = X.reshape(-1, X.shape[1], 1, X.shape[2])
            
        # Ensure the number of features matches input_dim
        if X.shape[-1] != self.input_dim:
            raise ValueError(f"Input features dimension {X.shape[-1]} does not match model input_dim {self.input_dim}")
            
        return X.to(self.torch_device)

    def forward(self, X):
        """
        Forward pass through the model
        Args:
            X: Input tensor of shape [batch_size, num_nodes, num_timesteps, num_features]
        Returns:
            Predictions of shape [batch_size, num_nodes, prediction_horizon, output_dim]
        """
        # Forward pass through GraphWaveNet
        outputs = self.model(X)  # [batch_size, num_nodes, num_timesteps, hidden_dim]
        
        # Reshape for final layer
        batch_size = outputs.shape[0]
        outputs = outputs.reshape(batch_size * self.num_nodes, -1)  # [batch_size * num_nodes, hidden_dim]
        
        # Final layer
        predictions = self.final_layer(outputs)  # [batch_size * num_nodes, output_dim * prediction_horizon]
        
        # Reshape to match expected output format
        predictions = predictions.reshape(batch_size, self.num_nodes, self.prediction_horizon, self.output_dim)
        
        return predictions

    def predict(self, X):
        """
        Make predictions
        Args:
            X: Input data
        Returns:
            Predictions as numpy array of shape [batch_size, prediction_horizon, output_dim]
        """
        self.eval()
        with torch.no_grad():
            X = self._prepare_data(X)
            predictions = self(X)
            
            # Convert to numpy array
            predictions = predictions.cpu().numpy()
            
            # Reshape to match expected output format [batch_size, prediction_horizon, output_dim]
            predictions = predictions.reshape(-1, self.prediction_horizon, self.output_dim)
            
        return predictions

    def prepare_trend_features(self, X):
        """
        Prepare trend features from raw data
        Args:
            X: Raw data tensor [batch_size, num_nodes, num_timesteps, num_features]
        Returns:
            Trend features tensor [batch_size, num_nodes, num_features, trend_dim]
        """
        batch_size, num_nodes, num_timesteps, num_features = X.shape
        
        # Initialize list to store trend features
        trend_features = []
        
        # Process each feature
        for f in range(num_features):
            # Get data for this feature
            feature_data = X[:, :, :, f]  # [batch_size, num_nodes, num_timesteps]
            
            # Calculate trend features (strength and duration)
            # For now, using simple linear regression
            slopes = []
            durations = []
            
            for b in range(batch_size):
                for n in range(num_nodes):
                    # Get time series for this node
                    ts = feature_data[b, n].cpu().numpy()  # [num_timesteps]
                    
                    # Calculate slope using linear regression
                    x = np.arange(len(ts))
                    slope, _ = np.polyfit(x, ts, 1)
                    
                    # Calculate duration (number of timesteps)
                    duration = len(ts)
                    
                    slopes.append(slope)
                    durations.append(duration)
            
            # Reshape to [batch_size, num_nodes]
            slopes = np.array(slopes).reshape(batch_size, num_nodes)
            durations = np.array(durations).reshape(batch_size, num_nodes)
            
            # Stack slope and duration
            trend_feature = np.stack([slopes, durations], axis=-1)  # [batch_size, num_nodes, 2]
            trend_features.append(trend_feature)
        
        # Stack all features
        trend_features = np.stack(trend_features, axis=1)
        return torch.FloatTensor(trend_features).to(self.torch_device)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Required for scikit-learn compatibility.
        """
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_nodes': self.num_nodes,
            'dropout': self.dropout,
            'supports': self.supports,
            'gcn_bool': self.gcn_bool,
            'adapt_adj': self.adapt_adj,
            'hidden_dim': self.hidden_dim,
            'residual_channels': self.residual_channels,
            'dilation_channels': self.dilation_channels,
            'skip_channels': self.skip_channels,
            'end_channels': self.end_channels,
            'kernel_size': self.kernel_size,
            'blocks': self.blocks,
            'layers': self.layers,
            'learning_rate': self.learning_rate,
            'device': self.device,  # Return the string device
            'prediction_horizon': self.prediction_horizon,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs
        }

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        Required for scikit-learn compatibility.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, Y):
        """
        Train the model
        Args:
            X: Input data
            Y: Target data
        Returns:
            self
        """
        # Convert inputs to tensors
        X = self._prepare_data(X)
        Y = torch.from_numpy(Y).float().to(self.torch_device)
        
        # Training loop
        for epoch in range(self.n_epochs):
            self.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self(X)
            
            # Reshape predictions and targets for loss calculation
            batch_size = predictions.shape[0]
            predictions = predictions.reshape(batch_size, -1)
            Y_reshaped = Y.reshape(batch_size, -1)
            
            # Calculate loss
            loss = self.loss_fn(predictions, Y_reshaped)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
        return self 