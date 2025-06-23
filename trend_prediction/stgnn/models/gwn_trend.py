import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gwn import GraphWaveNet
from common._base_estimator import _Estimator
from common._pytorch_utility import create_tensor_from_numpy, detach_tensor
from preprocessing.features import trend, local_data
import numpy as np
import pandas as pd
import time


class GWNTrendPredictor(torch.nn.Module, _Estimator):
    def __init__(self, input_dim, output_dim=2, num_nodes=1, dropout=0.3, supports=None, 
                 gcn_bool=True, adapt_adj=True, hidden_dim=32, residual_channels=32, 
                 dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2, 
                 blocks=4, layers=2, learning_rate=1e-3, device='cuda', prediction_horizon=5,
                 batch_size=64, n_epochs=100):
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
            prediction_horizon: Number of future trends to predict
            batch_size: Batch size for training
            n_epochs: Number of training epochs
        """
        super(GWNTrendPredictor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.prediction_horizon = prediction_horizon
        self.device = device
        self.torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
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
            out_dim=output_dim,
            residual_channels=residual_channels,
            dilation_channels=dilation_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            kernel_size=kernel_size,
            blocks=blocks,
            layers=layers
        )
        
        # Move model to device
        self.to(self.torch_device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(self, X):
        """
        Forward pass of the model.
        
        Args:
            X: Input tensor of shape [batch_size, num_nodes, num_timesteps, num_features]
            
        Returns:
            Predictions of shape [batch_size, num_nodes, output_dim]
        """
        # Permute from [batch, nodes, timesteps, features] to [batch, features, nodes, timesteps]
        X = X.permute(0, 3, 1, 2)
        
        # Process through GraphWaveNet
        out = self.model(X)  # Shape: [batch_size, output_dim, num_nodes, num_timesteps]
        
        # Take the last timestep
        out = out[:, :, :, -1]  # Shape: [batch_size, output_dim, num_nodes]
        
        # Transpose to get [batch_size, num_nodes, output_dim]
        out = out.transpose(1, 2)
        
        return out

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters
        ----------
        X : numpy.ndarray or torch.Tensor
            Input data of shape [batch_size, num_nodes, num_timesteps, num_features]
        
        Returns
        -------
        numpy.ndarray
            Predictions of shape [batch_size, num_nodes, output_dim]
        """
        self.eval()  # Set to evaluation mode
        
        # Convert input to tensor if needed
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
            
        # Ensure input is 4D
        if X.dim() != 4:
            raise ValueError(f"Input to predict must be 4D, got shape {X.shape}")
        
        # Print shapes for debugging
        print(f"Input shape to predict: {X.shape}")
        print(f"Expected shape: [batch_size, {self.num_nodes}, timesteps, {self.input_dim}]")
        
        # Verify dimensions
        if X.shape[1] != self.num_nodes or X.shape[3] != self.input_dim:
            raise ValueError(
                f"Input shape mismatch. Got shape {X.shape}, "
                f"but expected [batch_size, {self.num_nodes}, timesteps, {self.input_dim}]"
            )
        
        # Move to CPU for prediction to avoid memory issues
        device = next(self.parameters()).device
        self.to('cpu')
        X = X.to('cpu')
        
        try:
            predictions = []
            
            # Process in batches
            with torch.no_grad():
                for i in range(0, len(X), self.batch_size):
                    batch_X = X[i:i + self.batch_size]
                    batch_pred = self(batch_X)
                    predictions.append(batch_pred.cpu().numpy())
            
            # Concatenate all batch predictions
            predictions = np.concatenate(predictions, axis=0)
            
            # Move model back to original device
            self.to(device)
            
            return predictions
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            self.to(device)
            raise

    def fit(self, X, Y):
        """
        Train the model
        
        Args:
            X: Input data of shape [batch_size, num_nodes, num_timesteps, num_features]
            Y: Target data of shape [batch_size, num_nodes, output_dim]
        """
        start_time = time.time()
        
        # Convert inputs to tensors
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(Y, torch.Tensor):
            Y = torch.FloatTensor(Y)
            
        # Ensure input is 4D
        if X.dim() != 4:
            raise ValueError(f"Input X to fit must be 4D, got shape {X.shape}")
            
        # Print shapes for debugging
        print(f"\nInput shapes:")
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        print(f"Expected X shape: [batch_size, {self.num_nodes}, timesteps, {self.input_dim}]")
        print(f"Expected Y shape: [batch_size, {self.num_nodes}, {self.output_dim}]")
        
        # Verify dimensions
        if X.shape[1] != self.num_nodes or X.shape[3] != self.input_dim:
            raise ValueError(
                f"Input X shape mismatch. Got shape {X.shape}, "
                f"but expected [batch_size, {self.num_nodes}, timesteps, {self.input_dim}]"
            )
        if Y.shape[1] != self.num_nodes or Y.shape[2] != self.output_dim:
            raise ValueError(
                f"Input Y shape mismatch. Got shape {Y.shape}, "
                f"but expected [batch_size, {self.num_nodes}, {self.output_dim}]"
            )
        
        # Move to device
        Y = Y.to(self.torch_device)
        X = X.to(self.torch_device)
        
        # Calculate number of batches
        n_samples = X.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        print(f"\nStarting training for {self.n_epochs} epochs:")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches per epoch: {n_batches}")
        
        # Training loop
        for epoch in range(self.n_epochs):
            self.train()
            epoch_loss = 0.0
            batch_losses = []
            
            # Process batches
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, n_samples)
                
                # Get batch
                X_batch = X[start_idx:end_idx]
                Y_batch = Y[start_idx:end_idx]
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self(X_batch)
                
                # Calculate loss
                loss = self.loss_fn(predictions, Y_batch)
                batch_losses.append(loss.item())
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Print batch progress every 10 batches
                if i % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.n_epochs}, Batch {i+1}/{n_batches}, Loss: {loss.item():.6f}")
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / n_batches
            min_batch_loss = min(batch_losses)
            max_batch_loss = max(batch_losses)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.n_epochs} Summary:")
            print(f"  Average Loss: {avg_loss:.6f}")
            print(f"  Min Batch Loss: {min_batch_loss:.6f}")
            print(f"  Max Batch Loss: {max_batch_loss:.6f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}\n")
        
        print("Training completed!")
        self._fit_time = time.time() - start_time
        return self

    def set_prediction_horizon(self, horizon):
        """Update the prediction horizon."""
        self.prediction_horizon = horizon
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
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
            
        # Input shape should be [batch_size, num_nodes, num_timesteps, num_features]
        # GraphWaveNet expects [batch_size, num_features, num_nodes, num_timesteps]
        if len(X.shape) == 4:  # [batch_size, num_nodes, num_timesteps, num_features]
            # Transpose to match GraphWaveNet's expected format
            X = X.permute(0, 3, 1, 2)
        elif len(X.shape) == 2:  # [batch_size, num_features]
            X = X.reshape(-1, X.shape[1], 1, 1)
        elif len(X.shape) == 3:  # [batch_size, num_nodes, num_features]
            X = X.reshape(-1, X.shape[2], X.shape[1], 1)
            
        # Ensure the number of features matches input_dim
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Input features dimension {X.shape[1]} does not match model input_dim {self.input_dim}")
            
        return X.to(self.torch_device)

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
            'device': self.device,
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
            if parameter == 'prediction_horizon':
                self.set_prediction_horizon(value)
            else:
                setattr(self, parameter, value)
        return self 