from common._base_estimator import _Estimator
from common._pytorch_utility import create_tensor_from_numpy, detach_tensor
import torch
import torch.nn as nn
from sklearn.utils import gen_batches
import gc
import numpy as np


class BOHBLSTMRegressor(torch.nn.Module, _Estimator):
    def __init__(self, input_dim, output_dim, n_cells, batch_size, n_layers_per_lstm=1,
                 dropout=0.0, n_epochs=100, learning_rate=1e-3, weight_decay=0.0, warm_start=False,
                 warm_start_n_epoch=0.0025, lowest_train_loss=True, validation_fraction=0.05, verbose=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.n_cells = n_cells
        self.dropout = dropout
        self.n_layers_per_lstm = n_layers_per_lstm

        self.lstm_layers = torch.nn.ModuleList()
        for in_features, out_features in zip([self.input_dim] + self.n_cells[: -1], self.n_cells):
            self.lstm_layers.append(torch.nn.LSTM(in_features, out_features, self.n_layers_per_lstm,
                                                  batch_first=True, dropout=0.0))
        self.dense_layer = torch.nn.Linear(self.n_cells[-1], self.output_dim)
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=1)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.lowest_train_loss = lowest_train_loss
        self.warm_start_n_epoch = warm_start_n_epoch
        self.warm_start = warm_start
        if 0.0 < self.warm_start_n_epoch < 1:
            self.warm_start_n_epoch = int(self.warm_start_n_epoch * self.n_epochs)
        self.verbose = verbose
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.hidden = self.init_hidden(self.n_cells[0])

        # n_gpus = torch.cuda.device_count()
        # if n_gpus > 1:
        # 	self = torch.nn.DataParallel(self, list(range(n_gpus)))
        self.to(self.device)

        def weight_init(model):
            if isinstance(model, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(model.weight, mode='fan_in', nonlinearity='relu')
                if model.bias is not None:
                    torch.nn.init.zeros_(model.bias)
        self.apply(weight_init)

    def init_hidden(self, n_cells):
        # This is what we'll initialise our hidden state as
        hidden_1 = torch.empty((self.n_layers_per_lstm, self.batch_size, n_cells))
        hidden_1 = torch.nn.init.xavier_normal_(hidden_1)
        hidden_2 = torch.empty(self.n_layers_per_lstm, self.batch_size, n_cells)
        hidden_2 = torch.nn.init.xavier_normal_(hidden_2)
        hidden_1 = torch.nn.Parameter(hidden_1, requires_grad=True)
        hidden_2 = torch.nn.Parameter(hidden_2, requires_grad=True)
        return hidden_1.to(self.device), hidden_2.to(self.device)

    def forward(self, X, batch_size=None):
        if batch_size is None:
            batch_size = min(self.batch_size, len(X))
        for lstm_layer in self.lstm_layers:
            lstm_layer.flatten_parameters()
            X = lstm_layer(X.view(batch_size, 1, -1))[0]
            X = self.activation(X.contiguous())
            X = torch.nn.Dropout(p=self.dropout)(X)
        return self.dense_layer(X.view(batch_size, -1))

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fit(self, X, Y):
        if self.warm_start:
            try:
                self.load_state_dict(torch.load("models/checkpoints/lstm_regressor_parent_model.pth"))
                self.n_epochs = self.warm_start_n_epoch
            except FileNotFoundError:
                pass
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                      betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=self.weight_decay, amsgrad=False)
        loss_function = torch.nn.MSELoss()
        if self.validation_fraction is not None:
            train_end = int(Y.shape[0]*(1 - self.validation_fraction))
            val_loss, stopping_counter = 0, -1
        else:
            train_end = Y.shape[0]
        if self.lowest_train_loss:
            best_parameters = self.state_dict()
            from sys import maxsize
            min_loss = maxsize
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            self.train()
            for batch_slice in gen_batches(train_end, self.batch_size):
                # Forward pass
                y_pred = self(create_tensor_from_numpy(X[batch_slice]))
                loss = loss_function(y_pred, create_tensor_from_numpy(Y[batch_slice]))
                epoch_loss += loss.item() * len(Y[batch_slice])
                # Zero out gradient, else they will accumulate between epochs
                optimizer.zero_grad()
                # Backward pass
                loss.backward()
                # Update parameters
                optimizer.step()
            if self.lowest_train_loss:
                train_loss = loss_function(self(create_tensor_from_numpy(X), X.shape[0]),
                                           create_tensor_from_numpy(Y)).item()
                if train_loss < min_loss:
                    min_loss = train_loss
                    best_parameters = self.state_dict()
                if self.verbose:
                    print("Epoch ", epoch, " - batch Loss: ", train_loss)
            if self.verbose:
                print("Epoch ", epoch, " - min-batched Loss: ", epoch_loss / train_end)
            if self.validation_fraction is not None:
                temp_val_loss = loss_function(self(create_tensor_from_numpy(X[train_end:]), len(X[train_end:])),
                                              create_tensor_from_numpy(Y[train_end:])).item()
                if temp_val_loss > val_loss:
                    stopping_counter += 1
                elif temp_val_loss < val_loss:
                    stopping_counter = 0
                val_loss = temp_val_loss
                if self.verbose:
                    print("Validation Loss: ", temp_val_loss)
                if stopping_counter > 10:
                    if self.verbose:
                        print("Early Stopping")
                    break
        if self.lowest_train_loss:
            self.load_state_dict(best_parameters)
        if self.warm_start:
            torch.save(self.state_dict(), "models/checkpoints/lstm_regressor_parent_model.pth")
        return self

    def predict(self, X):
        self.to(torch.device("cpu"))
        self.eval()
        return detach_tensor(self.forward(torch.from_numpy(X).type(torch.FloatTensor), X.shape[0]))
