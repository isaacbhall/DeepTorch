import timm
import pandas as pd
import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
explained_variance_score, mean_squared_log_error, median_absolute_error)
import numpy as np
import hashlib
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import logging
from sklearn.impute import SimpleImputer
from typing import Optional, List, Union
from sklearn.base import TransformerMixin, BaseEstimator
from scipy.sparse import issparse, csr_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import ta
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optuna.trial import Trial
from torchvision import transforms
from sklearn.model_selection import TimeSeriesSplit
import requests
from scipy.signal import savgol_filter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    A class to evaluate the performance of a model using various metrics.

    Attributes:
        true_values (array-like): The ground truth values.
        predicted_values (array-like): The predicted values from the model.

    Usage:
        evaluator = ModelEvaluator(true_values, predicted_values)
        mse = evaluator.MSE()
    """

    def __init__(self, true_values, predicted_values):
        self.true_values = true_values
        self.predicted_values = predicted_values

    def explained_variance(self) -> float:
        """Calculates the explained variance score."""
        return explained_variance_score(self.true_values, self.predicted_values)

    def MSLE(self) -> float:
        """Calculates the Mean Squared Logarithmic Error."""
        return mean_squared_log_error(self.true_values, self.predicted_values)

    def MedAE(self) -> float:
        """Calculates the Median Absolute Error."""
        return median_absolute_error(self.true_values, self.predicted_values)

    def MSE(self) -> float:
        """Calculates the Mean Squared Error."""
        return mean_squared_error(self.true_values, self.predicted_values)

    def RMSE(self) -> float:
        """Calculates the Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(self.true_values, self.predicted_values))

    def MAE(self) -> float:
        """Calculates the Mean Absolute Error."""
        return mean_absolute_error(self.true_values, self.predicted_values)

    def r_squared(self) -> float:
        """Calculates the R-squared score."""
        return r2_score(self.true_values, self.predicted_values)

    def MAPE(self) -> float:
        """
        Calculates the Mean Absolute Percentage Error.
        Avoids division by zero by excluding zero values.
        """
        self.true_values = np.array(self.true_values)
        nonzero_element_indices = self.true_values != 0
        true_values_nonzero = self.true_values[nonzero_element_indices]
        predicted_values_nonzero = np.array(self.predicted_values)[nonzero_element_indices]
        return np.mean(np.abs((true_values_nonzero - predicted_values_nonzero) / true_values_nonzero)) * 100

    def mean_directional_accuracy(self) -> float:
        """Calculates the Mean Directional Accuracy."""
        direction_true = np.sign(np.diff(self.true_values))
        direction_pred = np.sign(np.diff(self.predicted_values))
        return np.mean(direction_true == direction_pred) * 100

    def theils_u_statistic(self) -> float:
        """Calculates Theil's U statistic."""
        error = self.true_values - self.predicted_values
        mse = np.mean(error ** 2)
        denom = np.mean(self.true_values ** 2) + np.mean(self.predicted_values ** 2) + mse
        return np.sqrt(mse / denom)

    def profit_or_loss(self) -> float:
        """
        Calculates the profit or loss assuming 1 unit bought/sold at each time step.
        """
        return np.sum(np.diff(self.true_values) * np.sign(self.predicted_values[:-1]))

    def cumulative_return(self) -> float:
        """Calculates the cumulative return over the period."""
        return (self.true_values[-1] / self.true_values[0]) - 1
    
    def coverage_probability(self, lower_bounds, upper_bounds) -> float:
        """Calculates the coverage probability of the confidence intervals."""
        coverage = np.mean((lower_bounds <= self.true_values) & (self.true_values <= upper_bounds))
        return coverage * 100
    
    def precision(self) -> float:
        """Calculates the precision."""
        return precision_score(self.true_values, self.predicted_values)

    def recall(self) -> float:
        """Calculates the recall."""
        return recall_score(self.true_values, self.predicted_values)

    def f1(self) -> float:
        """Calculates the F1 score."""
        return f1_score(self.true_values, self.predicted_values)

    def auc_roc(self) -> float:
        """Calculates the area under the ROC curve."""
        return roc_auc_score(self.true_values, self.predicted_values)
    
    def residual_histogram(self):
        """Plots a histogram of the residuals."""
        residuals = self.true_values - self.predicted_values
        plt.hist(residuals, bins=30)
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.title('Residual Histogram')
        plt.show()

    def residual_autocorrelation(self):
        """Calculates the autocorrelation of the residuals."""
        residuals = self.true_values - self.predicted_values
        return pd.Series(residuals).autocorr()
    
    @staticmethod
    def compare_models(evaluators: list):
        """Compares multiple models using the defined metrics."""
        metrics = {}
        for i, evaluator in enumerate(evaluators):
            metrics[f'Model {i + 1}'] = {
                'MSE': evaluator.MSE(),
                'RMSE': evaluator.RMSE(),
                'MAE': evaluator.MAE(),
                # ... (other metrics)
            }
        return pd.DataFrame(metrics)

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    A class to preprocess data by handling missing values and scaling features.

    Attributes:
        scaling_strategy (str): The strategy for scaling features. Either 'standard' or 'minmax'.
        missing_values_strategy (str): The strategy for handling missing values. Either 'mean', 'median', or 'most_frequent'.
        selected_features (array-like, optional): The indices of features to keep.

    Usage:
        preprocessor = DataPreprocessor(scaling_strategy='minmax')
        transformed_data = preprocessor.fit_transform(data)
    """

    def __init__(self, scaling_strategy: str = "standard", missing_values_strategy: str = "mean", selected_features: Optional[np.ndarray] = None):
        self.validate_params(scaling_strategy, missing_values_strategy)
        self.scaling_strategy = scaling_strategy
        self.missing_values_strategy = missing_values_strategy
        self.selected_features = selected_features
        self.pipeline = self.build_pipeline()


    def validate_params(self, scaling_strategy, missing_values_strategy):
        """Validates the parameters for scaling and imputation strategies."""
        assert scaling_strategy in [None, "standard", "minmax"], "Invalid scaling strategy. Choose either 'standard', 'minmax', or None."
        assert missing_values_strategy in ["mean", "median", "most_frequent", None], "Invalid imputation strategy. Choose either 'mean', 'median', 'most_frequent', or None."

    def build_pipeline(self):
        """Builds a pipeline for imputation and scaling."""
        if self.missing_values_strategy == None:
            imputer = None
        else:
            imputer = SimpleImputer(strategy=self.missing_values_strategy)
        if self.scaling_strategy == None:
            scaler = None
        elif self.scaling_strategy == "standard":
            scaler = StandardScaler() 
        else:
            MinMaxScaler()
        return Pipeline([('imputer', imputer), ('scaler', scaler)])

    def fit(self, data: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> 'DataPreprocessor':
        """Fits the preprocessor to the data."""
        try:
            logger.info("Fitting DataPreprocessor...")
            if self.selected_features is not None:
                data = data[:, self.selected_features]  # Keep only selected features
            self.pipeline.fit(data)
        except Exception as e:
            logger.error(f"Failed to fit DataPreprocessor: {e}")
            raise
        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> torch.Tensor:
        """Transforms the data using the fitted preprocessor."""
        try:
            logger.info("Transforming data...")
            if self.selected_features is not None:
                data = data[:, self.selected_features]  # Keep only selected features
            transformed_data = self.pipeline.transform(data)
        except Exception as e:
            logger.error(f"Failed to transform data: {e}")
            raise
        return torch.tensor(transformed_data, dtype=torch.float32) if not issparse(transformed_data) else csr_matrix(transformed_data)

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> torch.Tensor:
        """Fits the preprocessor to the data and then transforms the data."""
        try:
            logger.info("Fitting and transforming data...")
            if self.selected_features is not None:
                data = data[:, self.selected_features]  # Keep only selected features
            transformed_data = self.pipeline.fit_transform(data)
        except Exception as e:
            logger.error(f"Failed to fit and transform data: {e}")
            raise
        return torch.tensor(transformed_data, dtype=torch.float32) if not issparse(transformed_data) else csr_matrix(transformed_data)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A class to engineer features by applying a series of transformers.

    Attributes:
        n_components (int, optional): Number of components for PCA. If None, PCA is not applied.
        additional_transformers (list, optional): List of additional transformers to apply.
        selected_features (array-like, optional): Indices of features to keep.

    Usage:
        engineer = FeatureEngineer(n_components=2)
        engineered_data = engineer.fit_transform(data)

    """

    def __init__(self, n_components: Optional[int] = None, additional_transformers: Optional[List[TransformerMixin]] = None, selected_features: Optional[np.ndarray] = None):
        self.n_components = n_components
        self.additional_transformers = additional_transformers or []
        self.selected_features = selected_features
        self.pipeline = self.build_pipeline()

    def build_pipeline(self):
        """Builds a pipeline of transformers."""
        transformers = [(('pca', PCA(n_components=self.n_components)) if self.n_components else None)] + self.additional_transformers
        transformers = [t for t in transformers if t]  # Remove None if it exists
        return Pipeline(transformers)

    def fit(self, data: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> 'FeatureEngineer':
        """Fits the transformers to the data."""
        try:
            logger.info("Fitting FeatureEngineer...")
            if self.selected_features is not None:
                data = data[:, self.selected_features]  # Keep only selected features
            self.pipeline.fit(data)
        except Exception as e:
            logger.error(f"Failed to fit FeatureEngineer: {e}")
            raise
        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> torch.Tensor:
        """Transforms the data using the fitted transformers."""
        try:
            logger.info("Engineering features...")
            if self.selected_features is not None:
                data = data[:, self.selected_features]  # Keep only selected features
            engineered_features = self.pipeline.transform(data)
        except Exception as e:
            logger.error(f"Failed to engineer features: {e}")
            raise
        return torch.tensor(engineered_features, dtype=torch.float32) if not issparse(engineered_features) else csr_matrix(engineered_features)

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> torch.Tensor:
        """Fits the transformers to the data and then transforms the data."""
        try:
            logger.info("Fitting and engineering features...")
            if self.selected_features is not None:
                data = data[:, self.selected_features]  # Keep only selected features
            engineered_features = self.pipeline.fit_transform(data)
        except Exception as e:
            logger.error(f"Failed to fit and engineer features: {e}")
            raise
        return torch.tensor(engineered_features, dtype=torch.float32) if not issparse(engineered_features) else csr_matrix(engineered_features)

class CombatOverfit:
    """
    The CombatOverfit class provides a collection of methods to combat overfitting in neural networks.
    It includes regularization techniques, dropout, batch normalization, early stopping, data augmentation,
    data splitting, cross-validation, and learning rate reduction.

    Attributes:
        l1_regularization (float, optional): The L1 regularization coefficient.
        l2_regularization (float, optional): The L2 regularization coefficient.
        dropout_prob (float, optional): The dropout probability.
        batch_norm (bool, optional): Whether to use batch normalization.

    Usage:
        combat_overfit = CombatOverfit(l1_regularization=0.01, dropout_prob=0.5, batch_norm=True)
        l1_reg_loss = combat_overfit.apply_l1_regularization(model.parameters())
        dropout_output = combat_overfit.apply_dropout(output)
        normalized_output = combat_overfit.apply_batch_norm(output)
    """

    def __init__(self, l1_regularization=None, l2_regularization=None, dropout_prob=None, batch_norm=False, early_stopping_params=None):
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization  # New attribute for L2 regularization coefficient
        self.dropout_prob = dropout_prob
        if dropout_prob is not None and not (0 <= dropout_prob <= 1):
            raise ValueError("dropout_prob should be between 0 and 1")
        self.batch_norm = batch_norm
        self.dropout = None
        self.batch_norm_layer = None
        self.early_stopping = EarlyStopping(**early_stopping_params) if early_stopping_params else None
        self.scheduler = None

    def apply_l1_regularization(self, parameters):
        """
       Applies L1 regularization to the given parameters.

       Parameters:
           parameters (iterable): An iterable of torch parameters.

       Returns:
           torch.Tensor: The L1 regularization loss.
       """
       
        l1_reg = sum(param.abs().sum() for param in parameters)
        return self.l1_regularization * l1_reg if self.l1_regularization else 0

    def apply_l2_regularization(self, parameters):
        """
        Applies L2 regularization to the given parameters.

        Parameters:
            parameters (iterable): An iterable of torch parameters.

        Returns:
            torch.Tensor: The L2 regularization loss.
        """
        
        l2_reg = sum(param.pow(2).sum() for param in parameters)
        return self.l2_regularization * l2_reg if self.l2_regularization else 0  # New method for L2 regularization

    def apply_dropout(self, x):
        """
        Applies dropout to the given tensor.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor after applying dropout.
        """
        
        if self.dropout is None:
            self.dropout = nn.Dropout(self.dropout_prob)
        return self.dropout(x) if self.dropout_prob else x

    def apply_batch_norm(self, x):
        """
        Applies batch normalization to the given tensor.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor after applying batch normalization.
        """
        
        if self.batch_norm_layer is None:
            self.batch_norm_layer = nn.BatchNorm1d(x.size(1))
        return self.batch_norm_layer(x) if self.batch_norm else x
    
    def check_early_stopping(self, epoch, logs):
        """
        Checks for early stopping based on the logs provided.

        Parameters:
            epoch (int): The current epoch.
            logs (dict): A dictionary containing the logs for the current epoch.

        Returns:
            bool: True if early stopping should be performed, False otherwise.
        """
        if self.early_stopping:
            return self.early_stopping.on_epoch_end(epoch, logs)
        return False  # Continue training if early stopping is not configured
    
    def data_augmentation(self, augment_transforms):
        """
       Applies data augmentation using specified torchvision transforms.

       Parameters:
           augment_transform (callable): A torchvision transform for data augmentation.

       Returns:
           torchvision.transforms.Compose: A composed transform including the specified data augmentation transform.
       """
       
        return transforms.Compose(augment_transforms + [transforms.ToTensor()])
    
    def split_data(self, X, y, train_size=0.8):
        """
        Splits the data into training and validation sets while maintaining temporal coherence.

        Parameters:
            X (array-like): The input data.
            y (array-like): The target data.
            train_size (float): The proportion of the dataset to include in the train split.

        Returns:
            tuple: The training and validation data split.
        """
        
        # Determine the index at which to split the data
        split_idx = int(len(X) * train_size)

        # Split the data
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        return (X_train, y_train), (X_val, y_val)
    
    def cross_validate(self, model, X, y, n_splits=5):
        """
        Applies the ESN to multiple parts of the dataset and averages the loss.

        Parameters:
            X (array-like): The input data.
            y (array-like): The target data.
            n_splits (int): The number of splits for cross-validation.

        Returns:
            float: The average loss across all splits.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        total_loss = 0

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Reset the model parameters before each fold
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

            # Fit the model on the training data
            model.fit(X_train, y_train)

            # Evaluate the model on the validation data
            loss = model.evaluate(X_val, y_val)
            total_loss += loss

        average_loss = total_loss / n_splits
        return average_loss
    
    def learning_rate_reduction(self, optimizer, val_loss):
        """
        Reduces learning rate when validation loss plateaus.

        Parameters:
            optimizer (torch.optim.Optimizer): The optimizer.
            val_loss (float): The validation loss.

        Returns:
            None
        """
        if optimizer:
            if self.scheduler is None:
                self.scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
            self.scheduler.step(val_loss)
        
    def get_cross_validation_score(self, model, X, y, cv=5):
        """
        Computes cross-validation score.

        Parameters:
            model (nn.Module): The PyTorch model.
            X (array-like): The input data.
            y (array-like): The target data.
            cv (int, optional): The number of folds in cross-validation.

        Returns:
            float: The mean cross-validation score.
        """
        
        # Assuming model has a 'fit' method and a 'score' method
        tscv = TimeSeriesSplit(n_splits=cv)
        scores = []
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            scores.append(score)
        return np.mean(scores)

class EarlyStopping:
    """
   Implements early stopping to halt training when validation score is not improving.

   Attributes:
       patience (int): Number of epochs to wait before stopping after the best score has been found.
       monitor (str): Metric name to monitor for early stopping.
       mode (str): One of {'min', 'max'}. Whether to look for the minimum or maximum of the monitored quantity.

   Usage:
       early_stopping = EarlyStopping(patience=5, monitor='val_loss', mode='min')
       for epoch in range(num_epochs):
           # ... training logic ...
           val_loss = # ... compute validation loss ...
           if early_stopping.on_epoch_end(epoch, {'val_loss': val_loss}):
               print(f'Stopping early at epoch {epoch}')
               break
   """
   
    def __init__(self, patience: int = 10, monitor: str = 'val_loss', mode: str = 'min'):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.wait = 0
        assert mode in ['min', 'max'], "Mode must be 'min' or 'max'"
        
    def on_epoch_end(self, epoch: int, logs: dict) -> bool:
        if self.monitor not in logs:
            raise ValueError(f'Monitor key "{self.monitor}" not found in logs')
        current_score = logs[self.monitor]
        if self.best_score is None or (self.mode == 'min' and current_score < self.best_score) or (self.mode == 'max' and current_score > self.best_score):
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logging.info(f'Early stopping triggered at epoch {epoch}')
                return True  # Signal to stop training
            return False  # Continue training
                
class ESNReservoirLayer(nn.Module):
    """
    Echo State Network (ESN) Reservoir Layer implemented in PyTorch.

    Attributes:
        n_inputs (int): The number of input features.
        n_reservoir (int): The number of neurons in the reservoir.
        leaking_rate (float): The leaking rate of the reservoir neurons.
        spectral_radius (float): The spectral radius of the reservoir weight matrix.
        sparsity (float, optional): The sparsity of the reservoir weight matrix.
        noise (float, optional): The amount of noise to add to the reservoir state update.
        input_scaling (float, optional): The scaling of the input weights.
        feedback_scaling (float, optional): The scaling of the feedback weights.
        random_state (int, optional): Seed for reproducibility.
        l1_regularization (float, optional): The L1 regularization coefficient.

    Usage:
        reservoir_layer = ESNReservoirLayer(
            n_inputs=10, 
            n_reservoir=100, 
            leaking_rate=0.3, 
            spectral_radius=0.95
        )
        reservoir_output = reservoir_layer(input_data)

    """
    def __init__(self, n_inputs, n_reservoir, leaking_rate, spectral_radius, 
             sparsity=None, noise=None, input_scaling=None, 
             random_state=None):
        
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_scaling = input_scaling
        self.random_state = random_state
        
        # Initialize input weights
        self.Win = nn.Parameter(torch.randn(n_reservoir, n_inputs), requires_grad=False)
        
        # Initialize reservoir weights
        self.W = nn.Parameter(self.initialize_reservoir_weights(), requires_grad=False)
        
        # State of the reservoir
        self.register_buffer('state', torch.zeros(n_reservoir))
        
    def initialize_reservoir_weights(self):
        # Create a random reservoir weight matrix with the specified sparsity
        W = torch.rand(self.n_reservoir, self.n_reservoir, dtype=torch.float32) - 0.5
        if self.sparsity is not None:
            mask = torch.rand(self.n_reservoir, self.n_reservoir, dtype=torch.float32) > self.sparsity
            W[mask] = 0
        # Scale the eigenvalues to the specified spectral radius
        eigenvalues, eigenvectors = torch.eig(W, eigenvectors=False)
        W = (self.spectral_radius / torch.max(torch.abs(eigenvalues))) * W
        return W
        
    def forward(self, x):
        x = x.view(-1, self.n_inputs)  # Ensure the input is 2D (batch_size, n_inputs)
        pre_activation = torch.matmul(x, self.Win.t()) + torch.matmul(self.state.expand(x.size(0), -1), self.W.t())
        if self.noise is not None:
            pre_activation += self.noise * torch.randn_like(pre_activation)
        new_state = (1 - self.leaking_rate) * self.state + self.leaking_rate * torch.tanh(pre_activation)
        return new_state
        
    def reset_parameters(self):
        """
        Resets the model parameters to their initial states.
        """
        # Reset the state of the reservoir
        self.state.zero_()

        # Re-initialize input weights
        self.Win.data = torch.randn(self.n_reservoir, self.n_inputs)

        # Re-initialize reservoir weights
        self.W.data = self.initialize_reservoir_weights()

class CustomReadoutLayer(nn.Module):
    """
    Custom Readout Layer implemented in PyTorch.

    Attributes:
        n_inputs (int): The number of input features.
        n_outputs (int): The number of output features.
        activation_fn (callable, optional): The activation function to apply to the layer output.

    Usage:
        readout_layer = CustomReadoutLayer(n_inputs=100, n_outputs=1)
        readout_output = readout_layer(reservoir_output)

    """
    
    def __init__(self, n_inputs, n_outputs, activation_fn=None):
        super(CustomReadoutLayer, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x
    
    def reset_parameters(self):
        """
        Resets the model parameters. This method should be implemented to reset
        the parameters of the CustomReadoutLayer before each fold of cross-validation.
        """
        # Reset parameters of the linear layer
        self.linear.reset_parameters()
            
class EchoStateNetwork(nn.Module):
    """
    An implementation of an Echo State Network (ESN).

    Attributes:
        input_size (int): The number of input features.
        reservoir_size (int): The size of the reservoir.
        output_size (int): The number of output features.
        leaking_rate (float, optional): The leaking rate of the reservoir.
        spectral_radius (float, optional): The spectral radius of the reservoir weights.
        density (float, optional): The density of the reservoir weights.
        input_scaling (float, optional): The scaling of the input weights.
        bias_scaling (float, optional): The scaling of the bias.
        activation (callable, optional): The activation function to use.
        random_state (int, optional): Seed for reproducibility.
        device (torch.device, optional): The device to run the network on.

    Usage:
        esn = EchoStateNetwork(input_size=10, reservoir_size=100, output_size=1)
        output = esn(input_data)

    """

    def __init__(self, input_size, reservoir_size, output_size, 
             leaking_rate=0.3, spectral_radius=0.95, 
             input_scaling=1.0, random_state=None,
             combat_overfit_params=None):
        super(EchoStateNetwork, self).__init__()

        # Store parameters
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling

        # Optional: Seed for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
                
        # Instantiate CombatOverfit object
        self.combat_overfit = CombatOverfit(**combat_overfit_params) if combat_overfit_params else None
        
        # Instantiate ESNReservoirLayer
        self.reservoir_layer = ESNReservoirLayer(
            input_size=input_size,
            reservoir_size=reservoir_size,
            leaking_rate=leaking_rate,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            random_state=random_state
        )
        
        # Instantiate CustomReadoutLayer
        self.readout_layer = CustomReadoutLayer(
            n_inputs=reservoir_size,
            n_outputs=output_size
        )
        
    def forward(self, x):
        reservoir_output = self.reservoir_layer(x)
        readout_output = self.readout_layer(reservoir_output)
        return readout_output

    def reset_parameters(self):
        """
        Resets the model parameters. This method should be implemented to reset
        the parameters of the ESN before each fold of cross-validation.
        """
        self.reservoir_layer.reset_parameters()
        self.readout_layer.reset_parameters()

        # Reset parameters of the CombatOverfit object, if it exists
        if self.combat_overfit and hasattr(self.combat_overfit, 'batch_norm_layer'):
            self.combat_overfit.batch_norm_layer.reset_parameters()

class DeepESNModule(nn.Module):
    """
    A Deep Echo State Network (DeepESN) module implemented in PyTorch.

    Attributes:
        n_inputs (int): The number of input features.
        n_outputs (int): The number of output features.
        n_reservoirs (int): The number of reservoir layers.
        reservoir_sizes (list): A list of sizes for each reservoir layer.
        leaking_rates (list): A list of leaking rates for each reservoir layer.
        spectral_radiuses (list): A list of spectral radiuses for each reservoir layer.
        regression_parameters (list): A list of regression parameters for each reservoir layer.
        sparsity (float, optional): The sparsity of the reservoir weights.
        noise (float, optional): The amount of noise to add to the reservoir state update.
        input_scaling (float, optional): The scaling of the input weights.
        feedback_scaling (float, optional): The scaling of the feedback weights.
        random_state (int, optional): Seed for reproducibility.
        l1_regularization (float, optional): The L1 regularization coefficient.
        custom_layer (nn.Module, optional): A custom layer to insert after the reservoir layers.
        dropout_prob (float, optional): The dropout probability.
        batch_norm (bool, optional): Whether to use batch normalization.
        learning_rate (float, optional): The learning rate for the optimizer.
        scheduler_step_size (int, optional): The step size for the learning rate scheduler.
        scheduler_gamma (float, optional): The gamma value for the learning rate scheduler.
        device (torch.device, optional): The device to run the network on.
        readout_activation_fn (callable, optional): The activation function for the readout layer.

    Usage:
        deep_esn = DeepESNModule(
            n_inputs=10, 
            n_outputs=1, 
            n_reservoirs=3, 
            reservoir_sizes=[100, 100, 100], 
            leaking_rates=[0.3, 0.3, 0.3], 
            spectral_radiuses=[0.95, 0.95, 0.95], 
            regression_parameters=[1e-3, 1e-3, 1e-3]
        )
        output = deep_esn(input_data)

    """
    
    def __init__(self, 
                 n_inputs: int, 
                 n_outputs: int, 
                 n_reservoirs: int, 
                 reservoir_sizes: List[int], 
                 leaking_rates: List[float], 
                 spectral_radiuses: List[float], 
                 regression_parameters: List[float], 
                 sparsity: Optional[float] = None, 
                 noise: Optional[float] = None, 
                 input_scaling: Optional[float] = None, 
                 feedback_scaling: Optional[float] = None,
                 random_state: Optional[int] = None,
                 l1_regularization: Optional[float] = None,
                 custom_layer: Optional[nn.Module] = None,
                 dropout_prob: Optional[float] = None,
                 batch_norm: bool = False,
                 learning_rate: float = 0.001,
                 scheduler_step_size: int = 10,
                 scheduler_gamma: float = 0.1,
                 device: Optional[torch.device] = None,
                 combat_overfit: Optional[nn.Module] = None,
                 readout_activation_fn: Optional[callable] = None,
                 optimizer_config: Optional[dict] = None,
                 loss_fn: Optional[nn.Module] = None):
        super(DeepESNModule, self).__init__()
        if not (isinstance(n_inputs, int) and n_inputs > 0):
            raise ValueError(f"Invalid n_inputs: {n_inputs}")
            
        # Validate configurations
        assert len(reservoir_sizes) == len(leaking_rates) == len(spectral_radiuses) == n_reservoirs, \
            "Mismatch in the length of hyperparameters arrays"

        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize reservoir layers
        self.reservoir_layers = nn.ModuleList()
        for i in range(n_reservoirs):
            reservoir_layer = ESNReservoirLayer(
                n_inputs if i == 0 else reservoir_sizes[i-1],  # input size
                reservoir_sizes[i],  # reservoir size
                leaking_rates[i],
                spectral_radiuses[i],
                sparsity=sparsity,
                noise=noise,
                input_scaling=input_scaling,
                feedback_scaling=feedback_scaling,
                random_state=random_state
            ).to(self.device)  # Ensure layer is on the correct device
            self.reservoir_layers.append(reservoir_layer)

        # Initialize custom layer if provided
        self.custom_layer = custom_layer.to(self.device) if custom_layer else None  # Ensure layer is on the correct device

        # Initialize readout layer
        self.readout_layer = CustomReadoutLayer(reservoir_sizes[-1], n_outputs, readout_activation_fn)  # Ensure layer is on the correct device
        
        # Initialize loss function
        self.loss_fn = loss_fn or nn.MSELoss()

        # Optimizer configuration
        self.optimizer_config = optimizer_config
        
        # Combating overfitting methods
        self.combat_overfit = combat_overfit or CombatOverfit()

    def forward(self, x):
        # Initial input for the first reservoir layer
        reservoir_input = x

        # List to hold the outputs of each reservoir layer
        reservoir_outputs = []

        # Pass the input through each reservoir layer
        for reservoir_layer in self.reservoir_layers:
            reservoir_output = reservoir_layer(reservoir_input)
            reservoir_outputs.append(reservoir_output)
            # The output of the current layer is the input to the next layer
            reservoir_input = reservoir_output

        # If a custom layer is provided, pass the output of the final reservoir layer through it
        if self.custom_layer:
            custom_layer_output = self.custom_layer(reservoir_outputs[-1])
            # Update the final reservoir output to be the output of the custom layer
            reservoir_outputs[-1] = custom_layer_output

        # If dropout is enabled, apply dropout to the output of the final reservoir layer
        reservoir_outputs[-1] = self.combat_overfit.apply_dropout(reservoir_outputs[-1])

        # If batch normalization is enabled, apply batch normalization to the output of the final reservoir layer
        reservoir_outputs[-1] = self.combat_overfit.apply_batch_norm(reservoir_outputs[-1])

        # Pass the output of the final reservoir layer (or custom layer, if provided) through the readout layer
        readout_output = self.readout_layer(reservoir_outputs[-1])

        return readout_output

    def training_step(self, batch, batch_idx):
        # Get data from batch
        x, y = batch

        # Forward pass
        y_hat = self.forward(x)

        # Compute loss
        loss = self.loss_fn(y_hat, y)  # Use the initialized loss function

        # Optionally, add L1 and L2 regularization
        if self.combat_overfit.l1_regularization:
            l1_reg = self.combat_overfit.apply_l1_regularization(self.parameters())
            loss += l1_reg
        if self.combat_overfit.l2_regularization:  # Check for L2 regularization
            l2_reg = self.combat_overfit.apply_l2_regularization(self.parameters())
            loss += l2_reg

        # Log training loss
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, early_stopping_obj):
        # Get data from batch
        x, y = batch

        # Forward pass
        y_hat = self.forward(x)

        # Compute loss
        loss = self.loss_fn(y_hat, y)  # Use the initialized loss function

        # Compute RMSE and MAE
        rmse = torch.sqrt(loss)
        mae = nn.L1Loss()(y_hat, y)

        # Log validation metrics
        self.log('val_loss', loss)
        self.log('val_rmse', rmse)
        self.log('val_mae', mae)
        
        stop_training = self.combat_overfit.early_stopping(loss, early_stopping_obj)
        if stop_training:
            return None  # or handle early stopping as needed
        return loss

    def fit(self, X_train, y_train, optimizer=None):
        self.train()  # Set the module to training mode
        optimizer = optimizer or torch.optim.Adam(self.parameters())
        loss_fn = self.loss_fn  # Use the initialized loss function
        
        # Convert data to torch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device) if isinstance(X_train, (np.ndarray, list)) else X_train.to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device) if isinstance(y_train, (np.ndarray, list)) else y_train.to(self.device)

        
        optimizer.zero_grad()  # Reset gradients
        outputs = self.forward(X_train)  # Forward pass
        loss = loss_fn(outputs, y_train)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        # Apply learning rate reduction
        self.combat_overfit.learning_rate_reduction(optimizer, loss.item())
        return loss.item()  # Return the loss value

    def evaluate(self, X_val, y_val):
        self.eval()  # Set the module to evaluation mode
        loss_fn = self.loss_fn  # Use the initialized loss function
        
        # Convert data to torch tensors and move to the device
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device) if isinstance(X_val, (np.ndarray, list)) else X_val.to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device) if isinstance(y_val, (np.ndarray, list)) else y_val.to(self.device)
        
        with torch.no_grad():  # Disable gradient computation
            outputs = self.forward(X_val)  # Forward pass
            loss = loss_fn(outputs, y_val)  # Compute loss
            
        # Get cross-validation score
        cv_score = self.combat_overfit.get_cross_validation_score(self, X_val, y_val)
        print(f'Cross-validation score: {cv_score}')
        
        return loss.item()  # Return the loss value

    def predict(self, X_test):
        self.eval()  # Set the module to evaluation mode
        
        # Convert data to torch tensors and move to the device
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device) if isinstance(X_test, (np.ndarray, list)) else X_test.to(self.device)
        
        with torch.no_grad():  # Disable gradient computation
            outputs = self.forward(X_test)  # Forward pass
        
        return outputs.numpy()  # Convert outputs to numpy array and return

    def configure_optimizers(self):
        optimizer_config = self.optimizer_config or {
            'optimizer': 'RAdam',
            'lr': self.learning_rate,
            'la_steps': 5,
            'la_alpha': 0.8,
            'scheduler_factor': 0.1,
            'scheduler_patience': 10
        }

        # Initialize RAdam optimizer
        radam_optimizer = timm.optim.RAdam(
            self.parameters(),
            lr=optimizer_config['lr']
        )
        
        # Wrap RAdam with Lookahead
        lookahead_optimizer = timm.optim.Lookahead(
            radam_optimizer,
            la_steps=optimizer_config['la_steps'],
            la_alpha=optimizer_config['la_alpha']
        )
        
        # Initialize scheduler with the Lookahead optimizer
        scheduler = ReduceLROnPlateau(
            lookahead_optimizer,
            mode='min',
            factor=optimizer_config['scheduler_factor'],
            patience=optimizer_config['scheduler_patience'],
            verbose=True
        )
        
        return {
            'optimizer': lookahead_optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # or 'step'
            }
        }
    
    def reset_parameters(self):
        """
        Resets the model parameters. This method should be implemented to reset
        the parameters of the ESN before each fold of cross-validation.
        """
        # Reset parameters of each reservoir layer
        for reservoir_layer in self.reservoir_layers:
            reservoir_layer.reset_parameters()

        # Reset parameters of the readout layer
        self.readout_layer.reset_parameters()

        # Reset parameters of the custom layer, if it exists
        if self.custom_layer:
            self.custom_layer.reset_parameters()

        # Reset parameters of the batch normalization layer, if it exists
        if self.combat_overfit.batch_norm:
            self.combat_overfit.batch_norm.reset_parameters()
        
class AttentionNetwork(nn.Module):
    """
    A network for applying attention to the input data.

    Attributes:
        input_dim (int): The dimensionality of the input data.
        attention_dim (int): The dimensionality of the attention space.

    Usage:
        attention_network = AttentionNetwork(input_dim=10, attention_dim=5)
        attended_data, attention_weights = attention_network(input_data)

    """

    def __init__(self, input_dim: int, attention_dim: int):
       super().__init__()
       if not isinstance(input_dim, int) or input_dim <= 0:
           raise ValueError(f"Invalid input_dim: {input_dim}")
       if not isinstance(attention_dim, int) or attention_dim <= 0:
           raise ValueError(f"Invalid attention_dim: {attention_dim}")
       self.attention_layer = nn.Linear(input_dim, attention_dim)
       self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attention_weights = self.compute_attention_weights(x)
        attended_data = self.apply_attention(x, attention_weights)
        return attended_data, attention_weights
    
    def compute_attention_weights(self, x):
        attention_logits = self.attention_layer(x)
        attention_weights = self.softmax(attention_logits)
        return attention_weights
    
    def apply_attention(self, x, attention_weights):
        attended_data = x * attention_weights.unsqueeze(-1)
        return attended_data

class DeepESNModuleWithAttention(DeepESNModule):
    """
    A Deep Echo State Network (DeepESN) module with an attention mechanism.

    Attributes:
        input_dim (int): The dimensionality of the input data.
        reservoir_dim (int): The dimensionality of the reservoir space.
        output_dim (int): The dimensionality of the output space.
        attention_dim (int): The dimensionality of the attention space.

    Usage:
        deep_esn_attention = DeepESNModuleWithAttention(
            input_dim=10, 
            reservoir_dim=100, 
            output_dim=1, 
            attention_dim=5
        )
        output = deep_esn_attention(input_data)

    """
    
    def __init__(self, input_dim: int, reservoir_dim: int, output_dim: int, attention_dim: int):
        super().__init__(input_dim, reservoir_dim, output_dim)
        self.attention_network = AttentionNetwork(input_dim, attention_dim)
        self.logger = logging.getLogger(__name__)

    def evaluate(self, dataloader):
        # Ensure dataloader is compatible
        all_outputs = []
        all_attention_weights = []
        for data, targets in dataloader:
            attended_data, attention_weights = self.attention_network(data)
            outputs = super().evaluate(attended_data)  # Assuming evaluate returns outputs
            all_outputs.extend(outputs.cpu().numpy())
            all_attention_weights.extend(attention_weights.cpu().numpy())
        return np.array(all_outputs), np.array(all_attention_weights)
    
    def forward(self, x):
        self.logger.debug(f"Input shape: {x.shape}")
        attended_data, attention_weights = self.attention_network(x)
        self.logger.debug(f"Attended data shape: {attended_data.shape}")
        return super().forward(attended_data)

class CustomDataset(Dataset):
    """
    Custom Dataset class for loading data into PyTorch DataLoader.

    Attributes:
        data (Tensor): The input data.
        targets (Tensor): The target labels.

    Usage:
        custom_dataset = CustomDataset(data=input_data, targets=target_labels)
        dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

    """
    
    def __init__(self, data, targets):
        assert len(data) == len(targets), "Mismatched lengths: data and targets"
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class ESNUtilities:
    """
    Utility class for training and evaluating a model.

    Usage:
        # Define the model, criterion, optimizer, and dataloader
        model = ...
        criterion = ...
        optimizer = ...
        train_dataloader = ...
        val_dataloader = ...

        # Train the model
        train_loss = ESNUtilities.train(model, train_dataloader, criterion, optimizer)

        # Evaluate the model
        val_loss, val_outputs, val_targets = ESNUtilities.evaluate(model, val_dataloader, criterion)

    """
    
    @staticmethod
    def train(model, dataloader, criterion, optimizer):
        """
        Trains the model for one epoch and returns the average training loss.

        Args:
            model (nn.Module): The model to train.
            dataloader (DataLoader): The DataLoader for the training data.
            criterion (nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.

        Returns:
            float: The average training loss for the epoch.
        """
        
        model.train()
        total_loss = 0.0
        for data, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    @staticmethod
    def evaluate(model, dataloader, criterion):
        """
        Evaluates the model on the validation data and returns the average validation loss,
        the model outputs, and the target labels.

        Args:
            model (nn.Module): The model to evaluate.
            dataloader (DataLoader): The DataLoader for the validation data.
            criterion (nn.Module): The loss function.

        Returns:
            float: The average validation loss.
            Tensor: The model outputs.
            Tensor: The target labels.
        """
        
        model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        with torch.no_grad():  # Disable gradient computation
            all_outputs = []
            all_targets = []
            for data, targets in dataloader:
                outputs = model(data)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                all_outputs.append(outputs)
                all_targets.append(targets)
        return total_loss / len(dataloader), torch.cat(all_outputs), torch.cat(all_targets)
    
class ESNHyperparameterTuning:
    """
    This class provides functionality for hyperparameter tuning of a Deep Echo State Network (DeepESN) module.
    It utilizes the Optuna framework for optimizing hyperparameters to minimize the validation loss.

    Attributes:
        deep_esn_module_class (class): The class of the DeepESN module to be optimized.
        training_data (array-like): The training data.
        training_labels (array-like): The labels for the training data.
        validation_data (array-like): The validation data.
        validation_labels (array-like): The labels for the validation data.
        cache (dict): A cache to store validation loss for a set of hyperparameters to avoid re-evaluation.

    Usage:
        # Assume DeepESNModule is the class of the DeepESN module to be optimized
        tuner = ESNHyperparameterTuning(DeepESNModule, train_data, train_labels, val_data, val_labels)
        best_params = tuner.tune_hyperparameters(n_trials=50)
        best_model = tuner.train_best_model(best_params)
    """
    
    def __init__(self, deep_esn_module_class, training_data, training_labels, validation_data, validation_labels):
        self.deep_esn_module_class = deep_esn_module_class
        self.training_data = training_data
        self.training_labels = training_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.cache = {}  # Initialize cache
        
    def hash_params(self, params):
        """
       Generates a hash for a set of hyperparameters to enable caching of results.

       Args:
           params (dict): A dictionary of hyperparameters.

       Returns:
           str: A hash string representing the set of hyperparameters.
       """
       
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()

    def objective(self, trial):
        """
        Objective function for Optuna optimization. It defines the hyperparameters to be optimized,
        and computes the validation loss for a given set of hyperparameters.

        Args:
            trial (optuna.trial.Trial): A trial object that suggests hyperparameters.

        Returns:
            float: The validation loss for the current set of hyperparameters.
        """
        
        # Hyperparameters to be optimized
        n_reservoirs = trial.suggest_int('n_reservoirs', 1, 3)
        reservoir_sizes = [trial.suggest_int(f'reservoir_size_{i}', 50, 500) for i in range(n_reservoirs)]
        leaking_rates = [trial.suggest_float(f'leaking_rate_{i}', 0.1, 1.0) for i in range(n_reservoirs)]
        spectral_radiuses = [trial.suggest_float(f'spectral_radius_{i}', 0.1, 1.5) for i in range(n_reservoirs)]
        regression_parameters = [trial.suggest_float(f'regression_parameter_{i}', 0.1, 1.0) for i in range(n_reservoirs)]
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Generate a hash for the current set of hyperparameters
        params_hash = self.hash_params({
            'n_reservoirs': n_reservoirs,
            'reservoir_sizes': reservoir_sizes,
            'leaking_rates': leaking_rates,
            'spectral_radiuses': spectral_radiuses,
            'regression_parameters': regression_parameters,
            'learning_rate': learning_rate,
            'dropout_rate': dropout_rate,
            'batch_size': batch_size
        })

        # Check if these hyperparameters have been evaluated before
        if params_hash in self.cache:
            return self.cache[params_hash]  # Return cached validation loss

        # Create an instance of DeepESNModule with the suggested hyperparameters
        deep_esn_module = self.deep_esn_module_class(
            n_reservoirs=n_reservoirs,
            reservoir_sizes=reservoir_sizes,
            leaking_rates=leaking_rates,
            spectral_radiuses=spectral_radiuses,
            regression_parameters=regression_parameters,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            batch_size=batch_size
        )

        # Train the model with early stopping
        early_stopping_callback = EarlyStopping(patience=5, monitor='val_loss', mode='min')
        deep_esn_module.fit(self.training_data, self.training_labels, callbacks=[early_stopping_callback])

        # Evaluate the model on validation data
        validation_loss = deep_esn_module.evaluate(self.validation_data, self.validation_labels)
        
        # Cache the validation loss for these hyperparameters
        self.cache[params_hash] = validation_loss

        return validation_loss  # Objective is to minimize validation loss

    def tune_hyperparameters(self, n_trials=100, db_url='sqlite:///example.db'):
        """
        Initiates the hyperparameter tuning process using Optuna.

        Args:
            n_trials (int, optional): The number of trials for hyperparameter optimization. Default is 100.

        Returns:
            dict: A dictionary of the best hyperparameters found.
        """
        
        # Set up a storage database for parallelization
        storage = optuna.storages.RDBStorage(
            url=db_url,
            engine_kwargs={'connect_args': {'timeout': 30}}
        )
        pruner = optuna.pruners.HyperbandPruner()
        study = optuna.create_study(direction='minimize', storage=storage, study_name='esn_study', load_if_exists=True, pruner=pruner)
        study.optimize(self.objective, n_trials=n_trials, n_jobs=-1)  # n_jobs=-1 will use all available cores
        return study.best_params

    def train_best_model(self, best_params):
        """
        Trains a DeepESN module with the best hyperparameters found during tuning.

        Args:
            best_params (dict): A dictionary of the best hyperparameters.

        Returns:
            DeepESNModule: An instance of DeepESNModule trained with the best hyperparameters.
        """
        
        # Create an instance of DeepESNModule with the best hyperparameters
        best_deep_esn_module = self.deep_esn_module_class(
            n_reservoirs=best_params['n_reservoirs'],
            reservoir_sizes=[best_params[f'reservoir_size_{i}'] for i in range(best_params['n_reservoirs'])],
            leaking_rates=[best_params[f'leaking_rate_{i}'] for i in range(best_params['n_reservoirs'])],
            spectral_radiuses=[best_params[f'spectral_radius_{i}'] for i in range(best_params['n_reservoirs'])],
            regression_parameters=[best_params[f'regression_parameter_{i}'] for i in range(best_params['n_reservoirs'])],
            learning_rate=best_params['learning_rate'],
            dropout_rate=best_params['dropout_rate'],
            batch_size=best_params['batch_size']
        )

        # Train the model with the best hyperparameters
        best_deep_esn_module.fit(self.training_data, self.training_labels)
        return best_deep_esn_module

class ESNLogger:
    """
    This class provides logging and plotting utilities for monitoring the training and evaluation of a DeepESN module.

    Usage:
        logger = ESNLogger()
        logger.log_training(epoch, loss, metrics)
        logger.log_evaluation(loss, metrics)
        logger.plot_performance(metrics_history)
    """
    
    def log_training(self, epoch, loss, metrics):
        """
        Logs the training loss and metrics for a given epoch.

        Args:
            epoch (int): The current epoch.
            loss (float): The training loss.
            metrics (dict): A dictionary of training metrics.
        """
        
        print(f'Epoch {epoch}, Loss: {loss}, Metrics: {metrics}')

    def log_evaluation(self, loss, metrics):
        """
        Logs the validation loss and metrics.

        Args:
            loss (float): The validation loss.
            metrics (dict): A dictionary of validation metrics.
        """
        
        print(f'Validation Loss: {loss}, Metrics: {metrics}')

    def plot_performance(self, metrics_history):
        """
        Plots the training and validation performance over epochs.

        Args:
            metrics_history (dict): A dictionary where keys are metric names and values are lists of metric values over epochs.
        """
        
        epochs = range(1, len(next(iter(metrics_history.values()))) + 1)  # Get the length of one of the metric histories
        plt.figure()
        for metric, values in metrics_history.items():
            plt.plot(epochs, values, label=metric)
        plt.legend()
        plt.show()
    
class ModelManager:
    """
   Manages saving and loading of PyTorch models.

   Attributes:
       model (torch.nn.Module, optional): The PyTorch model to manage.

   Usage:
       model_manager = ModelManager(model)
       model_manager.save_model('model.pth')
       model_manager.load_model(MyModelClass, 'model.pth')
   """
   
    def __init__(self, model=None):
        self.model = model

    def save_model(self, file_path):
        """
        Save the model to the specified file path.
        
        Parameters:
        - file_path (str): The path where the model will be saved.
        """
        if self.model is None:
            raise ValueError("No model is set to be saved.")
        torch.save(self.model.state_dict(), file_path)
        print(f'Model saved to {file_path}')

    def load_model(self, model, file_path):
        """
        Load the model from the specified file path.
        
        Parameters:
        - model_class (torch.nn.Module): The class of the model to be loaded.
        - file_path (str): The path from where the model will be loaded.
        """
        self.model = model
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()  # Set the model to evaluation mode
        print(f'Model loaded from {file_path}')

class MyObjective:
    """
    Defines an objective function for Optuna hyperparameter optimization.

    Attributes:
        data (torch.Tensor): The input data.
        target (torch.Tensor): The target labels.
        evaluation_method (str): The evaluation method name as defined in the ModelEvaluator class.
        previous_params (dict, optional): Previous hyperparameters to use as defaults.
        maximize (bool, optional): Whether to maximize the objective. Default is False.
        batch_size (int, optional): Batch size for data loading. Default is 32.

    Usage:
        objective = MyObjective(data, target, 'accuracy_score', previous_params={'learning_rate': 1e-3}, maximize=True)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
    """
    
    def __init__(self, data, target, evaluation_method, previous_params=None, maximize=False, batch_size=32):
        self.data = torch.tensor(data, dtype=torch.float32) if isinstance(data, (list, np.ndarray)) else data
        self.target = torch.tensor(target, dtype=torch.float32) if isinstance(target, (list, np.ndarray)) else target
        self.evaluation_method = evaluation_method
        self.previous_params = previous_params or {}  # Store previous hyperparameters
        self.maximize = maximize  # Whether to maximize the objective
        self.dataloader = DataLoader(TensorDataset(self.data, self.target), batch_size=batch_size, shuffle=True)
        self.logger = logging.getLogger(__name__)  # Initialize logger

    def __call__(self, trial: Trial):
        """
        Defines the objective for Optuna optimization.

        Args:
            trial (optuna.trial.Trial): The trial object.

        Returns:
            float: The objective value.
        """
        
        # Default hyperparameter ranges
        param = {
            'n_inputs': self.data.shape[1],
            'n_outputs': 1,
            'n_reservoirs': trial.suggest_int('n_reservoirs', 1, 3),
            'reservoir_sizes': trial.suggest_categorical('reservoir_sizes', [(32,), (64,), (128,)]),
            'leaking_rates': trial.suggest_float('leaking_rates', 0.1, 1.0),
            'spectral_radiuses': trial.suggest_float('spectral_radiuses', 0.1, 1.5),
            'regression_parameters': trial.suggest_float('regression_parameters', 0.1, 1.0),
            'dropout_prob': trial.suggest_float('dropout_prob', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'scheduler_step_size': trial.suggest_int('scheduler_step_size', 5, 20),
            'scheduler_gamma': trial.suggest_float('scheduler_gamma', 0.1, 0.9),
        }
        
        # Override with previous hyperparameters if provided
        param.update(self.previous_params)

        param['attention_dim'] = trial.suggest_int('attention_dim', 1, self.data.shape[1])  # Corrected n_inputs to self.data.shape[1]
        deep_esn_module_with_attention = DeepESNModuleWithAttention(**param)
        
        # Assuming num_epochs and loss_fn are defined or passed as arguments
        num_epochs = 10  # Example value
        loss_fn = torch.nn.MSELoss()  # Example loss function
        
        # Train the model using the fit method from DeepESNModule class
        for epoch in range(num_epochs):
            train_loss = deep_esn_module_with_attention.fit(self.data.numpy(), self.target.numpy(), loss_fn=loss_fn)
            val_loss = deep_esn_module_with_attention.evaluate(self.data.numpy(), self.target.numpy(), loss_fn=loss_fn)
            print(f'Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}')
        
        # Evaluate the model using the evaluate method from DeepESNModule class
        predicted_values = deep_esn_module_with_attention.predict(self.data.numpy())
        evaluator = ModelEvaluator(self.target.numpy(), predicted_values)
        
        try:
            evaluation_score = getattr(evaluator, self.evaluation_method)()
        except AttributeError:
            raise ValueError(f"Evaluation method '{self.evaluation_method}' not found in ModelEvaluator class.")
            
        if isinstance(deep_esn_module_with_attention, DeepESNModuleWithAttention):
            # ... handle models with attention ...
            # For example, you might want to log attention weights or use them in some other way
            _, attention_weights = deep_esn_module_with_attention.attention_network(self.data)  # Corrected deep_esn_module to deep_esn_module_with_attention
            self.logger.info(f"Attention weights: {attention_weights}")
        
        # If maximizing, return the negative of the evaluation score so Optuna minimizes the negative value (i.e., maximizes the original value)
        return -evaluation_score if self.maximize else evaluation_score
    
class VisualizationTools:
    """
   A collection of static methods for visualizing various data types and model components.

   Usage:
       VisualizationTools.plot_reservoir_states(reservoir_states)
       VisualizationTools.plot_weight_matrix(weight_matrix)
       VisualizationTools.plot_learning_curve(train_losses, val_losses)
       VisualizationTools.plot_attention_weights(attention_weights)
   """
   
    @staticmethod
    def plot_reservoir_states(reservoir_states, title='Reservoir States'):
        """
        Plots the reservoir states over time.

        Parameters:
        - reservoir_states (array-like): The reservoir states to be plotted.
        - title (str, optional): The title of the plot. Default is 'Reservoir States'.
        """
        
        plt.figure(figsize=(10, 6))
        plt.plot(reservoir_states)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('State Value')
        plt.show()
        
    @staticmethod
    def plot_weight_matrix(weight_matrix, title='Weight Matrix'):
        plt.figure(figsize=(10, 6))
        sns.heatmap(weight_matrix, cmap='coolwarm', center=0)
        plt.title(title)
        plt.show()
        
    @staticmethod
    def plot_learning_curve(train_losses, val_losses, title='Learning Curve'):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    @staticmethod
    def plot_attention_weights(attention_weights, title='Attention Weights'):
        plt.figure(figsize=(10, 6))
        plt.plot(attention_weights)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Weight')
        plt.show()

class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance using a Random Forest Regressor.

    Attributes:
        model (object): The model to analyze.

    Usage:
        analyzer = FeatureImportanceAnalyzer(model)
        feature_importances = analyzer.analyze(X, y)
        analyzer.plot_feature_importances(feature_importances, feature_names)
    """
    
    def __init__(self, model):
        self.model = model

    def analyze(self, X, y):
        rf = RandomForestRegressor()
        rf.fit(X, y)
        feature_importances = rf.feature_importances_
        return feature_importances

    def plot_feature_importances(self, feature_importances, feature_names):
        plt.figure(figsize=(10, 6))
        plt.bar(feature_names, feature_importances)
        plt.title('Feature Importances')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.show()
        
class ErrorAnalysisTools:
    """
    A collection of static methods for performing error analysis.

    Usage:
        ErrorAnalysisTools.plot_confusion_matrix(y_true, y_pred, class_names)
        ErrorAnalysisTools.plot_residuals(y_true, y_pred)
        ErrorAnalysisTools.plot_error_distribution(y_true, y_pred)
        ErrorAnalysisTools.plot_feature_importance(X, y)
    """
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names):
        """
        Plots a confusion matrix.

        Parameters:
        - y_true (array-like): True labels.
        - y_pred (array-like): Predicted labels.
        - class_names (list): Names of the classes.
        """
        
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    @staticmethod
    def plot_residuals(y_true, y_pred):
        residuals = y_true - y_pred
        sns.residplot(x=y_pred, y=residuals, lowess=True, color="g")
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.show()
        
    @staticmethod
    def plot_error_distribution(y_true, y_pred):
        errors = y_true - y_pred
        sns.histplot(errors, kde=True)
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.show()
        
    @staticmethod
    def plot_feature_importance(X, y):
        rf = RandomForestRegressor()
        rf.fit(X, y)
        feature_importances = rf.feature_importances_
        plt.bar(range(len(feature_importances)), feature_importances)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.show()
        
class TemporalFeatures:
    """
    Engineered temporal features from a given time-series data.

    Attributes:
        data (pd.DataFrame): The input data.

    Usage:
        temporal_features = TemporalFeatures(data)
        temporal_features.add_moving_averages().add_exponential_moving_averages().add_rsi().add_macd().add_bollinger_bands()
        engineered_data = temporal_features.get_engineered_data()
    """
    
    def __init__(self, data):
        """
       Initializes the TemporalFeatures instance with the input data.

       Parameters:
       - data (pd.DataFrame): The input data.
       """
       
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data should be a Pandas DataFrame")
        self.data = data.copy()  # Create a copy of the data to avoid modifying the original DataFrame
        self.logger = logging.getLogger(__name__)
        self.logger.info("TemporalFeatures instance created")

    def check_column_existence(self, cols):
        """
        Checks the existence of specified columns in the data.

        Parameters:
        - cols (list): List of column names to check.

        Returns:
        - self: The current TemporalFeatures instance to allow for method chaining.
        """
        
        for col in cols:
            if col not in self.data.columns:
                raise ValueError(f"{col} column not found in DataFrame")
        return self  # Return self to allow for method chaining

    def add_moving_averages(self, short_window=12, long_window=26):
        """
        Adds short and long-term moving averages to the data.

        Parameters:
        - short_window (int, optional): Window size for short-term moving average. Default is 12.
        - long_window (int, optional): Window size for long-term moving average. Default is 26.

        Returns:
        - self: The current TemporalFeatures instance to allow for method chaining.
        """
        
        self.check_column_existence(['Close'])
        self.data['short_mavg'] = self.data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
        self.data['long_mavg'] = self.data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
        self.logger.info("Added moving averages")
        return self  # Return self to allow for method chaining

    def add_exponential_moving_averages(self, short_window=12, long_window=26):
        self.check_column_existence(['Close'])
        self.data['short_ema'] = self.data['Close'].ewm(span=short_window, adjust=False).mean()
        self.data['long_ema'] = self.data['Close'].ewm(span=long_window, adjust=False).mean()
        self.logger.info("Added exponential moving averages")
        return self  # Return self to allow for method chaining

    def add_rsi(self, window=14):
        self.check_column_existence(['Close'])
        self.data['rsi'] = ta.momentum.RSIIndicator(self.data['Close'], window).rsi()
        self.logger.info("Added RSI")
        return self  # Return self to allow for method chaining

    def add_macd(self):
        self.check_column_existence(['Close'])
        macd_indicator = ta.trend.MACD(self.data['Close'])
        self.data['macd'] = macd_indicator.macd()
        self.data['macd_signal'] = macd_indicator.macd_signal()
        self.data['macd_diff'] = macd_indicator.macd_diff()
        self.logger.info("Added MACD")
        return self  # Return self to allow for method chaining

    def add_bollinger_bands(self, window=20, num_std_dev=2):
        self.check_column_existence(['Close'])
        bollinger = ta.volatility.BollingerBands(self.data['Close'], window, num_std_dev)
        self.data['bollinger_hband'] = bollinger.bollinger_hband()
        self.data['bollinger_lband'] = bollinger.bollinger_lband()
        self.data['bollinger_mavg'] = bollinger.bollinger_mavg()
        self.logger.info("Added Bollinger Bands")
        return self  # Return self to allow for method chaining

    def add_vwap(self):
        """Adds Volume Weighted Average Price (VWAP) to the data."""
        self.check_column_existence(['High', 'Low', 'Close', 'Volume'])
        self.data['vwap'] = ta.volume.VolumeWeightedAveragePrice(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume']).volume_weighted_average_price()
        self.logger.info("Added VWAP")
        return self  # Return self to allow for method chaining

    def add_stochastic_oscillator(self, window=14):
        """Adds Stochastic Oscillator to the data."""
        self.check_column_existence(['High', 'Low', 'Close'])
        self.data['stochastic_oscillator'] = ta.momentum.StochasticOscillator(self.data['High'], self.data['Low'], self.data['Close'], window).stoch()
        self.logger.info("Added stochastic oscillator")
        return self  # Return self to allow for method chaining

    def add_datetime_features(self, datetime_col):
        """Adds datetime-related features to the data."""
        self.check_column_existence([datetime_col])
        # ... (rest of the code remains unchanged)
        self.logger.info("Added datetime features")
        return self  # Return self to allow for method chaining

    def add_lagged_features(self, price_col, lags):
        """Adds lagged features to the data."""
        self.check_column_existence([price_col])
        # ... (rest of the code remains unchanged)
        self.logger.info(f"Added lagged features with lags: {lags}")
        return self  # Return self to allow for method chaining
    
    @staticmethod
    def get_fear_and_greed_index():
        api_endpoint = "https://api.alternative.me/fng/"
        params = {"limit": 1}  # Get the latest index value
        response = requests.get(api_endpoint, params=params)
        if response.status_code == 200:
            data = response.json()
            return int(data['data'][0]['value'])
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            return None

    def add_fear_and_greed_index(self):
        index_value = self.get_fear_and_greed_index()
        if index_value is not None:
            # Assuming your DataFrame has a date index and you want to add the index value to the latest date
            latest_date = self.data.index[-1]
            self.data.loc[latest_date, 'fear_and_greed_index'] = index_value
            self.logger.info("Added Fear and Greed Index")
        else:
            self.logger.warning("Failed to retrieve Fear and Greed Index")
        return self  # Return self to allow for method chaining

    def get_engineered_data(self):
        """
        Returns the data with engineered features.

        Returns:
        - pd.DataFrame: The data with engineered features.
        """
        
        return self.data  # Return the modified DataFrame
    
    def fit(self, X, y=None):
        """
        Fit the transformer. Since no fitting is necessary for this transformer,
        this method just returns self.

        Parameters:
        - X (pd.DataFrame): The input data.
        - y (array-like, optional): The target variable. Not used.

        Returns:
        - self: The current TemporalFeatures instance.
        """
        return self

    def transform(self, X, y=None):
        """
        Apply transformations to the data.

        Parameters:
        - X (pd.DataFrame): The input data.
        - y (array-like, optional): The target variable. Not used.

        Returns:
        - pd.DataFrame: The data with engineered features.
        """
        self.data = X.copy()  # Update self.data with the new data
        # Apply all your feature engineering methods
        (self.add_moving_averages()
            .add_exponential_moving_averages()
            .add_rsi()
            .add_macd()
            .add_bollinger_bands()
            .add_vwap()
            .add_stochastic_oscillator())
        return self.get_engineered_data()

    def fit_transform(self, X, y=None):
        """
        Fit the transformer and transform the data.

        Parameters:
        - X (pd.DataFrame): The input data.
        - y (array-like, optional): The target variable. Not used.

        Returns:
        - pd.DataFrame: The data with engineered features.
        """
        return self.fit(X).transform(X)
    
# Define transformers
class SavitzkyGolayFilter:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return savgol_filter(X, 11, 3, axis=0)

class LagFeatures:
    def __init__(self, lag=1):
        self.lag = lag

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.hstack((X, np.roll(X, shift=self.lag, axis=0)))
    
class WindowingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size, step_size=1, drop_remainder=False):
        """
        Initialize the transformer.
        
        Parameters:
        - window_size (int): The size of each window.
        - step_size (int): The step size between windows.
        - drop_remainder (bool): Whether to drop the last window if it's smaller than window_size.
        """
        self.window_size = window_size
        self.step_size = step_size
        self.drop_remainder = drop_remainder

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything from the data,
        # so it just returns self.
        return self

    def transform(self, X, y=None):
        """
        Transform the input data into windows.
        
        Parameters:
        - X (array-like): The input data.
        
        Returns:
        - windows (array-like): The windowed data.
        """
        # Ensure X is a numpy array
        X = np.asarray(X)
        
        # Get the number of windows
        num_windows = (len(X) - self.window_size) // self.step_size + 1
        
        # Initialize an empty list to store the windows
        windows = []
        
        for i in range(0, num_windows * self.step_size, self.step_size):
            window = X[i:i + self.window_size]
            if len(window) == self.window_size or not self.drop_remainder:
                windows.append(window)
        
        # Convert the list of windows back to a numpy array
        return np.array(windows)