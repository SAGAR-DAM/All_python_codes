# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:53:23 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import glob
# from Curve_fitting_with_scipy import Gaussianfitting as Gf
# from Curve_fitting_with_scipy import Linefitting as Lf
from scipy.signal import fftconvolve
from collections import defaultdict
import PIL
import joblib
from tqdm import tqdm



from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import root_mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression



from xgboost import XGBRegressor, XGBClassifier, XGBRFClassifier, XGBRFRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transformers
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi'] = 120  # highres display




#########################################################
# Class defined for fully connected linear neural network
#########################################################
class FullyConnectedNeuralNetwork(nn.Module):
    def __init__(self, layer_dims, input_dimension, output_dimension):
        super(FullyConnectedNeuralNetwork, self).__init__()
        # Initialize a list to hold the layers
        self.layers = nn.ModuleList()
        
        # Input layer dimension
        self.in_dim = input_dimension  # Input size, e.g., for MNIST images : 784
        self.out_dim = output_dimension  # Output size, e.g., for MNIST images : 10
        
        # Create intermediate layers based on the provided dimensions
        for dim in layer_dims:
            self.layers.append(nn.Linear(self.in_dim, dim))
            self.in_dim = dim  # Update in_dim for the next layer
        # Output layer for 10 classes (digits 0-9)
        self.out = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        # Check if input has more than 2 dimensions (i.e., image data or other multi-dimensional data)
        if x.dim() > 2:
            # Flatten the input, but retain batch size: (batch_size, -1)
            x = x.view(x.size(0), -1)
        
        # Pass the input through each layer
        for layer in self.layers:
            x = torch.relu(layer(x))  # Apply ReLU activation
        
        # Get the output from the final layer
        x = self.out(x)
        return x
    
    
    
    



#########################################################
# Class defined for Convolutional linear neural network
#########################################################
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, conv_layers, fc_layers_dims, in_channels, output_dimension, pooling_types=None, pool_kernels=None):
        super(ConvolutionalNeuralNetwork, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        current_channels = in_channels  # Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        
        for (out_channels, kernel_size, stride, padding) in conv_layers:
            self.conv_layers.append(nn.Conv2d(current_channels, out_channels, kernel_size, stride, padding))
            current_channels = out_channels  # Update the number of input channels for the next layer

        # Pooling types and kernels
        self.pooling_types = pooling_types if pooling_types else ["max"] * len(conv_layers)  # Default "max" pooling if not specified
        self.pool_kernels = pool_kernels if pool_kernels else [2] * len(conv_layers)  # Default pooling kernel 2 if not specified
        
        if len(self.pool_kernels) != len(conv_layers) or len(self.pooling_types) != len(conv_layers):
            raise ValueError("The number of pooling kernels and pooling types must match the number of conv layers")

        # Fully connected layers placeholder (input dimension will be calculated later)
        self.fc_layers = nn.ModuleList()
        self.fc_layer_dims = fc_layers_dims  # Store FC layer dimensions
        self.output_dimension = output_dimension  # Output layer dimension (e.g., 10 for CIFAR-10)
        self.fc_initialized = False  # We will initialize FC layers dynamically in the forward pass
    
    def initialize_fc_layers(self, flattened_dim):
        """Dynamically initializes fully connected layers based on the flattened output from conv layers."""
        input_dim = flattened_dim  # Start with the flattened conv output size
        for dim in self.fc_layer_dims:
            self.fc_layers.append(nn.Linear(input_dim, dim))
            input_dim = dim  # Update input size for next layer
        self.output_layer = nn.Linear(input_dim, self.output_dimension)  # Final output layer
        self.fc_initialized = True
    
    def forward(self, x):
        # Pass through convolutional layers
        for idx, conv in enumerate(self.conv_layers):
            x = F.relu(conv(x))
            
            # Apply specific pooling type and kernel size for each layer
            if self.pooling_types[idx] == "max":
                x = F.max_pool2d(x, kernel_size=self.pool_kernels[idx])
            elif self.pooling_types[idx] == "avg":
                x = F.avg_pool2d(x, kernel_size=self.pool_kernels[idx])
            elif self.pooling_types[idx] == "no":
                pass  # No pooling

        # Flatten the output of the conv layers
        x = x.view(x.size(0), -1)

        # Dynamically initialize fully connected layers based on flattened output size if not done already
        if not self.fc_initialized:
            self.initialize_fc_layers(flattened_dim=x.size(1))
        
        # Pass through fully connected layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        
        # Output layer (no activation, as we use CrossEntropyLoss which applies softmax internally)
        x = self.output_layer(x)
        return x

"""
# Use:
    
# Define the conv layers as specified
conv_layers = [
    (16, 4, 1, 1),  # Conv layer 1: 16 filters, 4x4 kernel, stride 1, padding 1
    (32, 3, 1, 0),  # Conv layer 2: 32 filters, 3x3 kernel, stride 1, padding 0
    (64, 4, 1, 0),  # Conv layer 3: 64 filters, 4x4 kernel, stride 1, padding 0
]
pool_kernels = [2, 2, 2]  # Different pooling kernels for each layer
pooling_types = ["no","max","avg"]  # Different pooling types for each layer


# Define fully connected layers
fc_layers_dims = [512, 256, 128] # excludind first layer (output_of_conv -> input_on_fc)
output_dimension=len(set(y_train))

# Instantiate the model
model = ConvolutionalNeuralNetwork(conv_layers=conv_layers,
                                   fc_layers_dims=fc_layers_dims,
                                   in_channels=image_channel,
                                   output_dimension=output_dimension, 
                                   pooling_types=pooling_types, 
                                   pool_kernels=pool_kernels)

"""







########################################################################################
# computer number of input layers for first fc
########################################################################################
def compute_conv_to_fc_input_dim(conv_layers, pooling_types, pool_kernels, input_size):
    """
    Computes the input dimension for the first fully connected layer.
    
    Parameters:
        conv_layers (list of tuples): Each tuple contains (out_channels, kernel_size, stride, padding) for each conv layer.
        pooling_types (list of str): List of pooling types for each layer ('max', 'avg', 'no').
        pool_kernels (list of int): List of pooling kernel sizes for each layer.
        input_size (tuple): Input size as (batch_size, channels, height, width).

    Returns:
        int: The input dimension for the first fully connected layer.
    """
    
    # Initialize dimensions
    batch_size, channels, height, width = input_size

    # Process each convolutional layer
    for idx, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
        # Apply convolution
        height = (height + 2 * padding - kernel_size) // stride + 1
        width = (width + 2 * padding - kernel_size) // stride + 1

        # Apply pooling if specified
        if pooling_types[idx] != "no":
            height = (height - pool_kernels[idx]) // pool_kernels[idx] + 1
            width = (width - pool_kernels[idx]) // pool_kernels[idx] + 1

        # Update channels
        channels = out_channels

    # The output of the final conv layer needs to be flattened for the FC layer
    return channels * height * width  # Total flattened dimension

"""
# Use: 
# Example input dimensions: (batch_size, channels, height, width)
input_size = (1, 3, 32, 32)  # Example for CIFAR with 3 channels and 32x32 size

# Define the conv layers as specified
conv_layers = [
    (64, 2, 1, 0),  # Conv layer 1: 16 filters, 4x4 kernel, stride 1, padding 1
    (128, 2, 1, 0),  # Conv layer 2: 32 filters, 3x3 kernel, stride 1, padding 0
    (128, 2, 1, 0),  # Conv layer 3: 64 filters, 4x4 kernel, stride 1, padding 0
    (256, 2, 1, 0), 
    (256, 2, 1, 0),
    (512, 2, 1, 0)
]


pooling_types = ["no","max","avg","max","max","max"]  # Different pooling types for each layer
pool_kernels = [1,1,2,1,2,2]  # Different pooling kernels for each layer


# Compute the input dimension for the first fully connected layer
fc_input_dim = compute_conv_to_fc_input_dim(conv_layers, pooling_types, pool_kernels, input_size)
print(f"The input dimension for the first fully connected layer is: {fc_input_dim}")
"""







########################################################################################
# Training and testing functions remain similar to your existing fully connected NN code
########################################################################################
class TorchPipeline:
    def __init__(self, model, optimizer, criterion=nn.CrossEntropyLoss(),  
                 batch_size=64, epochs=5, learning_rate=0.001):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
    
    def fit(self, x_train, y_train, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), print_after=10):
        self.model.to(device)
        self.model.train()
        num_samples = x_train.size(0)

        # ANSI color codes
        # Foreground (text) colors
        BLACK = "\033[30m"
        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        CYAN = "\033[96m"
        WHITE = "\033[97m"

        text_color = [RED,GREEN,YELLOW,BLUE,MAGENTA,CYAN,WHITE]

        # Background colors
        BG_BLACK = "\033[40m"
        BG_RED = "\033[41m"
        BG_GREEN = "\033[42m"
        BG_YELLOW = "\033[43m"
        BG_BLUE = "\033[44m"
        BG_MAGENTA = "\033[45m"
        BG_CYAN = "\033[46m"
        BG_WHITE = "\033[47m"

        # Reset color
        RESET = "\033[0m"

        for epoch in range(1, self.epochs + 1):
            # Initialize tqdm progress bar for the epoch
            C = text_color[epoch%len(text_color)-1]
            with tqdm(total=num_samples, desc=f'Epoch {epoch}/{self.epochs}', ncols=100, unit=" samples",
                     bar_format=f"{{l_bar}}{GREEN}{{bar}}{RESET}|{{n_fmt}}/{{total_fmt}} [{RED}{{percentage:.0f}}%{RESET}]{{postfix}}") as pbar:
                for batch_start in range(0, num_samples, self.batch_size):
                    batch_end = min(batch_start + self.batch_size, num_samples)
                    data = x_train[batch_start:batch_end].to(device).float()
                    target = y_train[batch_start:batch_end].to(device)
    
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
    
                    # Forward pass
                    output = self.model(data)
    
                    # Calculate loss
                    loss = self.criterion(output, target)
    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
    
                    # Update progress bar
                    pbar.update(batch_end - batch_start)
    
                    # Optionally print intermediate progress (e.g., after a certain number of batches)
                    if batch_start % (print_after * self.batch_size) == 0:
                        pbar.set_postfix({"Loss": f'{loss.item():.6f}'})
                print("==============================================================================\n")
                    
                    
    def predict(self, x_test, y_test, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.model.eval()  # Set model to evaluation mode
        num_samples = x_test.size(0)
        all_predictions = []  # To store all predicted labels
        correct = 0  # To count correct predictions

        with torch.no_grad():  # Disable gradient computation for prediction
            for batch_start in range(0, num_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_samples)
                data = x_test[batch_start:batch_end].to(device).float()
                target = y_test[batch_start:batch_end].to(device)

                # Forward pass
                output = self.model(data)
                
                # Get the index of the max log-probability (prediction)
                pred = output.argmax(dim=1, keepdim=True)  # Shape: [batch_size, 1]
                all_predictions.append(pred.cpu().numpy())  # Store predictions
                
                # Count correct predictions
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Concatenate all predictions into a single array
        all_predictions = np.concatenate(all_predictions)

        # Calculate accuracy
        accuracy = 100. * correct / num_samples
        print(f'\nTest set: Accuracy: {correct}/{num_samples} ({accuracy:.0f}%)\n')

        return all_predictions, accuracy

"""
# Use:

# Create the pipeline object
torchpipeline = TorchPipeline(model=model, 
                         criterion=criterion, 
                         optimizer=optimizer, 
                         batch_size=batch_size, 
                         epochs=epochs)
"""



#############################################################
#  Make a flattened array for convolutional to linear network
#############################################################
def calculate_flattened_size(input_height, input_width, conv_layers):
    height = input_height
    width = input_width
    for (out_channels, kernel_size, stride, padding) in conv_layers:
        # Calculate new height and width after each convolutional layer
        height = (height - kernel_size + 2 * padding) // stride + 1
        width = (width - kernel_size + 2 * padding) // stride + 1
        
        # Assuming max pooling follows each convolutional layer
        height = height // 2
        width = width // 2
        
    # Multiply by the number of output channels of the last conv layer
    return height * width * conv_layers[-1][0]





#############################################################
# Separating Numerical, Categorical and Boolian data
#############################################################
def get_numerical_categorical_boolean_columns(data):
    # Separate categorical and numerical columns
    categorical_columns = []
    numerical_columns = []
    boolean_columns = []
    
    for name in np.array(data.columns):
        i = 0
        # Using .iloc to index by position and avoid the KeyError
        while pd.isnull(data[name].iloc[i]):  # Handle None/NaN entries properly
            i += 1
        
        # Now we determine the type of the first valid value
        first_value = data[name].iloc[i]
        
        if isinstance(first_value, str):
            categorical_columns.append(name)
        elif isinstance(first_value, (int, float, np.integer, np.floating)):
            numerical_columns.append(name)
        elif isinstance(first_value, bool):
            boolean_columns.append(name)
    
    return numerical_columns, categorical_columns, boolean_columns





#############################################################
# Custom imputer for different data types
#############################################################
class TotalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_columns, categorical_columns, boolean_columns):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.boolean_columns = boolean_columns
        self.imputer_numeric = SimpleImputer(strategy='median')
        self.imputer_categoric = SimpleImputer(strategy='most_frequent')
        self.imputer_boolean = SimpleImputer(strategy='most_frequent')

    def fit(self, X, y=None):
        if len(self.numerical_columns) != 0:
            self.imputer_numeric.fit(X[self.numerical_columns])
        if len(self.categorical_columns) != 0:
            self.imputer_categoric.fit(X[self.categorical_columns])
        if len(self.boolean_columns) != 0:
            self.imputer_boolean.fit(X[self.boolean_columns])
        return self

    def transform(self, X):
        if len(self.numerical_columns) != 0:
            X[self.numerical_columns] = self.imputer_numeric.transform(X[self.numerical_columns])
        if len(self.categorical_columns) != 0:
            X[self.categorical_columns] = self.imputer_categoric.transform(X[self.categorical_columns])
        if len(self.boolean_columns) != 0:
            X[self.boolean_columns] = self.imputer_boolean.transform(X[self.boolean_columns])
        return X


#############################################################
# custom labelmaker
#############################################################
class CategoricalToNumerical(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns
        self.label_encoders = {col: LabelEncoder() for col in categorical_columns}
        self.label_mappings = {}

    def fit(self, X, y=None):
        # Fit the LabelEncoder for each categorical column
        for col in self.categorical_columns:
            self.label_encoders[col].fit(X[col])
            # Store the label mapping for each column
            self.label_mappings[col] = dict(zip(self.label_encoders[col].classes_, 
                                                self.label_encoders[col].transform(self.label_encoders[col].classes_)))
            print(f"Label mapping for '{col}': \n{self.label_mappings[col]}")
            print(f"\n#######################################################\n")
        return self

    def transform(self, X):
        X_copy = X.copy()  # Avoid modifying the original DataFrame
        # Transform each categorical column using the fitted label encoder
        for col in self.categorical_columns:
            X_copy[col] = self.label_encoders[col].transform(X[col])
        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_label_mappings(self):
        """Optional method to retrieve the label mappings outside the fit/transform methods"""
        return self.label_mappings


#############################################################
# custom scaling method
#############################################################
class NumericalStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_columns):
        self.numerical_columns = numerical_columns
        self.scalers = {col: StandardScaler() for col in numerical_columns}

    def fit(self, X, y=None):
        # Fit the StandardScaler for each numerical column
        for col in self.numerical_columns:
            self.scalers[col].fit(X[[col]])  # X[[col]] is used to maintain column shape
        return self

    def transform(self, X):
        X_copy = X.copy()  # Avoid modifying the original DataFrame
        # Transform each numerical column using the fitted scaler
        for col in self.numerical_columns:
            X_copy[col] = self.scalers[col].transform(X[[col]])
        return X_copy



#############################################################
# custom scling method
#############################################################
class NumericalNormalizedScaler(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_columns):
        self.numerical_columns = numerical_columns
        self.scalers = {col: MinMaxScaler() for col in numerical_columns}

    def fit(self, X, y=None):
        # Fit the MinMaxScaler for each numerical column
        for col in self.numerical_columns:
            self.scalers[col].fit(X[[col]])  # X[[col]] to maintain the column shape
        return self

    def transform(self, X):
        X_copy = X.copy()  # Avoid modifying the original DataFrame
        # Transform each numerical column using the fitted MinMaxScaler
        for col in self.numerical_columns:
            X_copy[col] = self.scalers[col].transform(X[[col]])
        return X_copy
    
    
    
    
    