import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import glob
#from Curve_fitting_with_scipy import Gaussianfitting as Gf
#from Curve_fitting_with_scipy import Linefitting as Lf
from scipy.signal import fftconvolve
from collections import defaultdict
import PIL
import joblib
from tqdm import tqdm
import pickle



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
#import torchvision.datasets as datasets
#import torchvision.transforms as transformers
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


#from Torch_modules.classes import FullyConnectedNeuralNetwork, TorchPipeline, get_numerical_categorical_boolean_columns
#from Torch_modules.classes import TotalImputer, CategoricalToNumerical, NumericalNormalizedScaler, NumericalStandardScaler
#from Torch_modules.classes import ConvolutionalNeuralNetwork


import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi'] = 120  # highres display










def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict









filename_train = glob.glob("/scratch/sagar.dam/python_codes/cifar/train_data/*")
filename_test = "/scratch/sagar.dam/python_codes/cifar/test_data/test_batch"




data_dict = unpickle(filename_train[0])
x_train = np.array(data_dict[b"data"])
y_train = np.array(data_dict[b"labels"])

for filename in filename_train[1:]:
    data_dict = unpickle(filename)
    x_train = np.concatenate((x_train,np.array(data_dict[b"data"])),axis=0)
    y_train = np.concatenate((y_train,np.array(data_dict[b"labels"])),axis=0)

print(f"x_train.shape: {x_train.shape};  y_train.shape: {y_train.shape}")


data_dict = unpickle(filename_test)
x_test = np.array(data_dict[b"data"])
y_test = np.array(data_dict[b"labels"])
print(f"x_test.shape: {x_test.shape};  y_test.shape: {y_test.shape}")

del data_dict











x_train = x_train.reshape(-1,3,32,32)
x_test = x_test.reshape(-1,3,32,32)

x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)

x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)

# Ensure that y_train is a LongTensor otherwise it will throw error while calculating accuracy
y_train = y_train.long()
# If you're using x_test and y_test in a similar manner, ensure you convert them too:
y_test = y_test.long()



x_train = x_train.to(device)
y_train = y_train.to(device)

x_test = x_test.to(device)
y_test = y_test.to(device)

image_width = x_train.shape[2]
image_height = x_train.shape[3]
image_channel = x_train.shape[1]








"""

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
        # Dynamically initializes fully connected layers based on the flattened output from conv layers.
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







class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, conv_layers, fc_layers_dims, in_channels, output_dimension, 
                 pooling_types=None, pool_kernels=None, activations=None, 
                 batch_norm=None, dropouts=None):
        super(ConvolutionalNeuralNetwork, self).__init__()
        
        self.device = device  # Set the device attribute
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        current_channels = in_channels  # Number of input channels (e.g., 1 for grayscale, 3 for RGB)

        for (out_channels, kernel_size, stride, padding) in conv_layers:
            self.conv_layers.append(nn.Conv2d(current_channels, out_channels, kernel_size, stride, padding))
            current_channels = out_channels  # Update the number of input channels for the next layer

        # Pooling types and kernels
        self.pooling_types = pooling_types if pooling_types else ["max"] * len(conv_layers)  # Default "max" pooling if not specified
        self.pool_kernels = pool_kernels if pool_kernels else [2] * len(conv_layers)  # Default pooling kernel 2 if not specified
        
        # Check if pooling types and kernels length matches conv layers
        if len(self.pool_kernels) != len(conv_layers) or len(self.pooling_types) != len(conv_layers):
            raise ValueError("The number of pooling kernels and pooling types must match the number of conv layers")

        # Batch normalization
        self.batch_norm = batch_norm if batch_norm else [False] * len(conv_layers)  # Default no batch norm
        self.bn_layers = nn.ModuleList([nn.BatchNorm2d(out_channels) if bn else None for bn, (out_channels, _, _, _) in zip(self.batch_norm, conv_layers)])

        # Activation functions
        self.activations = activations if activations else ["relu"] * len(conv_layers)  # Default ReLU activations
        
        # Dropout
        self.dropouts = dropouts if dropouts else [0.0] * len(fc_layers_dims)  # Default no dropout in fully connected layers
        self.dropout_layers = nn.ModuleList([nn.Dropout(p=drop) if drop > 0 else None for drop in self.dropouts])
        
        # Fully connected layers placeholder (input dimension will be calculated later)
        self.fc_layers = nn.ModuleList()
        self.fc_layer_dims = fc_layers_dims  # Store FC layer dimensions
        self.output_dimension = output_dimension  # Output layer dimension (e.g., 10 for CIFAR-10)
        self.fc_initialized = False  # We will initialize FC layers dynamically in the forward pass
    """
    def initialize_fc_layers(self, flattened_dim):
        # Dynamically initializes fully connected layers based on the flattened output from conv layers.
        input_dim = flattened_dim  # Start with the flattened conv output size
        for dim in self.fc_layer_dims:
            self.fc_layers.append(nn.Linear(input_dim, dim))
            input_dim = dim  # Update input size for next layer
        self.output_layer = nn.Linear(input_dim, self.output_dimension)  # Final output layer
        self.fc_initialized = True
    """

    def initialize_fc_layers(self, flattened_dim):
        # Dynamically initializes fully connected layers based on the flattened output from conv layers.
        input_dim = flattened_dim  # Start with the flattened conv output size
        for dim in self.fc_layer_dims:
            fc = nn.Linear(input_dim, dim)
            fc = fc.to(self.device)  # Move fully connected layers to the same device
            self.fc_layers.append(fc)
            input_dim = dim  # Update input size for next layer
        self.output_layer = nn.Linear(input_dim, self.output_dimension)
        self.output_layer = self.output_layer.to(self.device)  # Move output layer to the device
        self.fc_initialized = True

    def forward(self, x):
        # Moe the data to right device:
        x = x.to(device)

        # Pass through convolutional layers
        for idx, conv in enumerate(self.conv_layers):
            x = conv(x)

            # Apply batch normalization if specified
            if self.bn_layers[idx] is not None:
                x = self.bn_layers[idx](x)

            # Apply activation function (customizable per layer)
            if self.activations[idx] == "relu":
                x = F.relu(x)
            elif self.activations[idx] == "sigmoid":
                x = torch.sigmoid(x)
            elif self.activations[idx] == "tanh":
                x = torch.tanh(x)
            elif self.activations[idx] == "leaky_relu":
                x = F.leaky_relu(x, negative_slope=0.01)
            # Add more activations if needed

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
        for idx, fc in enumerate(self.fc_layers):
            x = F.relu(fc(x))  # Apply ReLU after each fully connected layer
            
            # Apply dropout if specified for this layer
            if self.dropout_layers[idx] is not None:
                x = self.dropout_layers[idx](x)
        
        # Output layer (no activation, as we use CrossEntropyLoss which applies softmax internally)
        x = self.output_layer(x)
        return x







class TorchPipeline:
    def __init__(self, model, optimizer, criterion=nn.CrossEntropyLoss(),  
                 batch_size=64, epochs=5, learning_rate=0.001):
        self.model = model
        self.model = self.model.to(device)
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
                    # data = x_train[batch_start:batch_end].to(device).float()
                    # target = y_train[batch_start:batch_end].to(device)
                    # Reshape the data to (batch_size, 3, 32, 32)
                    data = x_train[batch_start:batch_end].reshape(-1, 3, 32, 32).to(device).float()
                    target = y_train[batch_start:batch_end].to(device)


                    # Debug: Ensure data and model are on the same device
                    # print(f"Data device: {data.device}")
                    # print(f"Model device: {next(self.model.parameters()).device}")
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
                print(f"epoch: {epoch}/{self.epochs};  loss:  {loss.item()};   progress: {epoch*100/self.epochs:.2f}")
                    
                    
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











# Define the conv layers as specified
conv_layers = [
    (64, 2, 1, 1),  # Conv layer 1: 16 filters, 4x4 kernel, stride 1, padding 1
    (128, 2, 1, 1),  # Conv layer 2: 32 filters, 3x3 kernel, stride 1, padding 0
    (128, 2, 1, 1),  # Conv layer 3: 64 filters, 4x4 kernel, stride 1, padding 0
    (256, 2, 1, 1), 
    (256, 2, 1, 0),
    (512, 2, 1, 0)
]


pooling_types = ["no","avg","avg","avg","avg","avg"]  # Different pooling types for each layer
pool_kernels = [1,1,2,1,2,2]  # Different pooling kernels for each layer
activations = ["relu", "relu", "relu", "relu", "relu", "relu"]  # Activations for each conv layer
batch_norm = [True, True, True, True, True, False]  # Batch normalization for each conv layer


# Define fully connected layers
fc_layers_dims = [4096,2048,1024,512,256,128] # excludind first layer (output_of_conv -> input_on_fc)
dropouts = [0.0, 0.0, 0.15, 0.1, 0.2, 0.0]  # Dropout for fully connected layers
# output_dimension=len(set(np.array(y_train)))
output_dimension = 10 #len(set(np.array(y_train.cpu())))

# Instantiate the model
#model = ConvolutionalNeuralNetwork(conv_layers=conv_layers,fc_layers_dims=fc_layers_dims,in_channels=image_channel,output_dimension=output_dimension, pooling="avg", pool_kernal=2)
# Instantiate the model
"""
model = ConvolutionalNeuralNetwork(conv_layers=conv_layers,
                                   fc_layers_dims=fc_layers_dims,
                                   in_channels=image_channel,
                                   output_dimension=output_dimension, 
                                   pooling_types=pooling_types, 
                                   pool_kernels=pool_kernels)
"""

model=ConvolutionalNeuralNetwork(conv_layers,fc_layers_dims,in_channels=3,output_dimension=output_dimension, 
                                   pooling_types=pooling_types, pool_kernels=pool_kernels, 
                                   activations=activations, batch_norm=batch_norm, dropouts=dropouts).to(device)

learning_rate = 0.0005
batch_size = 400
epochs = 500

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()









# Create the pipeline object
torchpipeline = TorchPipeline(model=model, 
                         criterion=criterion, 
                         optimizer=optimizer, 
                         batch_size=batch_size, 
                         epochs=epochs)







# Fitting the model
torchpipeline.fit(x_train, y_train, device=device, print_after=25)






print(model)






# Making predictions
y_predict, accuracy = torchpipeline.predict(x_test, y_test, device)







