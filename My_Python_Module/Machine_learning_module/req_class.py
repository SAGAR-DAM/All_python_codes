import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import glob
from Curve_fitting_with_scipy import Gaussianfitting as Gf
from Curve_fitting_with_scipy import Linefitting as Lf
from scipy.signal import fftconvolve
from collections import defaultdict
import joblib
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi'] = 120  # highres display




def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index


def moving_average(signal, window_size):
    # Define the window coefficients for the moving average
    window = np.ones(window_size) / float(window_size)
    
    # Apply the moving average filter using fftconvolve
    filtered_signal = fftconvolve(signal, window, mode='same')
    
    return filtered_signal


def hist_dataframe(df, bins=10):
    # Define a list of colors for each histogram
    colors = ['red', 'green', 'blue', 'magenta', 'cyan', 'purple', 'orange', 'black']
    # Create subplots with a dynamic number of rows, 3 columns per row
    fig, axes = plt.subplots(nrows=int(np.ceil(len(df.columns) / 3)), ncols=3, figsize=(18, 4.5*int(np.ceil(len(df.columns) / 3))))
    # Flatten the axes array for easy iteration (even if it's a 2D grid)
    axes = axes.flatten()
    
    # Plot each histogram individually
    for i, column in enumerate(df.columns):
        df[column].plot(kind='hist', ax=axes[i], color=colors[i%len(colors)], title=column, bins = bins)
        axes[i].grid(True, linewidth=0.5, color='k')  # Optional: add grid
    
    # Turn off any unused subplots (in case the number of columns is not a multiple of 3)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])  # Delete empty subplots
    
    plt.tight_layout()  # Adjust the layout
    plt.show()
    pass
    

def binned_mode(data, num_bins):     
    """ use this function to replace the missing value with the most probable value in a dataset...
        There are inbuilt functions for mean and median"""
    # Calculate the range of the data
    data_min, data_max = min(data), max(data)
    
    # Calculate the bin edges
    bins = np.linspace(data_min, data_max, num_bins + 1)
    
    # Group data into bins
    binned_data = defaultdict(list)
    for num in data:
        # Find the correct bin index for each number
        bin_index = np.digitize(num, bins) - 1  # subtract 1 to get 0-based index
        bin_index = min(bin_index, num_bins - 1)  # ensure last bin is included
        binned_data[bin_index].append(num)
    
    # Find the bin with the highest frequency
    most_frequent_bin = max(binned_data, key=lambda k: len(binned_data[k]))
    
    # Calculate the average of the values in the most frequent bin
    mode_value = np.mean(binned_data[most_frequent_bin])
    
    return mode_value


def plot_hollow_pillar_histogram(data, bins=30, edgecolor='black', linewidth=1.5):   #, xlabel='Value', ylabel='Frequency', title='Histogram with Hollow Pillar Bars'):
    """
    Plots a histogram with hollow pillar bars.

    Parameters:
    - data: Array of data to be plotted.
    - bins: Number of bins or bin edges (default is 30).
    - edgecolor: Color of the bar edges (default is 'black').
    - linewidth: Thickness of the bar edges (default is 1.5).
    - xlabel: Label for the x-axis (default is 'Value').
    - ylabel: Label for the y-axis (default is 'Frequency').
    - title: Title for the plot (default is 'Histogram with Hollow Pillar Bars').
    """
    # Create the histogram without plotting it (retrieve the counts and bin edges)
    counts, bin_edges = np.histogram(data, bins=bins)
    # Width of each bar
    bin_width = bin_edges[1] - bin_edges[0]
    # Create the plot
    for i in range(bins):
        plt.hist(bin_edges[i]*np.ones(counts[i]),bins=1, edgecolor='black', linewidth=0.5, rwidth=(max(data)-min(data))/bins)
        
    # Set limits for x and y axis
    # ax.set_xlim(bin_edges[0], bin_edges[-1])
    # ax.set_ylim(0, max(counts) * 1.1)
    # Add labels and title
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.title(title)
    pass



def get_numerical_categorical_boolean_columns(data):
    # Separate categorical and numerical columns
    categorical_columns = []
    numerical_columns = []
    boolean_columns = []
    
    for name in np.array(data.columns):
        i=0
        while(data[name][i] is None):
            i += 1

        if(type(data[name][i]) is str):
            categorical_columns.append(name)
        elif((type(data[name][i]) is float) or (type(data[name][i]) is int) or (type(data[name][i]) is bin) or (type(data[name][i]) is np.int64) 
            or (type(data[name][i]) is np.int32) or (type(data[name][i]) is np.int16) or (type(data[name][i]) is np.int8) or 
            (type(data[name][i]) is np.float16) or (type(data[name][i]) is np.float32) or (type(data[name][i]) is np.float64)):
            
            numerical_columns.append(name)
        elif((type(data[name][i]) is bool)):
             boolean_columns.append(name)
        else:
            pass
    return numerical_columns, categorical_columns, boolean_columns





# Custom imputer for different data types
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