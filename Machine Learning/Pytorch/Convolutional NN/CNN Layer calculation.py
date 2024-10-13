# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:53:06 2024

@author: mrsag
"""

#import torch
#import torch.nn as nn

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

# Example input dimensions: (batch_size, channels, height, width)
input_size = (1, 3, 32, 32)  # Example for CIFAR with 3 channels and 32x32 size

# Define the conv layers as specified
conv_layers = [
    (1,1,1,0),
    (16, 2, 1, 0),  # Conv layer 1: 16 filters, 4x4 kernel, stride 1, padding 1
    (32, 2, 1, 0),  # Conv layer 2: 32 filters, 3x3 kernel, stride 1, padding 0
    (64, 2, 1, 0),  # Conv layer 3: 64 filters, 4x4 kernel, stride 1, padding 0
]
pool_kernels = [2, 2, 2, 2]  # Different pooling kernels for each layer
pooling_types = ["no", "no","max","avg"]  # Different pooling types for each layer


# Compute the input dimension for the first fully connected layer
fc_input_dim = compute_conv_to_fc_input_dim(conv_layers, pooling_types, pool_kernels, input_size)
print(f"The input dimension for the first fully connected layer is: {fc_input_dim}")
