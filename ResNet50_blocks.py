import numpy as np
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d
from torch.nn import SiLU as ReLU
from torch.optim import SGD, Adam
import torch
from torch.nn import MaxPool2d, AvgPool2d, Linear, Dropout

class MainPath(Module):
    def __init__(self, in_channels, filters, kernel_size, stride=1):
        super().__init__()
        F1, F2, F3 = filters  # unpack filter sizes for the three conv layers

        # Define a sequential block of layers for the main path
        self.main_path = Sequential(
            Conv2d(in_channels, F1, kernel_size=1, stride=stride), #p=0
            BatchNorm2d(F1),                                           
            ReLU(),                                                    

            Conv2d(F1, F2, kernel_size=kernel_size, padding=kernel_size//2),  
            BatchNorm2d(F2),
            ReLU(),

            Conv2d(F2, F3, kernel_size=1), #p=0, s=1                              
            BatchNorm2d(F3),
        )

        # Initialize weights
        #self.apply(self._init_weights)

    def forward(self, x):
        # Pass input through the main path
        y = self.main_path(x)
        return y

# Define the identity residual block by inheriting from MainPath
class IdentityBlock(MainPath):
    def __init__(self, in_channels, filters, kernel_size):
        super().__init__(in_channels, filters, kernel_size) # default stride = 1
        self.relu = ReLU() 
        
    def forward(self, x):
        # Forward through the main path and add residual connection
        y = self.relu(self.main_path(x) + x)
        return y


class ConvolutionalBlock(MainPath):

    def __init__(self, in_channels, filters, kernel_size):
        super().__init__(in_channels, filters, kernel_size, stride=2) 
        self.relu = ReLU()

        self.shortcut_path = Sequential(
            Conv2d(in_channels, filters[2], kernel_size=1, stride=2),  # 1x1 convolution for dimension match, stride=2 -> downsample
            BatchNorm2d(filters[2])                                     
        )

        # Apply custom weight initialization to all layers
        #self.apply(self._init_weights)


    def forward(self, x):
        # Forward through the main path and the shortcut, then sum and apply ReLU
        y = self.relu(self.main_path(x) + self.shortcut_path(x))
        return y
        
'''
def _init_weights(self, module):
    # Xavier initialization for Linear layers
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()

    # Xavier initialization for Conv2d layers
    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
'''

class ResNet50(Module): 

    def __init__(self):
        super().__init__()
        self.network = Sequential(
            #Stage 1
            #Conv2d(3, 64, kernel_size=7, stride=2),
            Conv2d(3, 64, kernel_size=3, stride=1),
            Conv2d(64, 64, kernel_size=3, stride=2), 
            BatchNorm2d(64),
            MaxPool2d(kernel_size=3, stride=2),
            #Stage 2
            ConvolutionalBlock(64, [64, 64, 256], kernel_size=3),
            Dropout(0.2),
            IdentityBlock(256, [64, 64, 256], kernel_size=3),
            IdentityBlock(256, [64, 64, 256], kernel_size=3),
            #Stage 3
            ConvolutionalBlock(256, [128, 128, 512], kernel_size=3),
            Dropout(0.2),
            IdentityBlock(512, [128, 128, 512], kernel_size=3),
            IdentityBlock(512, [128, 128, 512], kernel_size=3),
            IdentityBlock(512, [128, 128, 512], kernel_size=3),
            #Stage 4
            ConvolutionalBlock(512, [256, 256, 1024], kernel_size=3),
            Dropout(0.2),
            IdentityBlock(1024, [256, 256, 1024], kernel_size=3),
            IdentityBlock(1024, [256, 256, 1024], kernel_size=3),
            IdentityBlock(1024, [256, 256, 1024], kernel_size=3),
            IdentityBlock(1024, [256, 256, 1024], kernel_size=3),
            IdentityBlock(1024, [256, 256, 1024], kernel_size=3),
            # Stage 5
            #ConvolutionalBlock(1024, [512, 512, 2048], kernel_size=3),
            #Dropout(0.2),
            #IdentityBlock(2048, [512, 512, 2048], kernel_size=3),
            #IdentityBlock(2048, [512, 512, 2048], kernel_size=3),
            #Average pooling
            AvgPool2d(kernel_size=2, stride=2)
        )
        self.lazy = torch.nn.LazyConv2d(512, 3, padding='same')
        # Final fully connected layer for classification
        self.classification_layer = torch.nn.LazyLinear(431)
        
        #self.apply(self._init_weights)

    def forward(self, x):
        # Forward pass through the network, flatten features, and classify
        y = self.lazy(self.network(x)).reshape((x.shape[0], -1))
        y = self.classification_layer(y)
        return y