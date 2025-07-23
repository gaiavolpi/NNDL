import numpy as np
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d
from torch.nn import SiLU as ReLU
from torch.optim import SGD, Adam
import torch
from torch.nn import MaxPool2d, AvgPool2d, Linear, Dropout
from torch.nn import AdaptiveAvgPool2d

class BasicBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1):
        
        # structure of a single block 
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) 
        self.bn1 = BatchNorm2d(out_channels)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_channels)
        
        self.shortcut = Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(Module):
    def __init__(self, num_classes=431):
        super(ResNet18, self).__init__()
        # initial stage
        self.in_channels = 64
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1) # 2 blocks with 64 filters 
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2) # 2 bblocks with 128 filters 
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2) # 2 blocks with 256 filters
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2) # 2 blocks with 512 filters 
        
        self.avgpool = AdaptiveAvgPool2d((1, 1)) # average pooling 
        self.fc = Linear(512, num_classes) # linear layer for classification 

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out