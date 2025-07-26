import torch
import torch.nn as nn
import torch.nn.functional as F

class OverFeat(nn.Module):
    def __init__(self, num_classes=431):
        super(OverFeat, self).__init__()
        
        # Feature extractor (conv layers)
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),  # conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # pool1
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2), # conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # pool2

            nn.Conv2d(256, 512, kernel_size=3, padding=1),# conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),# conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),# conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # pool5
        )
        
        # Fully connected layers (flattened)
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 6 * 6, 3072),  # adjust shape depending on input size
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(3072, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        features = self.classifier[:-1](x)  # use intermediate features for both heads
        class_logits = self.classifier[-1](features)
        return class_logits
