import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d
from torchvision.transforms import Compose, ToTensor, RandomAffine, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, Resize
import os
import re
import pandas as pd
from pathlib import Path    

from dataset import indexing_labels, ImageDataset
from ResNet50_blocks import ResNet50
from training_functions import network_training, FocalLoss


def plot_losses(train_loss_log, val_loss_log):
    # Plot losses
    plt.figure(figsize=(10,6))
    plt.semilogy(train_loss_log, label='Train loss')
    plt.semilogy(val_loss_log , label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.show()

def exponential_moving_average(log_loss, beta=0.2):
    """
    computes the exponential moving average (EMA) of the losses along the training through formula:
                    mu^{(i)} = beta * mu^{(i-1)} + (1-beta) * log_losses[i]
    """
    moving_averages = []
    mu = 0 #initial value for the moving average
    for i in range(len(log_loss)):
        mu = beta*mu + (1-beta)*log_loss[i]
        moving_averages.append(mu)
    return moving_averages



def table3(folder_path='table3/'):
    vp_to_name = {1 : 'front', 2: 'rear', 3:'side', 4:'front-side', 5:'rear-side'}

    folder = Path(folder_path)
    pattern = re.compile(r"topk_accuracies_vp(\d+)")
    
    data = []

    for file in folder.iterdir():
        if file.is_file() and file.name.startswith("topk_accuracies_vp"):
            match = pattern.search(file.stem)
            if match:
                vp = int(match.group(1))
                try:
                    values = np.loadtxt(file)
                    if len(values) == 2:
                        top1, top5 = values
                        data.append({'viewpoint': vp_to_name[vp], 'top1': top1, 'top5': top5})
                    else:
                        print(f"⚠️ File {file.name} does not contain exactly 2 values.")
                except Exception as e:
                    print(f"❌ Error reading {file.name}: {e}")
    
    df = pd.DataFrame(data)
    df = df.sort_values(by='viewpoint').reset_index(drop=True)
    print(df)
    return df