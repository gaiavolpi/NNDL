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

def debug(text, var, debug_mode=True):
    """
    text: the text that preceed the debug message
    var: list of variables whose values are to be displayed
    """
    if debug_mode:
        print("Debug: ", text, var)

def table3(folder_path='/mnt/shared_volume/table3/'):
    vp_to_name = {1: 'front', 2: 'rear', 3: 'side', 4: 'front-side', 5: 'rear-side'}

    folder = Path(folder_path)

    topk_pattern = re.compile(r"topk_accuracies_vp(\d+)")
    make_pattern = re.compile(r"accuracy_vp(\d+)")

    topk_data = {}
    make_data = {}

    for file in folder.iterdir():
        if not file.is_file():
            continue

        # Top-1 and Top-5
        topk_match = topk_pattern.search(file.stem)
        if topk_match:
            vp = int(topk_match.group(1))
            try:
                values = np.loadtxt(file)
                if len(values) == 2:
                    top1, top5 = values
                    topk_data[vp] = {'top1': top1, 'top5': top5}
                else:
                    print(f"⚠️ File {file.name} does not contain exactly 2 values.")
            except Exception as e:
                print(f"❌ Error reading {file.name}: {e}")

        # Make accuracy
        make_match = make_pattern.search(file.stem)
        if make_match:
            vp = int(make_match.group(1))
            try:
                value = float(np.loadtxt(file))
                make_data[vp] = value
            except Exception as e:
                print(f"❌ Error reading {file.name}: {e}")

    # Merge data
    data = []
    for vp in sorted(vp_to_name.keys()):
        row = {
            'viewpoint': vp_to_name[vp],
            'top1': topk_data.get(vp, {}).get('top1', None),
            'top5': topk_data.get(vp, {}).get('top5', None),
            'make': make_data.get(vp, None)
        }
        data.append(row)

    df = pd.DataFrame(data).set_index(keys='viewpoint')
    return df