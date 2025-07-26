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


def plot_losses(train_loss_log, val_loss_log, save=False):

    fig, ax = plt.subplots()
    ax.semilogy(train_loss_log, label='Train loss')
    ax.semilogy(val_loss_log, label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)
    ax.legend()
    if save:
        return fig, ax
    else:
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

def table3(folder_path='/mnt/shared_volume/table3/'): #da controollare
    vp_to_name = {0: 'all-view', 1: 'front', 2: 'rear', 3: 'side', 4: 'front-side', 5: 'rear-side'}
    folder = Path(folder_path)

    # New filename patterns
    model_pattern = re.compile(r"model_id_vp(\d+)")
    make_pattern = re.compile(r"make_id_vp(\d+)")

    topk_data = {}
    make_data = {}

    for file in folder.iterdir():
        if not file.is_file():
            continue

        model_match = model_pattern.search(file.stem)
        make_match = make_pattern.search(file.stem)

        try:
            values = np.loadtxt(file)
            if isinstance(values, np.ndarray) and values.ndim == 0:
                values = np.array([values])  # single value case

            if model_match:
                vp = int(model_match.group(1))
                if len(values) >= 2:
                    topk_data[vp] = {'top1': values[0], 'top5': values[1]}
                else:
                    print(f"⚠️ model_id file {file.name} should contain at least 2 values (top-1 and top-5).")
            elif make_match:
                vp = int(make_match.group(1))
                if len(values) >= 1:
                    make_data[vp] = values[0]  # only top-1 for make
                else:
                    print(f"⚠️ make_id file {file.name} should contain at least 1 value (top-1).")
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

    df = pd.DataFrame(data).set_index('viewpoint')
    return df.round(3)