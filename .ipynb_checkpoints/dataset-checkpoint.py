import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch


def indexing_labels(path_txt_file):
    """
    Aim:: map the original labels to new class indices, from 1 to max(nr.models)
    """
    labels=[]
    
    with open(path_txt_file, 'r') as f: 
        for line in f:
            path = line.strip() 
            parts = os.path.normpath(path).split(os.sep)
            model_id = parts[-3] #'origina label' is the model id
            labels.append(int(model_id))
    
    unique_labels = sorted(set(labels)) # Map original labels to class indices
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    return label_to_index

def split_valid_test():
    '''
    Aim: splitting the test set used in the paper in a 50-50 validation and test sets
    '''
    general_paths = []
    with open('./data/train_test_split/classification/test.txt', 'r') as f: #opens the test txt and store alle the paths to the images
        for line in f:
            path = line.strip()
            general_paths.append(path)
    valid_paths = general_paths[:int(len(general_paths)*0.5)] #store half paths for the new test set and the other half for the new valid set
    test_paths = general_paths[int(len(general_paths)*0.5):]

    # Write to valid.txt
    with open('./data/train_test_split/classification/valid.txt', 'w') as f:
        for path in valid_paths:
            f.write(path + '\n')

    # Write to test_updated.txt
    with open('./data/train_test_split/classification/test_updated.txt', 'w') as f:
        for path in test_paths:
            f.write(path + '\n')

def check_unbalance_dataset(loader, n_indices=3000, title=''):
    # Check unbalance in the dataset
    labels_array = []
    
    for batch in loader:
        _, labels = batch  # batch[1] is the labels
        labels_array.append(labels)
    
    # Concatenate into a single tensor or list
    labels_array = torch.cat(labels_array).numpy().astype(int)  # or .tolist() if you want a Python list
    
    labels_array_sorted = np.sort(labels_array)
    
    plt.hist(labels_array_sorted[:n_indices], bins=np.arange(min(labels_array_sorted[:n_indices]),max(labels_array_sorted[:n_indices])), edgecolor='black')
    plt.xticks(np.arange(min(labels_array_sorted[:n_indices]), max(labels_array_sorted[:n_indices]),step=10))
    plt.title(title)
    plt.xlabel('classes')
    plt.ylabel('counts')
    plt.show()