import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from functools import lru_cache
from sklearn.model_selection import train_test_split

def indexing_labels(path_txt_file, label_type='model_id'): 
    '''
    This function reads a text file containing paths to images and extracts labels based on the specified label_type.
    It returns a dictionary mapping original labels to class indices for training.'''
    
    labels=[]

    with open(path_txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            model_id = parts[1] #'origina label' is the model id
            make_id = parts[0] 
            if label_type == 'model_id':
              labels.append(int(model_id))
            elif label_type == 'make_id':
              labels.append(int(make_id))
            else:
              raise Exception('error with label_type argument')

    unique_labels = sorted(set(labels)) # Map original labels to class indices
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    return label_to_index

def check_unbalance_dataset(loader, n_indices=3000, title=''):
    '''
    This function checks the unbalance in the dataset by plotting a histogram of the class distribution.
    '''
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

def split_val_test(volume_dir, viewpoint=None, label_type='model_id', part=False):
    train = pd.read_csv((volume_dir + f'data/train_test_split/part/train_part_{part}.txt') if part else (volume_dir + 'data/train_test_split/classification/train.txt'), header=None)
    test = pd.read_csv((volume_dir + f'data/train_test_split/part/test_part_{part}.txt') if part else (volume_dir + 'data/train_test_split/classification/test.txt'), header=None)

    if viewpoint is not None:   
        train = train[train[2] == viewpoint]
        test = test[test[2] == viewpoint]

        # remove classes with just one samples in test set
        classes, counts = np.unique(test[1 if label_type == 'model_id' else 0], return_counts=True)
        classes_to_rem = classes[counts < 2]  
        for c in classes_to_rem:
            test = test[test[1 if label_type == 'model_id' else 0] != c]
            train = train[train[1 if label_type == 'model_id' else 0] != c]

    if part is not None:
        # remove classes with just one samples in test set
        classes, counts = np.unique(test[1 if label_type == 'model_id' else 0], return_counts=True)
        classes_to_rem = classes[counts < 2]  
        for c in classes_to_rem:
            test = test[test[1 if label_type == 'model_id' else 0] != c]
            train = train[train[1 if label_type == 'model_id' else 0] != c]

    # stratified split test into test and validation
    test, val = train_test_split(test, test_size=0.5, stratify=test[1 if label_type == 'model_id' else 0], random_state=42)

    return train, test, val

def dataset_factory(volume_dir, label_to_index, transforms_train, transforms, paths, labels, part=False):
    '''
    This function creates a dataset factory that generates train, test, and validation datasets based on the specified viewpoint.
    It returns a function that can be called with a specific viewpoint to generate the datasets.
    '''

    def generate():
        fullpath = os.path.join(volume_dir, "data", "image" if part is False else "part")
        train_dataset = ImageDataset(fullpath, label_to_index, transforms_train, paths["train"], labels["train"])
        test_dataset = ImageDataset(fullpath, label_to_index, transforms, paths["test"], labels["test"])
        valid_dataset = ImageDataset(fullpath, label_to_index, transforms, paths["valid"], labels["valid"])
        return train_dataset, test_dataset, valid_dataset
    return generate

@lru_cache(maxsize=None)
def load_image(path):
    """
    Load an image from the given path and convert it to RGB format.
    This function is cached to avoid reloading the same image multiple times.
    """
    image = Image.open(path).convert("RGB")
    return image

class ImageDataset(Dataset):

    def __init__(self, base_path, dict_labels, transform, paths, labels):
        '''
        label_type='model_id', 'make_id'
        '''
        self.transform = transform
        self.dict_labels=dict_labels

        self.base_path = base_path
        self.paths = paths
        self.labels = labels
            
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = load_image(os.path.join(self.base_path, img_path))  # Load the image using the cached function
        label = self.dict_labels[int(self.labels[idx])] #labelling the image, converting the original label to index through the dictionary
        if self.transform:
            image = self.transform(image)
        return image, label