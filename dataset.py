import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


def indexing_labels(path_txt_file, label_type='model_id'): 
    '''
    This function reads a text file containing paths to images and extracts labels based on the specified label_type.
    It returns a dictionary mapping original labels to class indices for training.'''
    labels=[]

    with open(path_txt_file, 'r') as f:
        for line in f:
            path = line.strip()
            parts = os.path.normpath(path).split(os.sep)
            model_id = parts[-3] #'origina label' is the model id
            make_id = parts[-4] 
            if label_type == 'model_id':
              labels.append(int(model_id))
            elif label_type == 'make_id':
              labels.append(int(make_id))
            else:
              raise Exception('error with label_type argument')

    unique_labels = sorted(set(labels)) # Map original labels to class indices
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    return label_to_index

def split_valid_test(volume_dir):
    '''
    This function splits the test set into a new validation set and a new test set,
    ensuring an even class-wise distribution.
    '''
    label_to_paths = defaultdict(list)

    # Read all test paths and group by class
    with open(volume_dir+'data/train_test_split/classification/test.txt', 'r') as f:
        for line in f:
            path = line.strip()
            parts = os.path.normpath(path).split(os.sep)
            label = parts[-3]
            label_to_paths[label].append(path)

    valid_paths, test_paths = [], []

    # For each class, split 50/50
    for paths in label_to_paths.values():
        mid = len(paths) // 2
        valid_paths.extend(paths[:mid])
        test_paths.extend(paths[mid:])

    # Save valid.txt
    with open(volume_dir+'data/train_test_split/classification/valid.txt', 'w') as f:
        for path in valid_paths:
            f.write(path + '\n')

    # Save test_updated.txt
    with open(volume_dir+'data/train_test_split/classification/test_updated.txt', 'w') as f:
        for path in test_paths:
            f.write(path + '\n')

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

def dataset_factory(volume_dir, label_to_index, transforms_train, transforms, label_type='model_id'):
    '''
    This function creates a dataset factory that generates train, test, and validation datasets based on the specified viewpoint.
    It returns a function that can be called with a specific viewpoint to generate the datasets.
    
    label_type='model_id', 'make_id'
    '''
    def generate(viewpoint, label_type=label_type):
        train_dataset = ImageDataset(volume_dir + "data", volume_dir + "data/train_test_split/classification/train.txt", label_to_index, transforms_train, viewpoint, label_type=label_type)
        test_dataset = ImageDataset(volume_dir + "data", volume_dir + "data/train_test_split/classification/test_updated.txt", label_to_index, transforms, viewpoint, label_type=label_type)
        valid_dataset = ImageDataset(volume_dir + "data", volume_dir + "data/train_test_split/classification/valid.txt", label_to_index, transforms, viewpoint, label_type=label_type)
        return train_dataset, test_dataset, valid_dataset
    return generate
# Ti chiederai perchè serve questa? Così non devi duplicare il codice ogni volta che devi generare dataset per un diverso viewpoint. 
# In più, se cambia qualcosa devi solo aggiornare la factory!

class ImageDataset(Dataset):

    def __init__(self, dataset_folder, path_txt_file, dict_labels, transform=None, viewpoint=None, label_type='model_id'):
        '''
        label_type='model_id', 'make_id'
        '''
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.dict_labels=dict_labels


        # load the paths to the images you need
        with open(path_txt_file, 'r') as f:
            for line in f:
                relative_path = line.strip() # get the paths of the images used in the paper

                if viewpoint: # if we want to train/test on a single viewpoint
                  label_path = os.path.join(dataset_folder, 'label', relative_path.replace('.jpg', '.txt'))
                  with open(label_path, 'r') as f:
                    lines = f.readlines()
                    vp = int(lines[0].strip())

                  if vp == viewpoint: # use only the images with the desired viewpoint
                    image_path = os.path.join(dataset_folder, 'image', relative_path)
                    self.image_paths.append(image_path)

                else: #load all viewpoints
                  image_path = os.path.join(dataset_folder, 'image', relative_path)
                  self.image_paths.append(image_path)

        # Extract label model_id from path
        for path in self.image_paths:
            parts = os.path.normpath(path).split(os.sep)
            model_id = parts[-3]
            make_id = parts[-4]
            if label_type == 'model_id':
              self.labels.append(model_id)
            elif  label_type == 'make_id':
              self.labels.append(make_id)
            else:
              raise Exception('error with label_type argument')
            '''
            # run this if you instead want the set of three labels (make_id, model_id, year)
            year = parts[-2]
            self.labels.append((make_id, model_id, year))
            '''
            
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.dict_labels[int(self.labels[idx])] #labelling the image, converting the original label to index through the dictionary
        if self.transform:
            image = self.transform(image)
        return image, label