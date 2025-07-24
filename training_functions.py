import torch
from torch.optim import SGD, Adam
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d
from torchvision.transforms import Compose, ToTensor, RandomAffine, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, Resize
from sklearn.utils.class_weight import compute_class_weight

from ResNet50_blocks import ResNet50
from utils import ImageDataset
from dataset import indexing_labels


class FocalLoss(Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss

# We define the transformations to be applied to our datasets
# Orignial images have different shapes, we resize them to square images.
# For the training we rotate some of the images randomly to make the learning more robust

transforms_train = Compose([
    Resize((224, 224)), 
    ToTensor(), #this converts numpy or Pil image to torch tensor and normalizes it in 0, 1
    RandomAffine((0.05, 0.05)),
    RandomHorizontalFlip(),
    RandomVerticalFlip()
])

transforms = Compose([
    Resize((224, 224)),
    ToTensor()
])


def network_training(class_weights, train_dataloader, valid_dataloader, model=ResNet50(), opt=Adam, loss_fn=CrossEntropyLoss(), epochs=30, pretrained=None, viewpoint=0, volume_dir='/mnt/shared_volume/'):
    '''
    viewpoint=0: all images
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    
    opt = opt(model.parameters(), lr=1e-3, weight_decay = 0.00005)
    
    if loss_fn == CrossEntropyLoss():
        loss_fn = loss_fn(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    elif loss_fn == FocalLoss():
        loss_fn = loss_fn(alpha=torch.tensor(class_weights, dtype=torch.float32).to(device), gamma=torch.tensor(2).to(device))

    if pretrained:
        net_state_dict = torch.load(volume_dir+'model.pt', map_location=device)
        model.load_state_dict(net_state_dict)
        
        opt_state_dict = torch.load(volume_dir+'optimizer_state.torch', map_location=device)
        opt.load_state_dict(opt_state_dict)
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
        best_val = np.loadtxt(volume_dir+'best_val.txt')
        train_loss_log_previous = np.loadtxt(volume_dir+'train_loss_log.txt').tolist()
        val_loss_log_previous = np.loadtxt(volume_dir+'val_loss_log.txt').tolist()

    else:
        best_val = np.inf # threshold for validation loss to choose if to save models parameters
        train_loss_log_previous =[]
        val_loss_log_previous = []
    
    train_loss_log = []
    val_loss_log = []
    
    model.to(device)

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        
        ### TRAINING ###
        model.train()
        train_loss= [] 
        iterator = tqdm(train_dataloader) 
        for batch_x, batch_y in iterator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
    
            y_pred = model(batch_x) # forward pass
    
            loss = loss_fn(y_pred, batch_y) # compute loss for this batch
    
            opt.zero_grad() # reset gradients
            loss.backward() # backpropagation
            
            opt.step() # update the weights
            
            loss_batch = loss.detach().cpu().numpy()
            train_loss.append(loss_batch)
            
            iterator.set_description(f"Train loss: {loss_batch:.4f} - Acc: {((y_pred.argmax(dim=1) == batch_y).float().mean()*100):.1f}")
        # store average loss for this epoch
        train_loss = np.mean(train_loss) #mean over all batches
        train_loss_log.append(train_loss)

        ### EVALUATION ON VALIDATION SET ###
        model.eval()
        val_loss = []
        with torch.no_grad():
            predictions = []
            true = []
            for batch_x, batch_y in tqdm(valid_dataloader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
    
                y_pred = model(batch_x) # forward pass

                loss = loss_fn(y_pred, batch_y)
                loss_batch = loss.detach().cpu().numpy()
                val_loss.append(loss_batch)
    
                # Storing predicted labels and true labels for this batch, to compute the validation accuracy later 
                predictions.append(y_pred) 
                true.append(batch_y)
                
            predictions = torch.cat(predictions, axis=0) # concatenation along batch dimension
            true = torch.cat(true, axis=0) # concatenation along batch dimension
            val_acc = (predictions.argmax(dim=1) == true).float().mean() # picks the class with the highest logit (predicted class) and compares it with the true one 
            
            # store average loss for this epoch
            val_loss = np.mean(val_loss)
            val_loss_log.append(val_loss)             
            print(f"loss: {val_loss}, accuracy: {val_acc}")
    
        if val_loss < best_val:
                print("Saved Model")
                torch.save(model.state_dict(), volume_dir+"model.pt") #saves model learned parameters in /mnt/shared_volume
                torch.save(opt.state_dict(), volume_dir+'optimizer_state.torch') #saves optimizer state in /mnt/shared_volume
                best_val = val_loss
                best_epoch = epoch 

    val_loss_log_previous.extend(val_loss_log[:best_epoch+1])
    train_loss_log_previous.extend(train_loss_log[:best_epoch+1])
    np.savetxt(volume_dir+'val_loss_log.txt', val_loss_log_previous)
    np.savetxt(volume_dir+'train_loss_log.txt', train_loss_log_previous)
    np.savetxt(volume_dir+'best_val.txt', [best_val])
    
    #return
    if viewpoint != 0:  return  model
    else: return train_loss_log, val_loss_log




def evaluate_network(dataloader, model, kk, dataset_name):
    """
    dataloader: the  dataloader class associated to a specific dataset
    model: the network to use to compute predictions
    dataset_name: string, maybe useless
    kk: int or list, the k for the top-k classes

    Return the top-k accuracy (percentage)
    Note: if a list is passed for kk, then the output is a list with the corresponding top  k accuracies
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = []
        true = []
        i=0
        for batch_x, batch_y in tqdm(dataloader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
    
            y_pred = model(batch_x)
            predictions.append(y_pred)
            true.append(batch_y)
        predictions = torch.cat(predictions, axis=0)
        true = torch.cat(true, axis=0)
        
        if isinstance(kk, int) or isinstance(kk, float): #check if kk argument is int or float. If so, convert to single entry list (needed to the for cycle)
            kk = [kk]
        print(dataset_name, "results:")
        topk_accuracies = {}
        for k_classes in kk:
            ### Calcolo errore top-k  
            predictions_after_activation = torch.nn.functional.softmax(predictions, dim=1)
            topk_predictions = torch.topk(predictions_after_activation, k_classes)[1] #2d tensor containg one array for each sample, with the top k predicted labels 
            
            true_expanded = true.unsqueeze(1) #now is a 2d tensor with 1d arrays as entries
    
            
            matches = (topk_predictions == true_expanded) # 2d tensor whose entries are arrays of boolean values: true if the entry is equal to the true label
            result = matches.any(dim=1) #1d array: it makes the boolean arrays collapse: become True if at least 1 true entry was there, else False
            
            topk_accuracy  = result.sum()/result.shape[0] #nr of times the true label was in the top k classes / number of samples considered
            topk_accuracy = topk_accuracy.detach().cpu().numpy()

            #print and store
            print(f"\ttop-{k_classes} accuracy", np.round(topk_accuracy,3))      
            topk_accuracies[k_classes] = topk_accuracy
        
    return topk_accuracies 





def multi_viewpoint_training(epochs_model_vp, model=ResNet50(), chosen_viewpoints=None, k_list=[1,5], volume_dir='/mnt/shared_volume/'):
    """
    This function performs sequentially a training of a fixed number of epochs on different datasets. The different datasets contains different
    viewpoints of the cars. It saves the trained model for each dataset.

    Possible update: train till convergence
    Possible update: start from pretrained model
    viewpoint:
    -1 - uncertain
    1 - front
    2 - rear
    3 - side
    4 - front-side
    5 - rear-side
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vp_to_name = {1: 'front', 2: 'rear', 3:'side', 4:'front-side', 5:'rear-side', None: 'all'}
    label_to_index = indexing_labels(volume_dir + "data/train_test_split/classification/train.txt", label_type='model_id')

    if chosen_viewpoints is None: #here you go sequentially for all viewpoints datasets available
        viewpoints_considered = [1,2,3,4,5]
    elif chosen_viewpoints is not None: #here you go sequentially on the specific viewpoints datasets provided as arg 
        viewpoints_considered = chosen_viewpoints
    
    for vp in viewpoints_considered:
      # LOAD DATASET
      train_dataset =  ImageDataset(volume_dir +"data", volume_dir + "data/train_test_split/classification/train.txt", label_to_index, transforms_train, viewpoint=vp)
      test_dataset =  ImageDataset(volume_dir +"data", volume_dir + "data/train_test_split/classification/test_updated.txt", label_to_index, transforms, viewpoint=vp)
      valid_dataset =  ImageDataset(volume_dir +"data", volume_dir + "data/train_test_split/classification/valid.txt", label_to_index, transforms, viewpoint=vp)
      print(f"len viewpoint {vp}({vp_to_name[vp]}) dataset:\n \ttrain: {len(train_dataset)}, valid: {len(valid_dataset)}, test: {len(test_dataset)}")
      # Here we use the Dataloader function from pytorch to opportunely split the dataset in batches and shuffling data
      batch_size = 64
      train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=os.cpu_count())
      valid_dataloader = DataLoader(valid_dataset, os.cpu_count()*2, shuffle=False, num_workers=os.cpu_count())
      test_dataloader = DataLoader(test_dataset, os.cpu_count()*2, shuffle=False, num_workers=os.cpu_count())
        
      #class weights
      labels_array = []
      for batch in train_dataloader:
            _, labels = batch
            labels_array.append(labels)
      labels_array = torch.cat(labels_array).numpy().astype(int)
      class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_array), y=labels_array)
              
      # TRAIN NETWORK AND SAVE PARAMETERS
      model=network_training(class_weights, train_dataloader, valid_dataloader, model, viewpoint=vp, epochs=epochs_model_vp)

      # EVALUATE NETWORK
      topk_accuracy_test = evaluate_network(test_dataloader, model, k_list, f"Test Dataset viewpoint{vp}")
      np.savetxt(volume_dir+f'table3/topk_accuracies_vp{vp}.txt', np.array(list(topk_accuracy_test.items()))[:,1])   
      print()






def make_training(epochs_make=30, model=ResNet50(), chosen_viewpoints=None, k_list=[1], volume_dir='/mnt/shared_volume/'):
    """
    This function performs sequentially a training of a fixed number of epochs on different datasets. The different datasets contains different
    viewpoints of the cars. It saves the trained model for each dataset.

    Possible update: train till convergence
    Possible update: start from pretrained model
    viewpoint:
    -1 - uncertain
    1 - front
    2 - rear
    3 - side
    4 - front-side
    5 - rear-side
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vp_to_name = {1 : 'front', 2: 'rear', 3:'side', 4:'front-side', 5:'rear-side'}
    label_to_index_make = indexing_labels(volume_dir + "data/train_test_split/classification/train.txt", label_type='make_id')

    if chosen_viewpoints is None: #here you go sequentially for all viewpoints datasets available
        viewpoints_considered = [1,2,3,4,5]
    elif chosen_viewpoints is not None: #here you go sequentially on the specific viewpoints datasets provided as arg 
        viewpoints_considered = chosen_viewpoints
    
    for vp in viewpoints_considered:
      # LOAD DATASET
      train_dataset_make =  ImageDataset(volume_dir +"data", volume_dir + "data/train_test_split/classification/train.txt", label_to_index_make, transforms_train, vp, label_type='make_id')
      test_dataset_make =  ImageDataset(volume_dir +"data", volume_dir + "data/train_test_split/classification/test_updated.txt", label_to_index_make, transforms, vp, label_type='make_id')
      valid_dataset_make =  ImageDataset(volume_dir +"data", volume_dir + "data/train_test_split/classification/valid.txt", label_to_index_make, transforms, vp, label_type='make_id')
      print(f"len viewpoint {vp}({vp_to_name[vp]}) dataset:\n \ttrain: {len(train_dataset_make)}, valid: {len(valid_dataset_make)}, test: {len(test_dataset_make)}")
    # Here we use the Dataloader function from pytorch to opportunely split the dataset in batches and shuffling data
      batch_size = 64
      train_dataloader = DataLoader(train_dataset_make, batch_size, shuffle=True, num_workers=os.cpu_count())
      valid_dataloader = DataLoader(test_dataset_make, os.cpu_count()*2, shuffle=False, num_workers=os.cpu_count())
      test_dataloader = DataLoader(valid_dataset_make, os.cpu_count()*2, shuffle=False, num_workers=os.cpu_count())

      #class weights
      labels_array = []
      for batch in train_dataloader:
            _, labels = batch
            labels_array.append(labels)
      labels_array = torch.cat(labels_array).numpy().astype(int)
      class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_array), y=labels_array)
        
      # TRAIN NETWORK AND SAVE PARAMETERS
      model=network_training(class_weights, train_dataloader, valid_dataloader, model, viewpoint=vp, epochs=epochs_make)

      # EVALUATE NETWORK
      topk_accuracy_test = evaluate_network(test_dataloader, model, k_list, f"Test Dataset viewpoint{vp}") #top-1 accuracy is the accuracy, right? CONTROLLA
      np.savetxt(volume_dir+f'table3/accuracy_vp{vp}_make.txt', np.array(list(topk_accuracy_test.items()))[:,1])  
      print()