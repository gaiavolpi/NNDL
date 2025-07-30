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
import copy

from ResNet50_blocks import ResNet50
from utils import ImageDataset
from sklearn.utils.class_weight import compute_class_weight
from dataset import *
from torchvision.models import resnet18, ResNet18_Weights


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

class FocalLoss(Module):
    '''
    Implements Focal Loss for multi-class classification.
    '''
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # scalar or tensor of shape [num_classes]
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = CrossEntropyLoss(reduction='none')(inputs, targets)  # shape: [batch_size]
        pt = torch.exp(-ce_loss)

        if isinstance(self.alpha, torch.Tensor):
            # get per-sample alpha weight from per-class alpha
            alpha_t = self.alpha[targets]  # shape: [batch_size]
        else:
            alpha_t = self.alpha  # scalar

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def load_pretrain(volume_dir, device, model, opt=None, vp=0):
    '''
    Load a pre-trained model and optimizer state from the specified volume directory.
    '''
    vp = vp or '0' # if vp is None, we assume it is '0' 
    net_state_dict = torch.load(volume_dir+str(vp)+'/model.pt', map_location=device) # load the model state dict
    model.load_state_dict(net_state_dict)

    if opt is not None: # if an optimizer is provided, load its state dict
        opt_state_dict = torch.load(volume_dir+str(vp)+'/optimizer_state.torch', map_location=device)
        opt.load_state_dict(opt_state_dict)
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    best_val = np.loadtxt(volume_dir+str(vp)+'/best_val.txt')
    train_loss_log_previous = np.loadtxt(volume_dir+str(vp)+'/train_loss_log.txt').tolist()
    train_loss_log_previous.tolist() if np.ndim(train_loss_log_previous) else [float(train_loss_log_previous)]
    val_loss_log_previous = np.loadtxt(volume_dir+str(vp)+'/val_loss_log.txt').tolist()
    val_loss_log_previous.tolist() if np.ndim(val_loss_log_previous) else [float(val_loss_log_previous)]

    return model, opt, best_val, train_loss_log_previous, val_loss_log_previous

def fix_losses(class_weights, loss_fn, device):
    '''
    This function fixes the loss function to be used in the training.
    '''
    if loss_fn == CrossEntropyLoss:
        loss_fn = loss_fn(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    elif loss_fn == FocalLoss:
        loss_fn = loss_fn(alpha=torch.tensor(class_weights, dtype=torch.float32).to(device), gamma=torch.tensor(2).to(device))
    else:
        raise NotImplementedError("Loss function not implemented. Use CrossEntropyLoss or FocalLoss.")
    return loss_fn

def load_checkpoint(load, model, opt, device, volume_dir):
    '''
    This function returns a function that loads a pre-trained model and optimizer state from the specified volume
        directory if load is True, otherwise it initializes them. 
    '''
    def load_():
        return load_pretrain(volume_dir, device, model, opt) if load else (model, opt, np.inf, [], [])
    return load_

def save_checkpoint(model, opt, vp, volume_dir):
    '''
    This function saves the model and optimizer state to the specified volume directory.
    '''
    torch.save(model.state_dict(), volume_dir+str(vp)+"/model.pt")  
    torch.save(opt.state_dict(), volume_dir+str(vp)+'/optimizer_state.torch')
    print("Model and optimizer state saved.")

def store_metric(volume_dir):
    '''
    This function returns a function that saves the metric values to the specified volume directory.
    '''
    def save_(metric, values, vp): # questi argomenti sono passati da network_training
        np.savetxt(volume_dir+str(vp or "0")+"/"+metric, values)
    return save_

def network_training(train_dataloader, valid_dataloader, load_checkpoint, loss_fn, device, save_checkpoint, save_metric, epochs=30, patience=5, skip_first=20):
    '''
    This function trains the model on the training dataset and evaluates it on the validation dataset.
    '''
    model, opt, best_val, train_loss_log_previous, val_loss_log_previous = load_checkpoint() # load the model and optimizer state if load_checkpoint is True

    train_loss_log = []
    val_loss_log = []
    val_acc_log = []
    
    model.to(device)

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        
        ### TRAINING ###
        model.train()
        train_loss= [] 
        iterator = tqdm(train_dataloader) 
        accuracies = []
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

            accuracies.append(
                ((y_pred.argmax(dim=1) == batch_y).float().mean()*100).detach().cpu().numpy() # compute accuracy for this batch
            )
            
            iterator.set_description(f"Train loss: {loss_batch:.4f} - Acc: {np.mean(accuracies):.1f}")

        # store average loss for this epoch
        train_loss = np.mean(train_loss)
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
            val_acc_log.append(val_acc.detach().cpu().numpy()) 
            # store average loss for this epoch
            val_loss = np.mean(val_loss)
            val_loss_log.append(val_loss)             
            print(f"loss: {val_loss}, accuracy: {val_acc}")
    
    
        if val_acc_log[-1] >= np.max(val_acc_log) - 1e-5: # never compare floating point numbers with ==, use a small epsilon value
            #tmp = type(model)().load_state_dict(model.state_dict()) # reset the model to the best state
            tmp = copy.deepcopy(model)
            best_val = np.max(val_acc_log)
            best_epoch = epoch 

        # early stopping 
        if epoch > (skip_first or 20):
            #if np.all(np.array(val_acc_log)[-patience:] < np.max(val_acc_log)): # STOP ON ACCURACY
                # print(f"Early stopping (by accuracy decrease) at epoch {epoch+1} with best validation loss: {best_val:.4f}")
                # break
            if np.all(np.array(val_loss_log)[-patience:] > np.min(val_loss_log)): # STOP ON LOSS
                print(f"Early stopping (by loss increase) at epoch {epoch+1} with best validation loss: {best_val:.4f}")
                break

    val_loss_log_previous.extend(val_loss_log[:best_epoch+1])
    train_loss_log_previous.extend(train_loss_log[:best_epoch+1])

    save_metric('val_loss_log.txt', val_loss_log_previous) # save the loss log
    save_metric('train_loss_log.txt', train_loss_log_previous)
    save_metric('best_val.txt', [best_val])

    return train_loss_log, val_loss_log, tmp

def evaluate_network(dataloader, model, k_indices:list, device):
    '''
    This function evaluates the model on the test dataset and computes the top-k accuracy.
    '''
    assert isinstance(k_indices, list), "k_indices should be a list of integers" # imposes that k_indices is a list, super figa questa cosa non la sapevo!

    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = []
        true_labels = []
        for batch_x, batch_y in tqdm(dataloader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
    
            y_pred = model(batch_x)
            predictions.append(y_pred)
            true_labels.append(batch_y)
        predictions = torch.cat(predictions, axis=0)
        true_labels = torch.cat(true_labels, axis=0)
    probs = torch.nn.functional.softmax(predictions, dim=1)

    topk_score = {}
    for k in k_indices:
        topk_preds = torch.topk(probs, k, dim=1).indices #2d tensor containg one array for each sample, with the top k predicted labels 
        correct = topk_preds.eq(true_labels.unsqueeze(1)).any(dim=1) # 2d tensor whose entries are arrays of boolean values: true if the entry is equal to the true label
        score = correct.float().mean().item()
        print(f"\ttop-{k} score: {score:.3f}")
        topk_score[k] = score

    return topk_score

def calcul_class_weights(train_dataloader, num_classes=431):
    '''
    This function computes the class weights for the training dataset.
    '''
    labels_array = []
    for batch in train_dataloader:
        _, labels = batch
        labels_array.append(labels)

    labels_array = torch.cat(labels_array).numpy().astype(int).reshape(-1, 1)
    bincount = (labels_array == np.arange(num_classes).reshape(1, -1)).sum(axis=0)

    return len(train_dataloader) / (bincount.astype(float) * num_classes + 1e-8)

def noop(*args, **kwargs):
    pass

def pretrained_training(train_dataloader, valid_dataloader, loss_fn, device, volume_dir, epochs_fc_train=10, epochs_finetune=50, num_classes=431):

    # load pretrained model from pytorch
    weights = ResNet18_Weights.DEFAULT
    model_fn = lambda: resnet18(weights)
    model = model_fn()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes) # Replace final fully connected layer to match your 431/75 classes
    print("Model loaded with pre-trained weights.")

    loss = fix_losses(calcul_class_weights(train_dataloader, num_classes), loss_fn, device)

    print("Training the final fully connected layer only...")

    for param in model.parameters():
        param.requires_grad = False  # freeze every layer
    for param in model.fc.parameters():
        param.requires_grad = True  # do not freeze the last one 
    # optimize just the parameters of the last layer
    opt = Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-5)

    model.to(device)

    train_loss_log = []
    val_loss_log = []
    val_acc_log = []

    for epoch in range(epochs_fc_train): 
        model.train() 
        train_loss= [] 
        accuracies = []
        iterator = tqdm(train_dataloader) 

        for batch_x, batch_y in iterator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x) 
            loss_result = loss(y_pred, batch_y)

            opt.zero_grad()
            loss_result.backward()
            opt.step()
            loss_batch = loss_result.detach().cpu().numpy()
            train_loss.append(loss_batch)

            accuracies.append(
                ((y_pred.argmax(dim=1) == batch_y).float().mean()*100).detach().cpu().numpy() # compute accuracy for this batch
            )

            iterator.set_description(f"Train loss: {loss_batch:.4f} - Acc: {np.mean(accuracies):.1f}")
        
        train_loss = np.mean(train_loss)
        train_loss_log.append(train_loss)

    print("Training the whole model with fine-tuning...")
    
    for param in model.parameters():
        param.requires_grad = True # unfreeze all the layers 

    # optimize all the parameters now 
    opt_finetune = Adam(model.parameters(), lr=1e-4)  

    train_loss_log, val_loss_log, best_model = network_training(
        train_dataloader=train_dataloader, 
        valid_dataloader=valid_dataloader,
        load_checkpoint=load_checkpoint(
            load=False, 
            model=model,
            opt=opt_finetune, 
            device=device, 
            volume_dir=volume_dir
        ),
        loss_fn=loss,
        device=device,
        epochs=epochs_finetune,
        save_checkpoint=noop,
        save_metric=noop,
        patience=5,
        skip_first=20
    )
    return train_loss_log, val_loss_log, best_model