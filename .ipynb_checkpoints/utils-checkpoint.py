import numpy as np
import matplotlib.pyplot as plt

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