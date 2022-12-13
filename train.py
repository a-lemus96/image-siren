import os

from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import  Resize, Compose, ToTensor, Normalize
from typing import Tuple

from dataload import *
from loss import *
from models import *


# HYPERPARAMETERS

filepath = 'data/cameraman.png'
total_steps = 500
steps_til_summary = 10
lrate = 1e-4
lambda_ = 1e-4


# Instantiate dataset and SIREN model
cameraman = ImageDataset(filepath)
siren = Siren(in_features=2, out_features=1, hidden_features=256,
              hidden_layers=3, outermost_linear=True)
# Send to GPU
siren.cuda()


# TRAINING LOOP

def train(
        dataset,
        model,
        lrate: float = 1e-4,
        total_steps: int = 500,
        steps_til_summary: int = 10) -> None:
    r'''Training loop to fit an image using Siren model. This function uses ADAM
    optimizer with all the default parameters except for learning rate. It also
    assumes the whole image as the batch size
    ---------------------------------------------------------------------------
    Args:
        dataset: Image dataset with ground truth pixel values and coordinates.
        model: Siren model to be optimized.
        lrate: Learning rate.
        total_steps: Total number of iterations.
        steps_til_summary: Display rate for optimization procedure.
    Returns:
        None'''
    # Create ADAM optimizer instance
    optimizer = torch.optim.Adam(lr=lrate, params=model.parameters())

    # Create DataLoader instance
    dataloader = DataLoader(dataset, batch_size=len(dataset),
                            pin_memory=True, num_workers=0)

    # Retrieve whole dataset
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = (model_input[None, ...].cuda(),
                                 ground_truth[None, ...].cuda())

    for step in range(total_steps):
        # Compute model outputs
        model_output, coords = model(model_input)
        # Compute loss
        loss = mse_loss(model_output, coords,  ground_truth, lambda_)
        # Display optimization process info
        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            img_grad = gradient(model_output, coords)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(ground_truth.cpu().view(256, 256).detach().numpy(),
                           cmap='gray')
            axes[0].set_title('Original')
            axes[1].imshow(model_output.cpu().view(256, 256).detach().numpy(),
                           cmap='gray')
            axes[1].set_title(f'Model output: Iteration {step}')
            axes[2].imshow(img_grad.norm(dim=-1).cpu().view(256, 256).detach().numpy(),
                                       cmap='gray')
            axes[2].set_title(f'Output gradient: Iteration {step}')
            plt.savefig(f'out/iteration_{step}.png')
            plt.close()

        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
train(dataset=cameraman, model=siren) 
