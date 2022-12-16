import os
from typing import Any, Callable, List, Tuple

from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import  Resize, Compose, ToTensor, Normalize

from dataload import *
from loss import *
from models import *


# HYPERPARAMETERS

filepath = 'data/cameraman.png'
total_steps = 500
steps_til_summary = 10
lrate = 1e-4

# Instantiate dataset and SIREN model
cameraman = ImageDataset(filepath)
siren = Siren(in_features=2, out_features=1, hidden_features=256,
              hidden_layers=3, outermost_linear=True)
# Send to GPU
siren.cuda('cuda:1')

# TRAINING LOOP

def train(
        dataset,
        model,
        loss_fn: Callable,
        out_dir: str,
        sigma: float = 0.,
        lambda_: float = 0.1,
        lrate: float = 1e-4,
        total_steps: int = 500,
        steps_til_summary: int = 10) -> Tuple[float, torch.Tensor, torch.Tensor]:
    r'''Training loop to fit an image using Siren model. This function uses ADAM
    optimizer with all the default parameters except for learning rate. It also
    assumes the whole image as the batch size
    ---------------------------------------------------------------------------
    Args:
        dataset: Image dataset with ground truth pixel values and coordinates.
        model: Siren model to be optimized.
        loss_fn: Loss function to be used during optimization.
        out_dir: Path to store results.
        sigma: Standard deviation for Gaussian noise distribution.
        lambda_: Hyperparameter balancing regularization term.
        lrate: Learning rate.
        total_steps: Total number of iterations.
        steps_til_summary: Display rate for optimization procedure.
    Returns:
        loss_vals: (total_steps)-length list containing loss values during 
                   training process.
        otuput: (sidelenght, sidelength, 1)-shape numpy array 
                containing model output after total_steps.
        grads: (sidelength, sidelength, 1)-shape numpy array containing
               output gradient norm after total_steps.'''
    # Create folder to store results
    try:
        os.makedirs(os.path.join(f'out/{out_dir}/'))
    except:
        pass

    # Apply noise to original image in dataset
    dataset.apply_noise(sigma)

    # Create ADAM optimizer instance
    optimizer = torch.optim.Adam(lr=lrate, params=model.parameters())

    # Create DataLoader instance
    dataloader = DataLoader(dataset, batch_size=len(dataset),
                            pin_memory=True, num_workers=0)

    # Retrieve whole dataset
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = (model_input[None, ...].cuda('cuda:1'),
            ground_truth[None, ...].cuda('cuda:1'))

    # Initialize list to hold loss values
    loss_vals = []
    
    for step in range(total_steps):
        # Compute model outputs
        model_output, coords = model(model_input)
        # Compute loss
        loss = loss_fn(model_output, coords,  ground_truth, lambda_)
        # Save loss
        loss_vals.append(loss.item())
        # Display optimization process info
        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            img_grad = gradient(model_output, coords)

            # Prepare model output and gradients for display
            output = model_output.cpu().view(256, 256).detach().numpy()
            grads = img_grad.norm(dim=-1).cpu().view(256, 256).detach().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(ground_truth.cpu().view(256, 256).detach().numpy(),
                           cmap='gray')
            axes[0].set_title('Original')
            axes[1].imshow(output, cmap='gray')
            axes[1].set_title(f'Model output: Iteration {step}')
            axes[2].imshow(grads, cmap='gray')
            axes[2].set_title(f'Output gradient: Iteration {step}')
            plt.savefig(f'out/{out_dir}/iteration_{step}.png')
            plt.close()

        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_vals, output, grads
        
# Fit SIREN using MSE loss
mse_vals, mse_out, mse_grads = train(dataset=cameraman, model=siren,
                               loss_fn=mse_loss, out_dir='MSE',
                               lambda_=5e-4, sigma=0.2)

# Fit SIREN usin TV loss
tv_vals, tv_out, tv_grads = train(dataset=cameraman, model=siren,
                            loss_fn=tv_loss, out_dir='TV',
                            lambda_=5e-3, sigma=0.2)

# Plot final results
fig, axes = plt.subplots(2, 3, figsize=(23, 8),
                         gridspec_kw={'width_ratios': [1, 1, 3]})
axes[0,0].imshow(mse_out, cmap='gray')
axes[0,0].set_title('Model Output - MSE (500 steps)')

axes[0,1].imshow(mse_grads, cmap='gray')
axes[0,1].set_title('Output Gradient - MSE (500 steps)')

axes[0,2].plot(mse_vals)
axes[0,2].set_title('Loss values - MSE')

axes[1,0].imshow(tv_out, cmap='gray')
axes[1,0].set_title('Model Output - TV (500 steps)')

axes[1,1].imshow(tv_grads, cmap='gray')
axes[1,1].set_title('Output Gradient - TV (500 steps)')

axes[1,2].plot(tv_vals)
axes[1,2].set_title('Loss values - TV')

plt.savefig('out/results.png')
plt.close()
