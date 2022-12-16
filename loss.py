import os

from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import  Resize, Compose, ToTensor, Normalize
from typing import Tuple


# GRADIENT UTILITIES

def gradient(
        y: torch.Tensor, 
        x: torch.Tensor, 
        grad_outputs: torch.Tensor = None) -> torch.Tensor:
    r'''Computes gradient of tensor y with respect to tensor x.
    ---------------------------------------------------------------------------
    Args:
        y: output tensor.
        x: input tensor.
        grad_outputs: precomputed gradients w.r.t. each of the outputs.
    Returns:
        grad: gradient of y w.r.t. input x.'''
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)

    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs,
                               create_graph=True)
    grad = grad[0]

    return grad


# LOSS UTILITIES

def mse_loss(
        model_output: torch.Tensor, 
        coords: torch.Tensor,
        ground_truth: torch.Tensor,
        lambda_: float = .1) -> torch.Tensor:
    r'''Computes first order quadratic regularization loss function.
    ---------------------------------------------------------------------------
    Args:
        model_output: (batch_size, n_channels)-shape tensor.
        ground_truth: (batch_size, n_channels)-shape tensor.
        coords: (batch_size, input_dim)-shape tensor.
        lambda_: Regularization factor.
    Returns:
        loss: (1,)-shape tensor holding loss term scalar value.'''
    # Compute data fidelity term
    loss = ((model_output - ground_truth)**2).mean()
    # Compute gradients on the model outputs
    grads = gradient(model_output, coords) 
    # Compute and add gradient loss
    loss += lambda_ * torch.mean((grads**2).sum(-1))

    return loss

def tv_loss(
        model_output: torch.Tensor,
        coords: torch.Tensor,
        ground_truth: torch.Tensor,
        lambda_: float = .1) -> torch.Tensor:
    r'''Computes total variation regularization loss function.
    ---------------------------------------------------------------------------
    Args:
        model_output: (batch_size, n_channels)-shape tensor.
        ground_truth: (batch_size, n_channels)-shape tensor.
        coords: (batch_size, input_dim)-shape tensor.
        lambda_: Regularization factor.
    Returns:
        loss: (1,)-shape tensor holding loss term scalar value.'''
    # Compute data fidelity term
    loss = ((model_output - ground_truth)**2).mean()
    # Compute gradients on the model outputs
    grads = gradient(model_output, coords)
    # Compute and add TV loss
    loss += lambda_ * torch.mean(torch.abs(grads).sum(-1))

    return loss
