import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import  Resize, Compose, ToTensor, Normalize
from PIL import Image

from typing import Tuple


# DATA LOADING UTILITIES 

def get_mgrid(sidelength: int, dim: int = 2) -> torch.Tensor:
    r'''Generates a flattened tensor that represents a grid of spatio-temporal
    coordinates (x, y, ...) within the normalized range of (-1, 1).
    ---------------------------------------------------------------------------
    Args:
        sidelength: Image sidelenght. It is assumed that image is square.
        dim: Number of dimension for spatio-temporal coordinates.
    Returns:
        mgrid: (SxS, dim)-shape tensor containing flattened spatio-temporal
               coordinates.'''
    # Create tuple of (1,)-shape tensors for each dimension
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelength)])
    # Create meshgrid from tuple of tensor
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    # Flatten meshgrid
    mgrid = mgrid.reshape(-1, dim)

    return mgrid


def get_image_tensor(
        img_path: str,
        sidelength: int = 256) -> torch.Tensor:
    r'''Opens image from file path, applies transformations and returns it as a
    tensor.
    --------------------------------------------------------------------------- 
    Args:
        img_path: file location.
        sidelength: Square dimension. Image dims are resized to be equal.
    Returns:
        img: (S, S, N)-shape tensor. S stands for sidelenght parameter.'''
    # Open file 
    img = Image.open(img_path) 
    # Compose and apply transforms
    n_channels = 1 if img.mode == 'L' else 3
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(0.5 * torch.ones(n_channels), 0.5 * torch.ones(n_channels))
    ])
    img = transform(img)

    return img


# IMAGE DATASET

class ImageDataset(Dataset):
    r'''Dataset consisting of image coordinates (x, y) and their corresponding
    intensity values f(x, y) for grayscale images and RGB values (r, g, b) for
    color images.'''
    def __init__(
            self,
            img_path: str,
            sidelength: int = 256) -> None:
        r'''Dataset initialization method.
        -----------------------------------------------------------------------
        Args:
            img_path: file location.
            sidelength: Square dimension. Image dims are resized to be equal.
        Returns:
            None'''
        # Base class init method call
        super().__init__()

        # Get image as tensor
        img = get_image_tensor(img_path, sidelength=sidelength)
        self.n_channels = img.shape[0]
        self.pixels = img.permute(1, 2, 0).view(-1, self.n_channels) 
        # Get flattened grid of coordinates
        self.coords = get_mgrid(sidelength, 2)


    def __len__(self):
        r'''Internal method to retrieve length of the dataset.
        -----------------------------------------------------------------------
        Returns:
            number of training samples.'''

        return self.pixels.shape[0]


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        r'''Internal method to acces randomly to an element of the dataset
        according to an index parameter.
        -----------------------------------------------------------------------
        Args:
            idx: Element index of interest.
        Returns:
            Tuple of tensors containing element coords and pixel values.'''
        
        return self.coords[idx], self.pixels[idx]
