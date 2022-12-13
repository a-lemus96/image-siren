import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple 


# SINE LAYER MODULE

class SineLayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            is_first: bool = False,
            omega_0: float = 30.) -> None:
        r'''Sine layer initialization method.
        ----------------------------------------------------------------------- 
            Args:
                in_features: Number of input features
                out_features: Number of output features
                bias: If True, use bias parameters
                is_first: If False, divide weights by omega_0 frequency factor
                omega_0: Frequency factor to apply at the first layer
            Returns:
                None'''
        super().__init__()
        self.omega_0 = omega_0
        self.in_features = in_features
        self.is_first = is_first

        # Create linear layer
        self.linear = nn.Linear(in_features, out_features, bias)

        # Initialize weights
        self.__init_weights()


    def __init_weights(self) -> None:
        r'''Initialize layer weights according to initialization scheme proposed
        by authors.
        ------------------------------------------------------------------------ 
        Returns:
            None'''
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1. / self.in_features,
                                            1. / self.in_features)
            else:
                self.linear.weight.uniform_((-np.sqrt(6 / self.in_features) /
                                             self.omega_0),
                                            (np.sqrt(6 / self.in_features) /
                                             self.omega_0))


    def forward(
            self,
            input: torch.Tensor) -> torch.Tensor:
        r'''Forward pass through sine layer.
        ------------------------------------------------------------------------
        Args:
            input: (batch_size, self.in_features)-shape tensor
        Returns:
            (batch_size, out_features)-shape tensor''' 
        return torch.sin(self.omega_0 * self.linear(input))


    def forward_with_intermediate(
            self,
            input: torch.Tensor) -> Tuple[torch.Tensor, ...]: 
        r'''Forward pass through sine layer with intermediate activation info.
        ------------------------------------------------------------------------
        Args:
            input: (batch_size, self.in_features)-shape tensor
        Returns:
            (batch_size, out_features)-shape tensor
            intermediate: (batch_size, out_features)-shape tensor containing the
                          arguments to the sine activation function''' 
        intermediate = self.omega_0 * self.linear(input)

        return torch.sin(intermediate), intermediate


# SIREN (Sinusoidal Representation Network) MODULE

class Siren(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            hidden_layers: int,
            out_features: int,
            outermost_linear: bool = False,
            first_omega_0: float = 30.,
            hidden_omega_0: float = 30.) -> None:
        r'''SIREN module initialization method.
        ----------------------------------------------------------------------- 
            Args:
                in_features: Number of input features
                hidden_features: Width of internal layers
                hidden_layers: Number of hidden layers
                out_features: Number of output features
                outermost_linear: If True, use linear activation at final layer
                first_omega_0: Omega_0 value to be used at first layer
                hidden_omega_0: Omega_0 value to be uset at hidden layers
            Returns:
                None'''
        super().__init__()

        self.net = [] # Ordered list containing model layers
        # Append first layer
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        # Append hidden layers
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        # Append last layer
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            # Initialize weights accordingly
            with torch.no_grad():
                final_linear.weight.uniform_((-np.sqrt(6 / hidden_features) /
                                             hidden_omega_0),
                                            (np.sqrt(6 / hidden_features) /
                                             hidden_omega_0))
            
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        # Create sequential model from layers in net list
        self.net = nn.Sequential(*self.net)


    def forward(
            self, 
            coords: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        r'''Forward pass method through SIREN module.
        -----------------------------------------------------------------------
        Args:
            coords: (batch_size, in_features)-shape tensor
        Returns:
            output: (batch_size, out_features)-shape tensor
            coords: Clone of original input tensor that requires gradient''' 
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)

        return output, coords
