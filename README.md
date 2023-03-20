# Bayesian image recovery using Sinusoidal Representation Networks

Pytorch implementation for performing Bayesian image restoration with sinusoidal representation neural networks (SIRENs).


### Problem formulation
---
Our starting point will be the following observation model:

$$
    g(x) = f(x) + \eta(x)
$$

for $x \in \mathcal{D} \subset \mathbb{Z}^2$. Here $g$ is our observed and corrupted image, $f$ is the original image we would like to recover and $\eta \sim \mathcal{N}(0, \sigma^2)$ is Gaussian noise. Here $\mathcal{D}$ is our discretized image domain. We also assume that all samples taken from our contaminating Gaussian distribution are i.i.d.

Here we will find an approximation to $f$ using a Sinusoidal Representation Network $f_{\theta}$ optimizing the following cost functions:

1. First order quadratic regularization loss

$$
    L(f_{\theta}) = ||{f_{\theta}  - g}||^2 +\lambda ||{\nabla f_{\theta}}||^2
$$

2. Robust regularization using total variation

$$
    L(f_{\theta}) = ||{f_{\theta}  - g}||^2 +\lambda ||{\nabla f_{\theta}}||
$$

Here, $\lambda$ is a hyperparameter balancing the importance of the regularization and data terms. We will use the ubiquitous cameraman image, apply Gaussian noise to it and see how both approaches perform on recovering the original image.
