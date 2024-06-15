# Hopfield Model with Belief Propagation

This repository contains the code for the estimation of the radius of the Basins of Attraction of the Hopfield Model using a Belief Propagation approach.

## Hopfield model
The Hopfield model is a type of recurrent neural network used for storing and retrieving patterns, labeled as $\xi$ of a specific dimension $N$. It consists of a set of neurons interconnected in a fully connected manner, where each neuron can be in an active or inactive states, that is, $\xi = [-1, +1]^{N}$.
Patterns are stored by building a coupling matrix $J$ following the Hebb's rule

$$
J_{ij} = (1 - \delta_{ij})\frac{1}{N} \sum_{\mu = 1}^{M} \xi_i^\mu \xi_j^\mu.
$$

After storing patterns, a perturbed configuration can be restored to its original state through energy minimization

$$
E(\mathbf{\sigma}) = - \frac{1}{2} \sum_{i, j} J_{ij} \sigma_i \sigma_j.
$$

> TODO: Add details about Belief Propagation


