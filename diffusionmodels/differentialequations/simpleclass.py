"""
differentialequations.simpleclass
=================================

Defines various simple stochastic differential equations.

Classes
-------
ExplodingVariance
    A subclass of SDE that has no drift
"""


import torch

from .baseclass import StochasticDifferentialEquation

from ..utilities import unused_variables
from ..manifolds import Manifold


class ExplodingVariance(StochasticDifferentialEquation):
    """
    A subclass of SDE that has no drift
    """


    def __init__(self, manifold: Manifold, variance_scale = float) -> None:
        self._manifold = manifold
        self._variance_scale = variance_scale


    def manifold(self) -> Manifold:
        return self._manifold


    def drift(self, X: torch.Tensor, t: float) -> torch.Tensor:
        unused_variables(X, t)
        return torch.tensor(0.0, device = X.device)


    def diffusion(self, X: torch.Tensor, t: float) -> torch.Tensor:
        unused_variables(t)
        num_samples, num_dimension = X.shape[:2]
        return self._variance_scale * torch.randn(
            num_samples, num_dimension, *self._manifold.tangent_dimension(),
            device = X.device
        )
