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
    A subclass of SDE that has no drift, i.e., in the form of

        `dX = diffusion(X, t) dW`

    Methods
    -------
    `manifold()`
        Provides access to the manifold the SDE lives in

    `drift(X, t)`
        Evaluate the drift tensor field at (X, t)

    `diffusion(X, t)`
        Evaluate the diffusion tensor field at (X, t)
    """


    def __init__(self, manifold: Manifold, variance_scale = float) -> None:
        self._manifold = manifold
        self._variance_scale = variance_scale


    def manifold(self) -> Manifold:
        return self._manifold


    def drift(self, X: torch.Tensor, t: float) -> torch.Tensor:

        # -- there is no drift so no spatial or temporal dependency
        unused_variables(X, t)

        # -- this will be broadcasted, do not waste memory
        return torch.tensor(0.0, device = X.device)


    def diffusion(self, X: torch.Tensor, t: float) -> torch.Tensor:

        # -- there is no temporal dependency
        unused_variables(t)

        num_samples, num_dimension = X.shape[:2]

        # -- diffusion is a multivariate Gaussian
        return self._variance_scale * torch.randn(
            num_samples, num_dimension, *self._manifold.tangent_dimension(),
            device = X.device
        )
