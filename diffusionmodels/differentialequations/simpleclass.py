"""
diffusionmodels.differentialequations.simpleclass
=================================================

Defines various simple stochastic differential equations.

Classes
-------
StandardOU
    An implementation of the standard Ornstein-Uhlenbeck process
"""


import torch

from .baseclass import StochasticDifferentialEquation

from ..utilities import unused_variables
from ..manifolds import Manifold


class StandardOU(StochasticDifferentialEquation):
    """
    A class of stochastic differential equations with the following:
        - drift(X, t) = -speed * X
        - diffusion(X, t) = volatility * N(0, I)

    Attributes
    ----------
    manifold: Manifold
        Provides the manifold structures where the SDE lives

    speed : float
        Determines how fast X decays to zero

    volatility : float
        Determines how much noise is added to the process
    """

    def __init__(self, manifold: Manifold, speed: float, volatility: float) -> None:
        super().__init__(manifold)
        self.speed = speed
        self.volatility = volatility


    def drift(self, X: torch.Tensor, t: float) -> torch.Tensor:
        """
        Evaluate the drift tensor of a standard OU process

        Parameters
        ----------
        X : torch.Tensor
            The spatial location to evaluate

        t : float
            Unused to compute drift of a standard OU process

        Returns
        -------
        torch.Tensor
            drift = -speed * X
        """

        # -- Acknowledge unused variables
        unused_variables(t)

        return -self.speed * X


    def diffusion(self, X: torch.Tensor, t: float) -> torch.Tensor:
        """
        Evaluate the diffusion tensor of a standard OU process

        Parameters
        ----------
        X : torch.Tensor
            Unused to compute diffusion of a standard OU process

        t : float
            Unused to compute diffusion of a standard OU process

        Returns
        -------
        torch.Tensor
            diffusion = volatility * N(0, I)
        """

        # -- Acknowledge unused variables
        unused_variables(X, t)

        return self.volatility * torch.normal(
            mean = torch.zeros(X.shape, device = X.device), std = 1.0
        )


class ExplodingRotationVariance(StochasticDifferentialEquation):

    def __init__(self, manifold: Manifold) -> None:
        super().__init__(manifold)


    def drift(self, X, t):

        unused_variables(X, t)

        return 0.0

    def diffusion(self, X, t):
        num_samples, num_dimension, num_row, num_col = X.shape
        unused_variables(t, num_row, num_col)
        return torch.randn(num_samples, num_dimension, 3, device = "cuda")
