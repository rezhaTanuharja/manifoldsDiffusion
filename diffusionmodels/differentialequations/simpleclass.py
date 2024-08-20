"""
diffusionmodels.differentialequations.simpleclass
=================================================

This module defines the simple classes for differential_equations.

Classes
-------
StandardOU
    An implementation of the standard Ornstein-Uhlenbeck process
"""

import torch

from ..utilities import unused_variables
from .baseclass import StochasticDifferentialEquation



class StandardOU(StochasticDifferentialEquation):
    """
    A class of stochastic differential equations with the following:
        - drift(X, t) = -speed * X
        - diffusion(X, t) = volatility * N(0, I)

    Parameters
    ----------
    speed : float
        Determines how fast X decays to zero

    volatility : float
        Determines how much noise is added to the process
    """

    def __init__(self, speed: float, volatility: float) -> None:
        super().__init__()
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
            mean = torch.zeros(X.shape), std = 1.0
        )
