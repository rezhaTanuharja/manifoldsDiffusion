"""
diffusion_models.time_integrators.simple_class
==============================================

This module defines the simple classes for time_integrators

Classes
-------
Euler_Maruyama
    An implementation of an explicit time integrator
"""


import torch

from ..differential_equations import stochastic_differential_equation
from .base_class import explicit


class Euler_Maruyama(explicit):
    """
    An explicit time integrator in the form of
        X(t + dt) = X(t) + dX(X(t), t)

    Methods
    -------
    step_forward(differential_equation, X, t, dt)
        Returns X(t) + dX(X(t), t)
    """

    def __init__(self):
        pass

    def step_forward(
        self,
        sde: stochastic_differential_equation,
        X: torch.Tensor,
        t: float,
        dt: float
    ) -> torch.Tensor:
        """
        Evaluate X(t + dt) from a given stochastic differential equation

        Parameters
        ----------
        differential_equation: stochastic_differential_equation
            A differential equation in the form of
            dX = drift(X, t) dt + diffusion(X, t) dW

        X : torch.Tensor
            The current value, i.e., X(t)

        t : float
            The current time

        dt: float
            The time increment or the temporal step

        Returns
        -------
        torch.Tensor
            the predicted value of X(t + dt)
        """

        return X + dt * sde.drift(X, t) + torch.sqrt(torch.tensor(dt)) * sde.diffusion(X, t)
