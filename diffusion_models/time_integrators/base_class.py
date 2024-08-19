"""
diffusion_models.time_integrators.base_class
============================================

This module defines the abstract base classes for time_integrators

Classes
-------
first_order
    Time integration method that only requires values at a single timestamp
"""


from abc import ABC, abstractmethod
from ..differential_equations import stochastic_differential_equation

import torch


class first_order(ABC):
    """
    An abstract class of first-order time integrations.

    Methods
    -------
    step_forward(differential_equation, X, t, dt)
        Returns X(t + dt)
    """


    def __init__(self) -> None:
        pass


    @abstractmethod
    def step_forward(
        self,
        differential_equation: stochastic_differential_equation,
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
            The value of X but not necessarily the present value

        t : float
            The current time

        dt: float
            The time increment or the temporal step

        Returns
        -------
        torch.Tensor
            the predicted value of X(t + dt)
        """
        raise NotImplementedError("Subclasses must implement this method")
