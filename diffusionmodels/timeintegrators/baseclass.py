"""
diffusionmodels.timeintegrators.baseclass
=========================================

Defines the abstract base classes for time integrators

Classes
-------
FirstOrder
    Time integration method that only requires values at a single timestamp
"""


from abc import ABC, abstractmethod
from ..differentialequations import StochasticDifferentialEquation

import torch


class FirstOrder(ABC):
    """
    An abstract class of first-order time integrations.

    Methods
    -------
    step_forward(differential_equation, X, t, dt)
        Returns X(t + dt)
    """


    @abstractmethod
    def step_forward(
        self,
        sde: StochasticDifferentialEquation,
        X: torch.Tensor,
        t: float,
        dt: float
    ) -> torch.Tensor:
        """
        Evaluate X(t + dt) from a given stochastic differential equation

        Parameters
        ----------
        sde: StochasticDifferentialEquation
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
