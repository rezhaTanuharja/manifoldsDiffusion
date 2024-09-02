"""
diffusionmodels.timeintegrators.baseclass
=========================================

Provides the interface for all time integrators

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
    `step_forward(stochastic_de, X, t, dt)`
        Returns X(t + dt)
    """


    @abstractmethod
    def step_forward(
        self,
        stochastic_de: StochasticDifferentialEquation,
        X: torch.Tensor,
        t: float, dt: float
    ) -> torch.Tensor:
        """
        Evaluate X(t + dt) from a given stochastic differential equation

        Parameters
        ----------
        `stochastic_de: StochasticDifferentialEquation`
            A stochastic differential equation

        `X : torch.Tensor`
            The value of X at present time, i.e., X(t)

        `t : float`
            The current time

        `dt: float`
            The time increment or the temporal step

        Returns
        -------
        `torch.Tensor`
            the predicted value of X(t + dt)
        """
        raise NotImplementedError("Subclasses must implement this method")
