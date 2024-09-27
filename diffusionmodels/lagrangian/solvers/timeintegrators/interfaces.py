"""
diffusionmodels.processes.lagrangian.solvers.timeintegrators.interfaces
=======================================================================

Provides the interface for all time integrators

Classes
-------
TimeIntegrator
    Time integration method that only requires values at the present and past time
"""


from abc import ABC, abstractmethod
from typing import Union, List
from ...differentialequations import StochasticDifferentialEquation

import torch


class TimeIntegrator(ABC):
    """
    An abstract class of a time-integration

    Methods
    -------
    `to(device)`
        Moves any tensor attributes to device

    `step_forward(stochastic_de, X, t, dt)`
        Returns X(t + dt)
    """


    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def to(self, device: torch.device) -> None:
        """
        Move any tensor attributes to device

        Parameters
        ----------
        `device: torch.device`
            A device object from PyTorch
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def step_forward(
        self,
        stochastic_de: StochasticDifferentialEquation,
        X: Union[torch.Tensor, List[torch.Tensor]],
        t: Union[float, List[float]],
        dt: float
    ) -> torch.Tensor:
        """
        Evaluate X(t + dt) from a given stochastic differential equation

        Parameters
        ----------
        `stochastic_de: StochasticDifferentialEquation`
            A stochastic differential equation

        `X : torch.Tensor | List[torch.Tensor]`
            The value of X at present (and possibly past) time

        `t : float | List[float]`
            The current (and possibly past) time

        `dt: float | List[float]`
            The time increment or the temporal step

        Returns
        -------
        `torch.Tensor`
            the predicted value of X(t + dt)
        """
        raise NotImplementedError("Subclasses must implement this method")
